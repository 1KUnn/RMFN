import math
from typing import Optional, Tuple, Dict, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:
    raise ImportError("You must install torch-geometric to use GAT blocks (`pip install torch-geometric`).")


class RoPEPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(2)

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos_emb = emb.cos()
        sin_emb = emb.sin()

        x_rot = self._apply_rotary_pos_emb(x, cos_emb, sin_emb)

        return x_rot

    def _apply_rotary_pos_emb(self, x, cos, sin):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        cos = cos[..., ::2]
        sin = sin[..., ::2]

        rotated_x1 = x1 * cos.unsqueeze(0).unsqueeze(0) - x2 * sin.unsqueeze(0).unsqueeze(0)
        rotated_x2 = x1 * sin.unsqueeze(0).unsqueeze(0) + x2 * cos.unsqueeze(0).unsqueeze(0)

        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

        return rotated_x


class TransformerFMRIEncoder(nn.Module):
    def __init__(
        self,
        n_rois=50,
        seq_len=200,
        d_model=128,
        n_heads=8,
        n_layers=4,
        n_decoder_scales=3,
        dropout=0.1
    ):
        super().__init__()
        self.n_rois = n_rois
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_decoder_scales = n_decoder_scales

        self.input_projection = nn.Linear(1, d_model)

        self.rope = RoPEPositionalEncoding(d_model, max_seq_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.scale_projectors = nn.ModuleList()
        for scale in range(n_decoder_scales):
            k_proj = nn.Linear(d_model, d_model)
            v_proj = nn.Linear(d_model, d_model)
            self.scale_projectors.append(nn.ModuleDict({
                'k_proj': k_proj,
                'v_proj': v_proj
            }))

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, bold_ts, bold_adj=None):
        if bold_ts.dim() == 2:
            bold_ts = bold_ts.unsqueeze(0)

        B, N, L = bold_ts.shape

        bold_reshaped = bold_ts.view(B * N, L, 1)

        x = self.input_projection(bold_reshaped)
        x = self.dropout(x)

        x = x.view(B, N, L, self.d_model)
        x = self.rope(x)
        x = x.view(B * N, L, self.d_model)

        encoded = self.transformer_encoder(x)

        roi_features = encoded.view(B, N, L, self.d_model)

        roi_pooled = roi_features.mean(dim=2)

        multi_scale_kv = []
        for scale_idx, projector in enumerate(self.scale_projectors):
            scale_features = roi_pooled

            k = projector['k_proj'](scale_features)
            v = projector['v_proj'](scale_features)

            multi_scale_kv.append({
                'keys': k,
                'values': v
            })

        global_features = roi_pooled.mean(dim=1)
        global_features = self.global_proj(global_features)

        return {
            'multi_scale_kv': multi_scale_kv,
            'global_features': global_features,
            'roi_features': roi_features,
            'roi_pooled': roi_pooled
        }


class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, prompt_dim):
        super().__init__()
        self.gamma_proj = nn.Linear(prompt_dim, feature_dim)
        self.beta_proj = nn.Linear(prompt_dim, feature_dim)

    def forward(self, x, prompt):
        gamma = self.gamma_proj(prompt)
        beta = self.beta_proj(prompt)

        for _ in range(x.dim() - 2):
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        return gamma * x + beta


class PromptAcceptanceModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, smoothing_factor=0.1):
        super().__init__()
        self.smoothing_factor = smoothing_factor

        self.quality_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.recon_quality_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.register_buffer('old_score_ema', torch.tensor(0.5))
        self.register_buffer('update_count', torch.tensor(0))

    def forward(self, features, recon_loss, training=True):
        features_transposed = features.transpose(1, 2)
        feature_quality = self.quality_net(features_transposed)

        if recon_loss.dim() == 0:
            recon_loss = recon_loss.unsqueeze(0)
        if recon_loss.dim() == 1 and len(recon_loss) == 1:
            recon_loss = recon_loss.expand(features.size(0))

        recon_loss_normalized = torch.clamp(-torch.log(recon_loss + 1e-8), 0, 10) / 10
        recon_quality = self.recon_quality_net(recon_loss_normalized.unsqueeze(-1))

        new_score = 0.7 * feature_quality + 0.3 * recon_quality
        new_score = new_score.squeeze(-1)

        if training:
            current_mean = new_score.mean().detach()
            if self.update_count == 0:
                self.old_score_ema.copy_(current_mean)
            else:
                self.old_score_ema.copy_(
                    self.smoothing_factor * current_mean +
                    (1 - self.smoothing_factor) * self.old_score_ema
                )
            self.update_count += 1

        old_score = self.old_score_ema.expand_as(new_score)

        epsilon = 1e-6
        ratio = (new_score + epsilon) / (old_score + epsilon)

        ratio_smooth = torch.tanh(ratio - 1) + 1

        log_ratio = torch.log(ratio_smooth + epsilon)

        return {
            'ratio': ratio_smooth,
            'log_ratio': log_ratio,
            'new_score': new_score,
            'old_score': old_score,
            'feature_quality': feature_quality.squeeze(-1),
            'recon_quality': recon_quality.squeeze(-1)
        }


class GATEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes=50, heads=4, dropout=0.2):
        super().__init__()
        self.node_encoder = nn.Linear(in_channels, out_channels)
        self.gat = GATv2Conv(
            in_channels=out_channels,
            out_channels=out_channels,
            heads=heads,
            concat=False,
            dropout=dropout
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.num_nodes = num_nodes

    def forward(self, x, edge_index):
        B, N, C = x.shape
        x = self.node_encoder(x)

        out = []
        for b in range(B):
            h = self.gat(x[b], edge_index)
            h = self.relu(h)
            h = self.dropout(h)
            out.append(h)
        x_out = torch.stack(out, dim=0)
        return x_out


class NodeCrossAttention(nn.Module):
    def __init__(self, feature_dim, fmri_dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.fmri_dim = fmri_dim
        self.n_heads = n_heads

        self.q_proj = nn.Linear(feature_dim, fmri_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fmri_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.output_proj = nn.Linear(fmri_dim, feature_dim)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, node_features, fmri_keys, fmri_values):
        queries = self.q_proj(node_features)

        attn_output, _ = self.cross_attention(
            query=queries,
            key=fmri_keys,
            value=fmri_values
        )

        attn_output = self.output_proj(attn_output)

        node_features = self.norm1(node_features + attn_output)

        ffn_output = self.ffn(node_features)
        node_features = self.norm2(node_features + ffn_output)

        return node_features


class GATDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fmri_dim, num_nodes=50, heads=4, use_cross_attention=True, dropout=0.2):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=False,
            dropout=dropout
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        if use_cross_attention:
            self.cross_attention = NodeCrossAttention(
                feature_dim=out_channels,
                fmri_dim=fmri_dim,
                n_heads=8,
                dropout=dropout
            )
        self.num_nodes = num_nodes

    def forward(self, x, edge_index, fmri_keys=None, fmri_values=None):
        out = []
        for b in range(x.shape[0]):
            h = self.gat(x[b], edge_index)
            h = self.relu(h)
            h = self.dropout(h)
            if self.use_cross_attention and fmri_keys is not None and fmri_values is not None:
                h = self.cross_attention(h, fmri_keys[b], fmri_values[b])
            out.append(h)
        x_out = torch.stack(out, dim=0)
        return x_out


class RMFN(nn.Module):
    def __init__(
        self,
        t1w_shape=(176, 176, 176),
        n_rois=50,
        seq_len=200,
        fmri_dim=128,
        n_encoder_layers=1,
        base_channels=32
    ):
        super().__init__()
        self.t1w_shape = t1w_shape
        self.n_rois = n_rois
        self.seq_len = seq_len
        self.fmri_dim = fmri_dim
        self.n_encoder_layers = n_encoder_layers
        self.gat_heads = 4
        self.num_nodes = n_rois

        self.fmri_encoder = TransformerFMRIEncoder(
            n_rois=n_rois,
            seq_len=seq_len,
            d_model=fmri_dim,
            n_heads=4,
            n_layers=2,
            n_decoder_scales=1,
            dropout=0.1
        )

        self.input_conv = nn.Conv3d(1, base_channels, 3, padding=1)

        self.smri_node_pool = nn.AdaptiveAvgPool3d((n_rois, 1, 1))

        self.encoder = GATEncoderBlock(
            in_channels=base_channels,
            out_channels=base_channels*2,
            num_nodes=n_rois,
            heads=self.gat_heads,
            dropout=0.1
        )

        self.decoder = GATDecoderBlock(
            in_channels=base_channels*2,
            out_channels=base_channels*2,
            fmri_dim=fmri_dim,
            num_nodes=n_rois,
            heads=self.gat_heads,
            use_cross_attention=True,
            dropout=0.1
        )

        self.smri_classifier_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels*2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fmri_classifier_branch = nn.Sequential(
            nn.Linear(fmri_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        self.classifier = self.smri_classifier_branch

        self.output_proj = nn.Linear(base_channels*2, 1)

        self.prompt_acceptance = PromptAcceptanceModule(
            feature_dim=base_channels*2,
            hidden_dim=32,
            smoothing_factor=0.1
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _build_full_conn_edge(self, N):
        row = torch.arange(N).repeat_interleave(N)
        col = torch.arange(N).repeat(N)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index

    def forward(self, t1w, bold_ts, bold_adj):
        if t1w.dim() == 3:
            t1w = t1w.unsqueeze(0)
        if bold_ts.dim() == 2:
            bold_ts = bold_ts.unsqueeze(0)
        if bold_adj.dim() == 2:
            bold_adj = bold_adj.unsqueeze(0)

        B = t1w.shape[0]

        fmri_outputs = self.fmri_encoder(bold_ts, bold_adj)
        multi_scale_kv = fmri_outputs['multi_scale_kv']
        global_fmri_features = fmri_outputs['global_features']

        if t1w.dim() == 4:
            t1w = t1w.unsqueeze(1)
        x = self.input_conv(t1w)

        x_nodes = self.smri_node_pool(x)
        x_nodes = x_nodes.squeeze(-1).squeeze(-1).transpose(1,2)

        edge_index = self._build_full_conn_edge(self.n_rois).to(x_nodes.device)

        x = self.encoder(x_nodes, edge_index)

        if len(multi_scale_kv) > 1:
            fused_keys = torch.stack([kv['keys'] for kv in multi_scale_kv], dim=0).mean(0)
            fused_values = torch.stack([kv['values'] for kv in multi_scale_kv], dim=0).mean(0)
            fmri_keys = fused_keys
            fmri_values = fused_values
        else:
            scale_kv = multi_scale_kv[0]
            fmri_keys = scale_kv['keys']
            fmri_values = scale_kv['values']

        x = self.decoder(x, edge_index, fmri_keys, fmri_values)

        reconstructed = self.output_proj(x)
        reconstructed = reconstructed.transpose(1,2)

        smri_features = self.smri_classifier_branch(x.transpose(1, 2))

        fmri_features = self.fmri_classifier_branch(global_fmri_features)

        fused_features = torch.cat([smri_features, fmri_features], dim=1)
        logits = self.fusion_classifier(fused_features).squeeze(-1)

        return {
            "logits": logits,
            "reconstructed": reconstructed,
            "fmri_features": global_fmri_features,
            "decoder_outputs": [x],
            "final_features": x,
            "multi_scale_kv": multi_scale_kv
        }

    def compute_loss(self, outputs, t1w_target, y):
        logits = outputs["logits"]

        if y.dim() == 0:
            y = y.unsqueeze(0)
        if logits.shape != y.shape:
            if logits.dim() == 1 and y.dim() == 1:
                pass
            else:
                y = y.view_as(logits)
        cls_loss = F.binary_cross_entropy_with_logits(logits, y.float())

        reconstructed = outputs["reconstructed"]
        if t1w_target.dim() == 3:
            t1w_target = t1w_target.unsqueeze(0).unsqueeze(0)
        elif t1w_target.dim() == 4:
            t1w_target = t1w_target.unsqueeze(1)
        smri_node_target = F.adaptive_avg_pool3d(t1w_target, (self.n_rois, 1, 1))
        smri_node_target = smri_node_target.squeeze(-1).squeeze(-1)

        if reconstructed.shape != smri_node_target.shape:
            smri_node_target = F.interpolate(smri_node_target, size=reconstructed.shape[-1], mode='linear', align_corners=False)

        recon_loss = F.mse_loss(reconstructed, smri_node_target)

        final_features = outputs["final_features"]
        prompt_acceptance_outputs = self.prompt_acceptance(
            features=final_features,
            recon_loss=recon_loss,
            training=self.training
        )

        log_ratio = prompt_acceptance_outputs['log_ratio']
        ratio_loss = -log_ratio.mean()

        cls_weight = 1.0
        recon_weight = 0.5
        ratio_weight = 0.3

        total_loss = (
            cls_weight * cls_loss
            + recon_weight * recon_loss
            + ratio_weight * ratio_loss
        )

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "recon_loss": recon_loss,
            "ratio_loss": ratio_loss,
            "prompt_acceptance": prompt_acceptance_outputs
        }

    def iterative_optimization(self, t1w, bold_ts, bold_adj, n_iterations=3):
        self.eval()

        with torch.no_grad():
            best_outputs = None
            best_score = float('-inf')

            for iter_i in range(n_iterations):
                if iter_i > 0:
                    noise = torch.randn_like(bold_ts) * 0.01
                    bold_ts_perturbed = bold_ts + noise
                else:
                    bold_ts_perturbed = bold_ts

                outputs = self.forward(t1w, bold_ts_perturbed, bold_adj)

                reconstructed = outputs["reconstructed"]
                if t1w.dim() == 3:
                    t1w_eval = t1w.unsqueeze(0).unsqueeze(0)
                elif t1w.dim() == 4:
                    t1w_eval = t1w.unsqueeze(1)
                else:
                    t1w_eval = t1w

                if t1w_eval.dim() == 5:
                    t1w_eval = F.adaptive_avg_pool3d(t1w_eval, (reconstructed.shape[-1], 1, 1))
                    t1w_eval = t1w_eval.squeeze(-1).squeeze(-1)

                if reconstructed.shape != t1w_eval.shape:
                    t1w_eval = F.interpolate(t1w_eval, size=reconstructed.shape[-1], mode='linear', align_corners=False)

                recon_loss = F.mse_loss(reconstructed, t1w_eval)

                final_features = outputs["final_features"]
                acceptance_outputs = self.prompt_acceptance(
                    features=final_features,
                    recon_loss=recon_loss,
                    training=False
                )

                ratio_score = acceptance_outputs['ratio'].mean()
                cls_confidence = torch.sigmoid(outputs["logits"]).mean()
                recon_quality = 1.0 / (1.0 + recon_loss)

                score = 0.5 * ratio_score + 0.3 * cls_confidence + 0.2 * recon_quality

                if score.item() > best_score:
                    best_score = score.item()
                    best_outputs = outputs
                    best_outputs['acceptance_outputs'] = acceptance_outputs
                    best_outputs['iteration_score'] = score

        return best_outputs
