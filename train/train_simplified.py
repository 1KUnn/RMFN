import os
import sys
import time
import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.multimodal_dataset import MultimodalTaowuDataset, custom_collate
from model.simplified_hp_bfn import RMFN


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device, class_weight=None):
    model.train()

    total_loss = 0
    cls_loss_sum = 0
    recon_loss_sum = 0
    ratio_loss_sum = 0
    n_batches = 0

    for batch in loader:
        t1w = batch["t1w"].to(device)
        bold_ts = batch["bold_ts"].to(device)
        bold_adj = batch["bold_adj"].to(device)
        y = batch["y"].to(device).float()

        optimizer.zero_grad()

        outputs = model(t1w, bold_ts, bold_adj)

        loss_dict = model.compute_loss(outputs, t1w, y)

        if class_weight is not None and y.numel() > 0:
            logits = outputs["logits"]

            y_for_loss = y
            if y.dim() == 0:
                y_for_loss = y.unsqueeze(0)

            weighted_cls_loss = F.binary_cross_entropy_with_logits(
                logits, y_for_loss.float(), weight=class_weight[y.long()]
            )
            loss = (weighted_cls_loss
                   + 0.5 * loss_dict["recon_loss"]
                   + 0.3 * loss_dict["ratio_loss"])
            cls_loss_stat = weighted_cls_loss
        else:
            loss = loss_dict["total_loss"]
            cls_loss_stat = loss_dict["cls_loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        cls_loss_sum += cls_loss_stat.item()
        recon_loss_sum += loss_dict["recon_loss"].item()
        ratio_loss_sum += loss_dict["ratio_loss"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "cls_loss": cls_loss_sum / max(n_batches, 1),
        "recon_loss": recon_loss_sum / max(n_batches, 1),
        "ratio_loss": ratio_loss_sum / max(n_batches, 1)
    }


def evaluate(model, loader, device, use_iterative=False):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            t1w = batch["t1w"].to(device)
            bold_ts = batch["bold_ts"].to(device)
            bold_adj = batch["bold_adj"].to(device)
            y = batch["y"].to(device).float()

            if use_iterative:
                outputs = model.iterative_optimization(t1w, bold_ts, bold_adj, n_iterations=3)
            else:
                outputs = model(t1w, bold_ts, bold_adj)

            logits = outputs["logits"]
            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            labels_np = y.cpu().numpy()

            if probs_np.ndim == 0:
                probs_np = np.array([probs_np])
            if labels_np.ndim == 0:
                labels_np = np.array([labels_np])

            all_probs.extend(probs_np)
            all_labels.extend(labels_np)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    return {
        "acc": acc,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def main():
    parser = argparse.ArgumentParser(description="RMFN Training")

    parser.add_argument("--data_root", type=str, default="./data",
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Path to save results")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory for cached preprocessed data")

    parser.add_argument("--n_rois", type=int, default=100,
                        help="Number of brain ROIs")
    parser.add_argument("--seq_len", type=int, default=200,
                        help="Length of fMRI time series")
    parser.add_argument("--fmri_dim", type=int, default=128,
                        help="Hidden dimension for fMRI features")
    parser.add_argument("--n_encoder_layers", type=int, default=3,
                        help="Number of encoder layers")
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Base number of channels")
    parser.add_argument("--t1w_shape", type=int, nargs=3, default=[176, 176, 176],
                        help="Target shape for T1w volumes")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--val_ratio", type=float, default=0.3,
                        help="Ratio of data for validation")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    print("=" * 60)
    print("RMFN Training")
    print("=" * 60)
    print(f"Device: {args.device}")

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading multimodal dataset...")
    dataset = MultimodalTaowuDataset(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        n_rois=args.n_rois,
        t1w_shape=tuple(args.t1w_shape),
        minimize_memory=True
    )

    n_val = int(len(dataset) * args.val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)

    all_labels = [dataset[i]["y"].item() for i in range(len(dataset))]
    class_counts = np.bincount(all_labels)
    class_weight = torch.tensor(len(all_labels) / (2 * class_counts), dtype=torch.float32).to(device)

    print(f"\nDataset split:")
    print(f"  Train: {len(train_ds)} subjects")
    print(f"  Val: {len(val_ds)} subjects")
    print(f"  Class distribution: {class_counts}")
    print(f"  Class weights: {class_weight.cpu().numpy()}")

    print("\nBuilding RMFN model...")
    model = RMFN(
        t1w_shape=tuple(args.t1w_shape),
        n_rois=args.n_rois,
        seq_len=args.seq_len,
        fmri_dim=args.fmri_dim,
        n_encoder_layers=args.n_encoder_layers,
        base_channels=args.base_channels
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=args.lr * 0.01
    )

    print("\nStarting training...")
    print("-" * 60)

    best_score = -1
    best_metrics = {}
    early_stop_count = 0
    patience = 20
    best_auc = -1
    auc_patience = 0
    auc_patience_limit = 10

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_logs = train_one_epoch(
            model, train_loader, optimizer, device, class_weight
        )

        val_logs = evaluate(model, val_loader, device, use_iterative=False)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        dt = time.time() - t0

        print(f"Epoch {epoch:03d} | lr={current_lr:.2e} | time={dt:.1f}s")
        print(f"  Train | loss={train_logs['loss']:.4f} | cls={train_logs['cls_loss']:.4f}")
        print(f"  Val   | ACC={val_logs['acc']:.4f} | AUC={val_logs['auc']:.4f} | F1={val_logs['f1']:.4f}")

        combined_score = (val_logs['acc'] * 2 + val_logs['auc'] * 2 + val_logs['f1'] + val_logs['precision'] + val_logs['recall']) / 7
        if combined_score > best_score:
            best_score = combined_score
            best_metrics = val_logs.copy()
            best_auc = val_logs['auc']

            best_path = os.path.join(args.output_dir, "rmfn_best.pt")
            torch.save({
                'model': model.state_dict(),
                'args': vars(args),
                'val_metrics': val_logs,
                'train_logs': train_logs,
                'epoch': epoch
            }, best_path)

            early_stop_count = 0
            auc_patience = 0
            print(f"        New best: {combined_score:.4f}")
        else:
            early_stop_count += 1
            if val_logs['auc'] < best_auc:
                auc_patience += 1
            else:
                auc_patience = 0

        if early_stop_count >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

        if auc_patience >= auc_patience_limit:
            print(f"\nEarly stopping at epoch {epoch} (AUC no improvement)")
            break

    print("\n" + "=" * 60)
    print("Final evaluation with iterative optimization...")
    print("=" * 60)

    model.load_state_dict(torch.load(best_path, weights_only=False)['model'])
    final_metrics = evaluate(model, val_loader, device, use_iterative=True)

    print("\nFinal results:")
    for metric, value in final_metrics.items():
        print(f"   {metric.upper()}: {value:.4f}")

    combined_final = (final_metrics['acc'] + final_metrics['auc'] + final_metrics['f1'] +
                     final_metrics['precision'] + final_metrics['recall']) / 5
    print(f"   Combined: {combined_final:.4f}")

    final_path = os.path.join(args.output_dir, "rmfn_final.pt")
    torch.save({
        'model': model.state_dict(),
        'args': vars(args),
        'best_metrics': best_metrics,
        'final_metrics': final_metrics,
        'final_score': combined_final
    }, final_path)

    print(f"\nDone! Results saved to: {args.output_dir}")
    print(f"  Best model: {best_path}")
    print(f"  Final model: {final_path}")


if __name__ == "__main__":
    main()
