# RMFN Loss Function Formula Overview

## Total Loss Function

The total objective follows the paper `L_total = L_cls + 0.5 * L_recon + 0.3 * L_ratio`:

$$\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + 0.5\,\mathcal{L}_{\text{recon}} + 0.3\,\mathcal{L}_{\text{ratio}}}$$

---

## Loss Component Details

### 1. Classification Loss
$$\mathcal{L}_{\text{cls}} = -\frac{1}{N} \sum_{i=1}^{N} w_{y_i} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right]$$

**Class weights:** $w_c = \frac{N_{\text{total}}}{2 \times N_c}$

### 2. Reconstruction Loss
$$\mathcal{L}_{\text{recon}} = \frac{1}{N \times R} \sum_{i,r} \left( \hat{X}_{i,r} - X_{i,r} \right)^2$$

### 3. Policy Ratio Loss
$$\mathcal{L}_{\text{ratio}} = -\frac{1}{N} \sum_{i=1}^{N} \log(r_{\text{smooth}}^{(i)} + \epsilon)$$

**Where:**
- **New policy score:** $q_t^{(i)} = \alpha \cdot Q_{\text{feat}}^{(i)} + (1-\alpha) \cdot Q_{\text{recon}}^{(i)}$
- **Policy ratio:** $r_t^{(i)} = \frac{q_t^{(i)} + \epsilon}{\bar{b}_{t-1} + \epsilon}$
- **Smoothing:** $r_{\text{smooth}}^{(i)} = \tanh(r_t^{(i)} - 1) + 1$
- **EMA baseline:** $\bar{b}_t = \gamma \cdot q_t + (1-\gamma) \cdot \bar{b}_{t-1}$

---

## Physical Meaning of Policy Ratio Loss

| Condition | Ratio Value | Loss Change | Model Behavior |
|-----------|-------------|-------------|----------------|
| $r > 1$ | New policy better | Loss decreases | Encourage current fusion |
| $r = 1$ | Policies equivalent | Loss unchanged | Keep status quo |
| $r < 1$ | New policy worse | Loss increases | Penalize current fusion |

**Core idea:** Adaptively modulate the cross-modal fusion by comparing current fusion quality (feature + reconstruction) with the EMA-tracked historical baseline.

---

## Key Numerical Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| $\lambda_{\text{cls}}$ | 1.0 | Classification loss weight |
| $\lambda_{\text{recon}}$ | 0.5 | Reconstruction loss weight |
| $\lambda_{\text{ratio}}$ | 0.3 | Policy ratio loss weight |
| $\alpha$ | 0.7 | Feature-vs-recon balance in quality score |
| $\gamma$ | 0.1 | EMA smoothing factor for the quality baseline |
| $\epsilon$ | $10^{-6}$ | Numerical stability |
| Gradient clipping | 1.0 | Prevent gradient explosion |

---

## Optimization Strategy

### Learning Rate Scheduling

We use AdamW with cosine-annealing warm restarts (`CosineAnnealingWarmRestarts` with `T_0=10`, `T_mult=2`) in `train/train_simplified.py`.

### Iterative Self-Scoring (Inference Phase)

$$\text{Score} = 0.5 \times r_{\text{smooth}} + 0.3 \times \sigma(z) + 0.2 \times \frac{1}{1 + \mathcal{L}_{\text{recon}}}$$

---

## Design Highlights

1. **Multi-task balancing**: Classification + Reconstruction + Adaptive ratio
2. **Adaptive mechanism**: Policy ratio adjusts fusion quality
3. **Numerical stability**: Gradient clipping + epsilon protection + smoothing
4. **Class balancing**: Weighted loss handles data imbalance
5. **End-to-end optimization**: All components trained jointly

This loss function design enables RMFN to:
- Accurately classify brain diseases
- Learn meaningful anatomical representations
- Adaptively modulate cross-modal fusion
- Balance classification, reconstruction, and quality objectives
