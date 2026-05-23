# QAP-MFN Loss Function Formula Overview

## Total Loss Function

$$\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + 0.5\mathcal{L}_{\text{recon}} + 0.3\mathcal{L}_{\text{ratio}} + 0.01\mathcal{L}_{\text{reg}} + 0.1\mathcal{L}_{\text{cons}}}$$

---

## Loss Component Details

### 1. Classification Loss
$$\mathcal{L}_{\text{cls}} = -\frac{1}{N} \sum_{i=1}^{N} w_{y_i} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right]$$

**Class weights:** $w_c = \frac{N_{\text{total}}}{2 \times N_c}$

### 2. Reconstruction Loss
$$\mathcal{L}_{\text{recon}} = \frac{1}{N \times D \times H \times W} \sum_{i,d,h,w} \left( \hat{X}_{i,d,h,w} - X_{i,d,h,w} \right)^2$$

### 3. Policy Ratio Loss
$$\mathcal{L}_{\text{ratio}} = -\frac{1}{N} \sum_{i=1}^{N} \log(r_{\text{smooth}}^{(i)} + \epsilon)$$

**Where:**
- **New policy score:** $S_{\text{new}}^{(i)} = 0.7 \times Q_{\text{feat}}^{(i)} + 0.3 \times Q_{\text{recon}}^{(i)}$
- **Policy ratio:** $r^{(i)} = \frac{S_{\text{new}}^{(i)} + \epsilon}{\bar{S}_{\text{old}} + \epsilon}$
- **Smoothing:** $r_{\text{smooth}}^{(i)} = \tanh(r^{(i)} - 1) + 1$

### 4. Regularization Loss
$$\mathcal{L}_{\text{reg}} = \frac{1}{N} \sum_{i=1}^{N} \|F_{\text{fmri}}^{(i)}\|_2$$

### 5. Consistency Loss
$$\mathcal{L}_{\text{cons}} = 0 \quad \text{(simplified version)}$$

---

## Physical Meaning of Policy Ratio Loss

| Condition | Ratio Value | Loss Change | Model Behavior |
|-----------|-------------|-------------|----------------|
| $r > 1$ | New policy better | Loss decreases | Encourage current prompt |
| $r = 1$ | Policies equivalent | Loss unchanged | Keep status quo |
| $r < 1$ | New policy worse | Loss increases | Penalize current prompt |

**Core idea:** Adaptively optimize prompt generation strategy by comparing current fMRI prompt quality with historical average.

---

## Key Numerical Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| $\lambda_{\text{cls}}$ | 1.0 | Classification loss weight |
| $\lambda_{\text{recon}}$ | 0.5 | Reconstruction loss weight |
| $\lambda_{\text{ratio}}$ | 0.3 | Policy ratio loss weight |
| $\lambda_{\text{reg}}$ | 0.01 | Regularization weight |
| $\alpha$ | 0.1 | EMA smoothing factor |
| $\epsilon$ | $10^{-6}$ | Numerical stability |
| Gradient clipping | 1.0 | Prevent gradient explosion |

---

## Optimization Strategy

### Learning Rate Scheduling
$$\eta(t) = \begin{cases}
\eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & \text{warmup phase} \\
\eta_{\max} \cdot \frac{1}{2}\left(1 + \cos\left(\frac{\pi(t - T_{\text{warmup}})}{T_{\text{total}} - T_{\text{warmup}}}\right)\right) & \text{main training phase}
\end{cases}$$

### Iterative Self-Scoring (Inference Phase)
$$\text{Score} = 0.5 \times r_{\text{smooth}} + 0.3 \times \sigma(z) + 0.2 \times \frac{1}{1 + \mathcal{L}_{\text{recon}}}$$

---

## Design Highlights

1. **Multi-task balancing**: Classification + Reconstruction + Prompt optimization
2. **Adaptive mechanism**: Policy ratio automatically adjusts prompt quality
3. **Numerical stability**: Gradient clipping + epsilon protection + smoothing
4. **Class balancing**: Weighted loss handles data imbalance
5. **End-to-end optimization**: All components trained jointly

This loss function design enables QAP-MFN to:
- Accurately classify brain diseases
- Learn meaningful image representations
- Adaptively optimize multimodal fusion
- Balance multiple learning objectives
- Further optimize performance during inference
