# RMFN Loss Function Mathematical Formulation

## Total Loss Function

The total objective follows the paper `L_total = L_cls + lambda_recon * L_recon + lambda_ratio * L_ratio`:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{recon}} \mathcal{L}_{\text{recon}} + \lambda_{\text{ratio}} \mathcal{L}_{\text{ratio}}$$

Where the weight coefficients are:
- $\lambda_{\text{cls}} = 1.0$ (classification loss weight)
- $\lambda_{\text{recon}} = 0.5$ (reconstruction loss weight)
- $\lambda_{\text{ratio}} = 0.3$ (policy ratio loss weight)

---

## 1. Classification Loss $\mathcal{L}_{\text{cls}}$

### 1.1 Binary Cross-Entropy Loss

For binary classification (patient vs control), we use binary cross-entropy with logits:

$$\mathcal{L}_{\text{cls}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right]$$

Where:
- $N$ is the batch size
- $y_i \in \{0, 1\}$ is the true label (0=control, 1=patient)
- $z_i$ is the model output logits
- $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function

### 1.2 Class Weight Adjustment

To handle data imbalance, class weights are introduced:

$$w_c = \frac{N_{\text{total}}}{2 \times N_c}$$

Where $N_c$ is the number of samples in class $c$.

The weighted classification loss:

$$\mathcal{L}_{\text{cls}}^{\text{weighted}} = -\frac{1}{N} \sum_{i=1}^{N} w_{y_i} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right]$$

---

## 2. Reconstruction Loss $\mathcal{L}_{\text{recon}}$

### 2.1 Mean Squared Error Loss

The reconstruction loss is computed on ROI-level sMRI features pooled from the decoder output:

$$\mathcal{L}_{\text{recon}} = \frac{1}{N \times R} \sum_{i=1}^{N} \sum_{r=1}^{R} \left( \hat{X}_{i,r} - X_{i,r} \right)^2$$

Where:
- $X_{i,r}$ is the target ROI feature for subject $i$, region $r$
- $\hat{X}_{i,r}$ is the predicted ROI feature for subject $i$, region $r$
- $R$ is the number of ROIs

### 2.2 Size Adaptation

When the predicted feature length does not match the target, linear interpolation is used to align dimensions.

---

## 3. Policy Ratio Loss $\mathcal{L}_{\text{ratio}}$ (Core Innovation)

### 3.1 Quality Score Computation

The combined quality score consists of feature quality and reconstruction quality:

$$q_t^{(i)} = \alpha \cdot Q_{\text{feat}}^{(i)} + (1-\alpha) \cdot Q_{\text{recon}}^{(i)}$$

with $\alpha = 0.7$ by default.

#### Feature Quality Assessment:
$$Q_{\text{feat}}^{(i)} = \sigma\left( \text{MLP}_{\text{feat}}\left( \text{Pool}(F_{\text{fused}}^{(i)}) \right) \right)$$

Where:
- $F_{\text{fused}}^{(i)} \in \mathbb{R}^{R \times d}$ is the fused multimodal representation
- $\text{Pool}(\cdot)$ is adaptive average pooling over ROI nodes
- $\text{MLP}_{\text{feat}}$ is the feature quality assessment network

#### Reconstruction Quality Assessment:
$$Q_{\text{recon}}^{(i)} = \sigma\left( \mathbf{w}_r \cdot \mathrm{ReLU}\left(-\log(\mathcal{L}_{\text{recon}}^{t-1} + \epsilon)\right) + b_r \right)$$

Where $\mathcal{L}_{\text{recon}}^{t-1}$ is the cached reconstruction loss from the previous training step.

### 3.2 Policy Ratio Computation

The historical baseline uses exponential moving average (EMA):

$$\bar{b}_{t} = \gamma \cdot q_t + (1-\gamma) \cdot \bar{b}_{t-1}$$

Where $\gamma = 0.1$ is the smoothing factor.

Policy ratio:
$$r_t^{(i)} = \frac{q_t^{(i)} + \epsilon}{\bar{b}_{t-1} + \epsilon}$$

### 3.3 Ratio Smoothing

To prevent the ratio from being too large, tanh smoothing is applied:

$$r_{\text{smooth}}^{(i)} = \tanh(r_t^{(i)} - 1) + 1$$

### 3.4 Policy Ratio Loss

$$\mathcal{L}_{\text{ratio}} = -\frac{1}{N} \sum_{i=1}^{N} \log(r_{\text{smooth}}^{(i)} + \epsilon)$$

**Physical meaning:**
- When $r_{\text{smooth}}^{(i)} > 1$, the new fusion is better than the historical baseline and the loss decreases
- When $r_{\text{smooth}}^{(i)} < 1$, the new fusion is worse than the historical baseline and the loss increases
- Encourages the model to keep reliable cross-modal information flow while suppressing unreliable signals

---

## 4. Gradient and Optimization

### 4.1 Gradient Clipping

To prevent gradient explosion, gradient norm clipping is applied:

$$\nabla \theta \leftarrow \begin{cases}
\nabla \theta & \text{if } \|\nabla \theta\|_2 \leq 1.0 \\
\frac{\nabla \theta}{\|\nabla \theta\|_2} & \text{if } \|\nabla \theta\|_2 > 1.0
\end{cases}$$

### 4.2 Learning Rate Scheduling

Cosine annealing with warm restarts is used in `train/train_simplified.py`:

```
optimizer = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.01
)
```

Where:
- $\eta_{\max} = 10^{-4}$ is the maximum learning rate
- $T_0 = 10$ is the initial restart period

---

## 5. Iterative Self-Optimization Score Function

During inference, a composite score is used to select the best fusion result:

$$\text{Score}^{(i)} = 0.5 \times r_{\text{smooth}}^{(i)} + 0.3 \times \sigma(z^{(i)}) + 0.2 \times \frac{1}{1 + \mathcal{L}_{\text{recon}}^{(i)}}$$

Where:
- $r_{\text{smooth}}^{(i)}$ is the policy ratio score
- $\sigma(z^{(i)})$ is the classification confidence
- $\frac{1}{1 + \mathcal{L}_{\text{recon}}^{(i)}}$ is the reconstruction quality score

---

## 6. Numerical Stability Considerations

### 6.1 Epsilon Protection
All division and log operations add a small value $\epsilon = 10^{-6}$ to prevent division by zero and log-of-zero.

### 6.2 Gradient Clipping
Gradient norms are limited to a reasonable range (`max_norm=1.0`).

### 6.3 Loss Weight Balancing
The weights of each loss term are tuned through experiments to ensure training stability.

---

## Summary

The loss function design of RMFN combines:

1. **Supervised learning**: Classification loss provides the main supervision signal
2. **Self-supervised learning**: Reconstruction loss learns meaningful representations
3. **Adaptive modulation**: Policy ratio loss adaptively modulates cross-modal fusion
4. **Multi-task learning**: Balances multiple objectives to improve overall performance

This multi-level loss design enables the model to effectively fuse multimodal information and adaptively modulate the quality of cross-modal features, achieving strong performance on brain disease classification tasks.
