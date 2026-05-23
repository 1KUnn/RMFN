# QAP-MFN Loss Function Mathematical Formulation

## Total Loss Function

$$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{recon}} \mathcal{L}_{\text{recon}} + \lambda_{\text{ratio}} \mathcal{L}_{\text{ratio}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}}$$

Where the weight coefficients are:
- $\lambda_{\text{cls}} = 1.0$ (classification loss weight)
- $\lambda_{\text{recon}} = 0.5$ (reconstruction loss weight)
- $\lambda_{\text{ratio}} = 0.3$ (policy ratio loss weight)
- $\lambda_{\text{reg}} = 0.01$ (regularization loss weight)
- $\lambda_{\text{cons}} = 0.1$ (consistency loss weight)

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

The reconstruction loss measures the difference between reconstructed and original T1w images:

$$\mathcal{L}_{\text{recon}} = \frac{1}{N \times D \times H \times W} \sum_{i=1}^{N} \sum_{d=1}^{D} \sum_{h=1}^{H} \sum_{w=1}^{W} \left( \hat{X}_{i,d,h,w} - X_{i,d,h,w} \right)^2$$

Where:
- $X_{i,d,h,w}$ is the voxel value of the original T1w image
- $\hat{X}_{i,d,h,w}$ is the voxel value of the reconstructed T1w image
- $(D, H, W)$ is the 3D image dimensions

### 2.2 Size Adaptation

When the reconstructed image size does not match the target, trilinear interpolation is used:

$$X_{\text{target}}^{\text{resized}} = \text{Interpolate}(X_{\text{target}}, \text{size}=\hat{X}.\text{shape}[2:], \text{mode}=\text{'trilinear'})$$

---

## 3. Policy Ratio Loss $\mathcal{L}_{\text{ratio}}$ (Core Innovation)

### 3.1 Policy Score Computation

The new policy score consists of feature quality and reconstruction quality:

$$S_{\text{new}}^{(i)} = 0.7 \times Q_{\text{feat}}^{(i)} + 0.3 \times Q_{\text{recon}}^{(i)}$$

#### Feature Quality Assessment:
$$Q_{\text{feat}}^{(i)} = \sigma\left( \text{MLP}_{\text{feat}}\left( \text{GAP}(F_{\text{final}}^{(i)}) \right) \right)$$

Where:
- $F_{\text{final}}^{(i)} \in \mathbb{R}^{C \times D \times H \times W}$ is the final decoder features
- $\text{GAP}(\cdot)$ is global average pooling: $\text{GAP}(F) = \frac{1}{DHW} \sum_{d,h,w} F_{:,d,h,w}$
- $\text{MLP}_{\text{feat}}$ is the feature quality assessment network

#### Reconstruction Quality Assessment:
$$Q_{\text{recon}}^{(i)} = \sigma\left( \text{MLP}_{\text{recon}}\left( \frac{-\log(\mathcal{L}_{\text{recon}}^{(i)} + \epsilon)}{10} \right) \right)$$

### 3.2 Policy Ratio Computation

The old policy score uses exponential moving average (EMA):

$$\bar{S}_{\text{old}}^{(t)} = \alpha \cdot \frac{1}{N} \sum_{i=1}^{N} S_{\text{new}}^{(i)} + (1-\alpha) \cdot \bar{S}_{\text{old}}^{(t-1)}$$

Where $\alpha = 0.1$ is the smoothing factor.

Policy ratio:
$$r^{(i)} = \frac{S_{\text{new}}^{(i)} + \epsilon}{\bar{S}_{\text{old}} + \epsilon}$$

### 3.3 Ratio Smoothing

To prevent the ratio from being too large, tanh smoothing is applied:

$$r_{\text{smooth}}^{(i)} = \tanh(r^{(i)} - 1) + 1$$

### 3.4 Policy Ratio Loss

$$\mathcal{L}_{\text{ratio}} = -\frac{1}{N} \sum_{i=1}^{N} \log(r_{\text{smooth}}^{(i)} + \epsilon)$$

**Physical meaning:**
- When $r_{\text{smooth}}^{(i)} > 1$, new policy is better than historical average, loss decreases
- When $r_{\text{smooth}}^{(i)} < 1$, new policy is worse than historical average, loss increases
- Encourages the model to generate more effective fMRI prompt representations

---

## 4. fMRI Feature Regularization Loss $\mathcal{L}_{\text{reg}}$

Prevents fMRI global features from overfitting:

$$\mathcal{L}_{\text{reg}} = \frac{1}{N} \sum_{i=1}^{N} \|F_{\text{fmri}}^{(i)}\|_2$$

Where $F_{\text{fmri}}^{(i)} \in \mathbb{R}^{d_{\text{fmri}}}$ is the fMRI global feature vector for the $i$-th sample.

---

## 5. Multi-scale Consistency Loss $\mathcal{L}_{\text{cons}}$

In the simplified version, since only a single scale is used, this term is zero:

$$\mathcal{L}_{\text{cons}} = 0$$

In the full multi-scale version, this loss ensures consistency across different decoder scales:

$$\mathcal{L}_{\text{cons}} = \frac{1}{K-1} \sum_{k=1}^{K-1} \text{MSE}(\text{Resize}(F_k), F_{k+1})$$

Where $K$ is the number of decoder scales and $F_k$ is the feature map at scale $k$.

---

## 6. Gradient and Optimization

### 6.1 Gradient Clipping

To prevent gradient explosion, gradient norm clipping is applied:

$$\nabla \theta \leftarrow \begin{cases}
\nabla \theta & \text{if } \|\nabla \theta\|_2 \leq 1.0 \\
\frac{\nabla \theta}{\|\nabla \theta\|_2} & \text{if } \|\nabla \theta\|_2 > 1.0
\end{cases}$$

### 6.2 Learning Rate Scheduling

Warmup + cosine annealing scheduling:

$$\eta(t) = \begin{cases}
\eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t \leq T_{\text{warmup}} \\
\eta_{\max} \cdot \frac{1}{2}\left(1 + \cos\left(\frac{\pi(t - T_{\text{warmup}})}{T_{\text{total}} - T_{\text{warmup}}}\right)\right) & \text{if } t > T_{\text{warmup}}
\end{cases}$$

Where:
- $\eta_{\max} = 10^{-4}$ is the maximum learning rate
- $T_{\text{warmup}} = \min(5, T_{\text{total}}/2)$ is the number of warmup epochs
- $T_{\text{total}}$ is the total number of training epochs

---

## 7. Iterative Self-Optimization Score Function

During inference, a composite score is used to select the best prompt:

$$\text{Score}^{(i)} = 0.5 \times r_{\text{smooth}}^{(i)} + 0.3 \times \sigma(z^{(i)}) + 0.2 \times \frac{1}{1 + \mathcal{L}_{\text{recon}}^{(i)}}$$

Where:
- $r_{\text{smooth}}^{(i)}$ is the policy ratio score
- $\sigma(z^{(i)})$ is the classification confidence
- $\frac{1}{1 + \mathcal{L}_{\text{recon}}^{(i)}}$ is the reconstruction quality score

---

## 8. Numerical Stability Considerations

### 8.1 Epsilon Protection
All division operations add a small value $\epsilon = 10^{-6}$ to prevent division by zero.

### 8.2 Log Protection
Log operations use the form $\log(x + \epsilon)$.

### 8.3 Gradient Clipping
Gradient norms are limited to a reasonable range.

### 8.4 Loss Weight Balancing
The weights of each loss term are tuned through experiments to ensure training stability.

---

## Summary

The loss function design of QAP-MFN cleverly combines:

1. **Supervised learning**: Classification loss provides the main supervision signal
2. **Self-supervised learning**: Reconstruction loss learns meaningful representations
3. **Reinforcement learning concept**: Policy ratio loss optimizes prompt quality
4. **Regularization**: Prevents overfitting and improves generalization
5. **Multi-task learning**: Balances multiple objectives to improve overall performance

This multi-level loss design enables the model to effectively fuse multimodal information and adaptively optimize the quality of fMRI prompts, achieving excellent performance on brain disease classification tasks.
