# QAP-MFN GitHub Upload Guide

This guide helps you upload the QAP-MFN project to GitHub while ensuring medical data privacy compliance.

## Repository Structure

```
QAP-MFN/
├── README.md                       # Project documentation
├── LICENSE                         # Apache 2.0 License
├── requirements.txt               # Python dependencies
├── .gitignore                      # Git ignore rules
├── model/
│   └── simplified_hp_bfn.py       # Core model architecture
├── data/
│   └── multimodal_dataset.py       # Data loading and preprocessing
├── train/
│   └── train_simplified.py         # Training script
└── docs/
    ├── loss_function_formulation.md # Mathematical formulation
    └── loss_summary.md             # Loss function overview
```

## Privacy Compliance Checklist

### Files EXCLUDED from Upload (via .gitignore)

| Type | Pattern | Reason |
|------|---------|--------|
| Medical images | `*.nii.gz`, `*.nii`, `*.npy`, `*.npz` | Patient data |
| Data folders | `data/`, `datasets/`, `raw_data/` | Contains patient data |
| Cache folders | `cache/`, `nilearn_cache/` | Contains preprocessed data |
| Model weights | `*.pth`, `*.pt`, `checkpoints/` | Training outputs |
| Results | `results/`, `logs/`, `outputs/` | Contains metrics |
| Shell scripts | `*.sh` | May contain data paths |
| Test scripts | `*quick_test*.py` | Private scripts |
| Personal notes | `*我的*.md`, `*笔记*.md` | Private content |

### Files Safe to Upload

- Core model code (`model/*.py`)
- Data loading code (`data/*.py`)
- Training scripts (`train/*.py`)
- Documentation (`docs/*.md`, `README.md`)
- Configuration files (`requirements.txt`, `LICENSE`)
- `.gitignore`

## Upload Steps

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `QAP-MFN`
3. Description: `Quality-Aware Prompted Multimodal Fusion Network for Brain Disease Diagnosis`
4. Select **Public** or **Private**
5. **DO NOT** initialize with README
6. Click **Create repository**

### Step 2: Initialize Local Repository

Navigate to the QAP-MFN folder:

```bash
cd e:/work/cursor/QAP-MFN
git init
```

### Step 3: Configure Git

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### Step 4: Add Files and Commit

```bash
git add .
git commit -m "Initial commit: QAP-MFN model code and documentation"
```

### Step 5: Set Remote and Push

```bash
git remote add origin https://github.com/1KUnn/QAP-MFN.git
git branch -M main
git push -u origin main
```

### Step 6: Verify Upload

1. Go to https://github.com/1KUnn/QAP-MFN
2. Check that:
   - All core code files are visible
   - Medical data files are NOT present
   - `.gitignore` is working correctly

## Data Preparation (For Users)

After downloading the code, users should prepare their data in this format:

### Simple Format (Recommended)

```
data/
├── patient_001/
│   ├── T1w.nii.gz          # T1-weighted MRI
│   ├── bold_timeseries.npy  # fMRI time series (ROIs x Time)
│   ├── bold_adjacency.npy   # Functional connectivity matrix
│   └── label.txt            # Label: 0 (control) or 1 (patient)
├── patient_002/
│   └── ...
└── control_001/
    └── ...
```

### BIDS Format

```
data/
├── sub-001/
│   ├── anat/
│   │   └── sub-001_T1w.nii.gz
│   └── func/
│       └── sub-001_task-rest_bold.nii.gz
└── ...
```

Note: Labels should be inferred from folder names containing "patient" or "control".

## Troubleshooting

### Issue: Large files uploaded

Check if `.gitignore` is working:
```bash
git status
```

If large files appear, check `.gitignore` patterns.

### Issue: Medical data accidentally uploaded

1. Remove from git tracking:
   ```bash
   git rm --cached -r data/
   ```

2. Add to `.gitignore`:
   ```
   data/
   ```

3. Commit and push:
   ```bash
   git commit -m "Remove data from tracking"
   git push
   ```

4. **WARNING**: Data is still in git history. For sensitive data, recreate the repository.

### Issue: GitHub rejects large files

GitHub has a 100MB file limit. Use Git LFS for large files:
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

## Security Best Practices

1. **Never upload raw medical data**
2. **Review all files before commit**: `git status`
3. **Test .gitignore**: Verify excluded files are not tracked
4. **Use .gitignore patterns correctly**:
   - `data/` excludes the folder
   - `data` matches files named "data"
5. **Regular audits**: Periodically check repository for leaked data

## Contact

For questions about data privacy or the code, please contact:
- **Yao Zhang** (Corresponding Author) - zhangyao@pku.edu.cn
