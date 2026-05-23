# QAP-MFN

A multimodal deep learning framework for brain disease classification using structural and functional MRI.

## Installation

```bash
pip install torch torchvision torchaudio
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install nibabel nilearn scikit-learn pandas numpy
pip install -r requirements.txt
```

## Project Structure

```
QAP-MFN/
├── model/                   # Model architectures
│   └── simplified_hp_bfn.py
├── data/                    # Data loading
│   └── multimodal_dataset.py
├── train/                   # Training scripts
│   └── train_simplified.py
├── docs/                    # Documentation
└── requirements.txt
```

## Quick Start

### Data Format

```
data/
├── subject_001/
│   ├── T1w.nii.gz
│   ├── bold_timeseries.npy
│   ├── bold_adjacency.npy
│   └── label.txt
└── ...
```

### Training

```bash
python train/train_simplified.py \
    --data_root ./data \
    --output_dir ./results \
    --n_rois 100 \
    --seq_len 200 \
    --epochs 100 \
    --device cuda
```

## Citation

```bibtex
@article{yang2025qapmfn,
  title={A Quality-Aware Prompted Multimodal Fusion Network for Brain Disease Diagnosis},
  author={Yang, Hong and Huang, Ruiwen and Tan, Wenfeng and Zhang, Peng and Zhang, Yao and Zhang, Yanchun},
  journal={Nuclear Physics B},
  year={2025}
}
```

## License

Apache License 2.0
