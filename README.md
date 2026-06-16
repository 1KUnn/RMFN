# RMFN

A Reliable Multimodal Fusion Network for brain disease diagnosis using structural and functional MRI.

## Installation

```bash
pip install torch torchvision torchaudio
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install nibabel nilearn scikit-learn pandas numpy
pip install -r requirements.txt
```

## Project Structure

```
RMFN/
├── model/                   # Model architectures
│   └── simplified_hp_bfn.py
├── data/                    # Data loading
│   └── multimodal_dataset.py
├── train/                   # Training scripts
│   └── train_simplified.py
├── docs/                    # Documentation
└── requirements.txt
```


### Training

```bash
python train/train_simplified.py \
    --data_root ./data \
    --output_dir ./results \
    --device cuda
```

## Citation

```bibtex
@article{yang2025rmfn,
  title={RMFN: A Reliable Multimodal Fusion Network for Brain Disease Diagnosis},
  author={Yang, Hong and Huang, Ruiwen and Tan, Wenfeng and Zhang, Peng and Zhang, Yao and Zhang, Yanchun},
  journal={Nuclear Physics B},
  year={2025}
}
```

## License

Apache License 2.0
