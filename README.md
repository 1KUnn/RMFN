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

## Data

| Dataset | Description | Link |
|---|---|---|
| ABIDE I (Preprocessed) | Autism spectrum disorder, 1,025 subjects | [Link](https://ida.loni.usc.edu/login.jsp) |
| Taowu | Parkinson's disease, 40 subjects (20 PD + 20 controls) | [Link](https://fcp-indi.s3.amazonaws.com/data/Projects/INDI/umf_pd/taowu.tar.gz) |

## Citation

```bibtex
@article{yang2025rmfn,
  title={RMFN: A Reliable Multimodal Fusion Network for Brain Disease Diagnosis},
  author={Yang, Hong and Huang, Ruiwen and Zhang, Peng and Zhang, Yao and Zhang, Yanchun},
  journal={Nuclear Physics B},
  year={2025}
}
```

## License

Apache License 2.0
