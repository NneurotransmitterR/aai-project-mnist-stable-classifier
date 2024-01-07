# AAI Fall 2023 Project: Train a Stable MNIST Classifier
Train a classifier for modified MNIST with unstable (spurious) features.


## How to use
1. Install the requirements in 'requirements.txt'.
2. Extract the data to the 'processed_data' folder.
3. Run the test script with `python test.py`.

## Optional
- Inspect data with `data_inspection.ipynb`.
- Run the training script with `python main.py`.

## Directory Structure
```
aai-project-mnist-stable-classifier/
├── processed_data/
│   ├── train/
│   ├── test/
│   └── val/
├── models/
│   ├── CNN.py
│   ├── MLP.py
│   └── OCNN.py
├── saved_models/
│   ├── mlp_10000_20240106.pth
│   └── OriginalMNISTClassifier.pth
├── data_inspection.ipynb
├── OddMNIST.py
├── main.py
├── test.py
├── results.txt
├── requirements.txt
├── LICENSE
└── README.md
```

## References
Codes for these papers are used in this project:
1. [InvariantRiskMinimization](https://github.com/facebookresearch/InvariantRiskMinimization)
```
@article{InvariantRiskMinimization,
    title={Invariant Risk Minimization},
    author={Arjovsky, Martin and Bottou, L{\'e}on and Gulrajani, Ishaan and Lopez-Paz, David},
    journal={arXiv},
    year={2019}
}
```
