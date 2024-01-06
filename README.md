# AAI Fall 2023 Project: Train a Stable MNIST Classifier
Train a classifier for modified MNIST with unstable (spurious) features.

## How to use
1. Install the requirements with `pip install -r requirements.txt`.
2. Extract the data to the 'processed_data' folder.
3. Run the test script with `python test.py`.

## Optional
- Inspect data with `data_inspection.ipynb`.
- Run the training script with `python main.py`.

## References
Codes for these papers are used in this project:
```
@article{InvariantRiskMinimization,
    title={Invariant Risk Minimization},
    author={Arjovsky, Martin and Bottou, L{\'e}on and Gulrajani, Ishaan and Lopez-Paz, David},
    journal={arXiv},
    year={2019}
}
```
```
@inproceedings{bao2022learning,
  title={Learning Stable Classifiers by Transferring Unstable Features},
  author={Bao, Yujia and Chang, Shiyu and Barzilay, Regina},
  booktitle={International Conference on Machine Learning},
  pages={1483--1507},
  year={2022},
  organization={PMLR}
}
```