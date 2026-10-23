# Datasets

The paper uses two binary classification datasets from the
[LIBSVM data repository](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html):

| Dataset | Raw files | Resulting matrix | Used in |
|---|---|---|---|
| a9a | `a9a` | `A_a9a.npy` (32,561 × 123) | all figures |
| ijcnn1 | `ijcnn1`, `ijcnn1.t` | `A_ijcnn1.npy` (141,691 × 22) | all figures |

The experiments use a random 10 % split (at most 10,000 samples) of each dataset as the test set on which the
average loss is evaluated; the split is drawn with `np.random.seed(42)` inside the experiment scripts, so it is
reproducible without shipping the split.

## Download

```bash
mkdir -p data/data_a9a data/data_ijcnn1

# a9a  (~2.5 MB)
curl -o data/data_a9a/a9a      https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a

# ijcnn1 (training + test, ~20 MB)
curl -o data/data_ijcnn1/ijcnn1   https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1
curl -o data/data_ijcnn1/ijcnn1.t https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t
```

## Convert to `.npy`

```bash
pip install scikit-learn
python data/prepare_data.py
```

This writes `data/data_a9a/A_a9a.npy`, `data/data_a9a/y_a9a.npy`, `data/data_ijcnn1/A_ijcnn1.npy`,
`data/data_ijcnn1/y_ijcnn1.npy`.

**Note.** You only need the datasets to re-run the experiments from scratch. The per-run results needed to
regenerate the paper's figures are already included under `experiment/result/`.
