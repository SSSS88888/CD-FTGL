"""Build the ``.npy`` feature/label matrices used by the experiments from the raw LIBSVM files.

The paper uses two binary classification datasets:

* ``a9a``    -- 32,561 x 123
* ``ijcnn1`` -- 141,691 x 22  (49,990 training + 91,701 test, concatenated exactly as in the
  original experiments; the paper's "49990 examples" refers to the training portion)

Download the raw files first (see ``data/README.md``), then run::

    python data/prepare_data.py            # both datasets
    python data/prepare_data.py --name a9a

Each dataset directory ends up with ``A_<name>.npy`` (dense float64 features) and
``y_<name>.npy`` (labels in {-1, +1}).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.datasets import load_svmlight_file

HERE = os.path.dirname(os.path.abspath(__file__))

# dataset name -> list of raw LIBSVM files (concatenated in this order)
DATASETS = {
    "a9a": ["a9a"],
    "ijcnn1": ["ijcnn1", "ijcnn1.t"],
}


def build(name: str, parts: list[str]) -> None:
    folder = os.path.join(HERE, "data_" + name)
    xs, ys = [], []
    for part in parts:
        path = os.path.join(folder, part)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "%s not found.\nDownload the raw LIBSVM files into %s first (see data/README.md)."
                % (path, folder))
        X, y = load_svmlight_file(path)
        X = X.toarray()
        print("  %-12s %7d samples x %3d features" % (part, X.shape[0], X.shape[1]))
        xs.append(X)
        ys.append(y)

    A = np.vstack(xs)
    b = np.hstack(ys)

    np.save(os.path.join(folder, "A_%s.npy" % name), A)
    np.save(os.path.join(folder, "y_%s.npy" % name), b)
    print("  -> A_%s.npy %s, y_%s.npy %s (labels: %s)\n"
          % (name, A.shape, name, b.shape, np.unique(b)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", choices=sorted(DATASETS), action="append",
                    help="dataset to prepare (repeatable); default: all")
    args = ap.parse_args()

    for name in (args.name or sorted(DATASETS)):
        print("preparing %s ..." % name)
        build(name, DATASETS[name])


if __name__ == "__main__":
    main()
