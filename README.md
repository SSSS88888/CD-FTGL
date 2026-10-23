# CD-FTGL — Decentralized Online Convex Optimization with Compressed Communication

**English** · [简体中文](README.zh-CN.md)

Official code and reproduction package for

> **Revisiting Decentralized Online Convex Optimization with Compressed Communication**
> Hao Zhou, Xiaoyu Wang, Chang Yao, Mingli Song, Yuanyu Wan · Zhejiang University
> arXiv:2607.01665 — <https://arxiv.org/abs/2607.01665>

The paper proposes two FTRL-type algorithms for D-OCO with compressed communication:

* **CD-FTGL** (`CD_FTGL.py`, `feedback_type='full'`) — full-information setting, Choco-Gossip with `K = L`;
* **CD-FTBL** (`CD_FTGL.py`, `feedback_type='one-bandit'`) — one-point bandit feedback, Choco-Gossip with `K = L'` per block.

Baselines: Top-DOGD / Top-DOBD-1 (`Top_DOGD.py`) and DC-DOGD / DC-DOBD (`DC_DOGD.py`).

This repository is the **minimal artifact** for the paper: it contains the algorithm code, the experiment drivers,
and the exact per-run results plus the best hyper-parameter tables used in the paper. **One command regenerates all
nine figures without re-running any experiment.**

---

## Quick start

```bash
conda create -n cdftgl python=3.10 && conda activate cdftgl
pip install -r requirements.txt

python experiment/reproduce_figures.py        # ~1 minute, CPU only
```

Output — the 36 panels behind the paper's 9 figures, in `figures/`:

```
fig1_a9a_loss_vs_rounds.pdf      fig1_a9a_loss_vs_bits.pdf
fig1_ijcnn1_loss_vs_rounds.pdf   fig1_ijcnn1_loss_vs_bits.pdf
...                              (fig1 … fig9, a9a and ijcnn1, both x-axes)
```

Each panel is written as both PDF (vector, used in the paper) and PNG (300 dpi).
The script prints the full list and warns if any panel is missing.

> Verified: with the pinned environment, all 36 regenerated panels are **byte-identical** (PNG) to the ones
> embedded in the paper. PDF bytes differ only in the embedded creation timestamp.

---

## Reproduced figures

| Paper | Setting | Panels | Output files |
|---|---|---|---|
| Fig. 1 | 9 nodes / 18 edges, ω = 0.1 | loss vs rounds, loss vs bits (a9a, ijcnn1) | `fig1_<ds>_loss_vs_{rounds,bits}` |
| Fig. 2 | 9 nodes / 18 edges, ω = 0.5 | same | `fig2_*` |
| Fig. 3 | 9 nodes / 18 edges, ω = 0.05 | same | `fig3_*` |
| Fig. 4 | 50 nodes / 100 edges, ω = 0.5 | same | `fig4_*` |
| Fig. 5 | 50 nodes / 100 edges, ω = 0.1 | same | `fig5_*` |
| Fig. 6 | 50 nodes / 100 edges, ω = 0.05 | same | `fig6_*` |
| Fig. 7 | L = 100, K ∈ {2,5,10,20,50}, ω = 0.5 | CD-FTBL only, loss vs rounds / bits | `fig7_*` |
| Fig. 8 | L = 100, K ∈ {2,5,10,20,50}, ω = 0.1 | same | `fig8_*` |
| Fig. 9 | L = 100, K ∈ {2,5,10,20,50}, ω = 0.05 | same | `fig9_*` |

All panels compare CD-FTGL / CD-FTBL against Top-DOGD, Top-DOBD-1, DC-DOGD, DC-DOBD
(5 random seeds per configuration; solid line = mean over seeds, shaded band = min–max range).

---

## What is in the box

```
.
├── CD_FTGL.py                    # CD-FTGL (full) + CD-FTBL (bandit)
├── Top_DOGD.py                   # Top-DOGD / Top-DOBD-1  (Yang et al., 2026)
├── DC_DOGD.py                    # DC-DOGD / DC-DOBD       (Tu et al., 2022)
├── compressor.py                 # top-k / random-k / gossip / quantization compressors + bit accounting
├── logistic_base.py              # online logistic loss and gradient
├── parameter_CDFTGL.py           # theory-driven hyper-parameters (γ, L, h, ε)
├── parameter_TopDOGD.py
├── parameter_DCDOGD.py
├── data/
│   ├── prepare_data.py           # build the .npy matrices from the raw LIBSVM files
│   └── README.md                 # where to download a9a / ijcnn1
├── experiment/
│   ├── reproduce_figures.py      # ← regenerates all figures from the shipped results
│   ├── grid_search_algorithms.py     # grid search, 9 nodes
│   ├── grid_search_algorithms_n50.py # grid search, 50 nodes
│   ├── compare_best_algorithms.py    # select the best hyper-parameters per setting
│   ├── compare_best_algorithms_Lp.py # Fig. 7-9 (varying K at L = 100)
│   ├── plot_best_multiple_runs.py    # Fig. 1-6
│   └── result/
│       ├── grid_search_repeated_n9/  # per-run results (9 nodes)
│       ├── grid_search_repeated_n50/ # per-run results (50 nodes)
│       └── repeated_best_algorithms_comparison_{n9,n50}/best_parameters_tables/*.csv
└── figures/                      # generated (not tracked)
```

### Result files

Every run is stored as two `.npy` files (filename encodes the configuration):

* `*_SR.npy` — average cumulative loss of the **worst learner** per communication round
  (`max_i (1/t) Σ f(x_i(τ))` on the global test split);
* `*_Bits.npy` — cumulative number of transmitted bits at the same communication rounds.

The shipped subset (`experiment/result/**`, ~1,500 files, 60 MB) is exactly the set of files that the 36 figure panels
read — nothing else from the full experimental campaign is included.

The four CSVs in `best_parameters_tables/` are the **hyper-parameters actually used for the paper**
(h / η / ε scaling coefficients and the block size L, per algorithm, dataset, compression ratio and setting).
`reproduce_figures.py` reads them, so no tuning is performed at reproduction time.

---

## Reproducing the results from scratch (optional)

The figures above can be regenerated without training. To re-run the experiments themselves:

**1. Data.** Download `a9a` and `ijcnn1` from LIBSVM (see `data/README.md`), then

```bash
python data/prepare_data.py            # writes data/data_a9a/A_a9a.npy, ... and data/data_ijcnn1/*
```

**2. Grid search.** Every script resolves its paths relative to its own location, so any of the commands below can be
run from the repository root (or anywhere else). Outputs go to `experiment/result/`.

```bash
python experiment/grid_search_algorithms.py       # n = 9,  T = 10000, w ∈ {0.05, 0.1, 0.5}, seeds 42/123/456/789/1000
python experiment/grid_search_algorithms_n50.py   # n = 50
```

The configuration block at the top of each script selects dataset / compressor / topology / feedback type / grid;
completed runs are skipped automatically, so the search can be interrupted and resumed (`--start/--end` slices a run).

**3. Select hyper-parameters.**

```bash
python experiment/compare_best_algorithms.py      # writes experiment/result/repeated_best_algorithms_comparison_*/best_parameters_tables/*.csv
```

**4. Figures.**

```bash
python experiment/reproduce_figures.py
```

To regenerate the K-sensitivity study (Fig. 7-9) from scratch, set `L_PRIME_VALUES = [2, 5, 10, 20, 50]`
in `grid_search_algorithms.py` (default is `None`, i.e. `K = ⌈L/5⌉` as in all other experiments).

---

## Experimental setup (as in the paper)

| Item | Value |
|---|---|
| Loss | `f_{t,i}(x) = ln(1 + exp(-b·⟨a, x⟩))` (online logistic regression) |
| Decision set | `K = {x : ‖x‖₂ ≤ τ}`, τ = 10 |
| Datasets | a9a (32,561 × 123); ijcnn1 (141,691 × 22 = 49,990 train + 91,701 test, as in the code) |
| Rounds | T = 10,000, batch size 1 |
| Network | n = 9, G(n, 2n) random graph (NetworkX, seed 42); n = 50, 4-regular graph with 100 edges |
| Gossip matrix | local-degree weights: `A_ij = 1/max(d_i, d_j)`, `A_ii = 1 − Σ_{j≠i} A_ij` |
| Compressor | Top-k with compression ratio ω ∈ {0.05, 0.1, 0.5} |
| Bandit | one-point estimator, exploration radius ε (grid-searched over {0.1, 1, 10} in earlier sweeps) |
| Repeats | 5 seeds |

### Implementation notes

* The plotted quantity is the average loss of the **worst learner** at each communication round
  (`CD_FTGL.py:max_regret`); the paper writes it as `AL(t, i)` for a fixed learner `i`.
* In the bandit setting only the first `K = L'` rounds of each block communicate, hence the CD-FTBL curves cover
  `≈ T·K/L ≈ 2,000` rounds while the baselines cover all 10,000 — this is the communication saving reported in the paper.
* The bandit decision update projects onto the full ball `‖x‖ ≤ R` (the `(1 − ε/r)K` shrinkage of Algorithm 3 is not
  applied in the code); with ε = 1 and R = 10 this has no measurable effect on the reported curves.

---

## Environment

Python 3.10, `numpy==2.2.6`, `pandas==2.3.3`, `matplotlib==3.10.7`, `networkx==3.4.2`, `scipy==1.15.3`, `tqdm==4.67.1`.
`scikit-learn` is only needed by `data/prepare_data.py`.

Figures were produced with matplotlib 3.10.7; a different matplotlib version reproduces the same curves but may change
pixel-level rendering details.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{zhou2026revisiting,
  title   = {Revisiting Decentralized Online Convex Optimization with Compressed Communication},
  author  = {Zhou, Hao and Wang, Xiaoyu and Yao, Chang and Song, Mingli and Wan, Yuanyu},
  journal = {arXiv preprint arXiv:2607.01665},
  year    = {2026}
}
```
