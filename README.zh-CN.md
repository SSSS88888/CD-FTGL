# CD-FTGL —— 带压缩通信的分布式在线凸优化

[English](README.md) · **简体中文**

本仓库是下述论文的官方代码与复现包：

> **Revisiting Decentralized Online Convex Optimization with Compressed Communication**
> Hao Zhou, Xiaoyu Wang, Chang Yao, Mingli Song, Yuanyu Wan · 浙江大学
> arXiv:2607.01665 —— <https://arxiv.org/abs/2607.01665>

论文针对带压缩通信的 D-OCO 提出两个 FTRL 型算法：

* **CD-FTGL**（`CD_FTGL.py`，`feedback_type='full'`）—— full-information 设置，Choco-Gossip 每块迭代 `K = L` 次；
* **CD-FTBL**（`CD_FTGL.py`，`feedback_type='one-bandit'`）—— 单点 bandit 反馈，Choco-Gossip 每块迭代 `K = L'` 次。

对比基线：Top-DOGD / Top-DOBD-1（`Top_DOGD.py`）与 DC-DOGD / DC-DOBD（`DC_DOGD.py`）。

本仓库是论文的**最小复现包**：包含算法代码、实验驱动脚本，以及论文实际使用的逐次运行结果与最优超参数表。
**一条命令即可重新生成论文的全部 9 张图，无需重跑任何实验。**

---

## 快速开始

```bash
conda create -n cdftgl python=3.10 && conda activate cdftgl
pip install -r requirements.txt

python experiment/reproduce_figures.py        # 约 1 分钟，仅用 CPU
```

输出：论文 9 张图对应的 36 个子图，位于 `figures/` 目录：

```
fig1_a9a_loss_vs_rounds.pdf      fig1_a9a_loss_vs_bits.pdf
fig1_ijcnn1_loss_vs_rounds.pdf   fig1_ijcnn1_loss_vs_bits.pdf
...                              （fig1 … fig9，a9a 与 ijcnn1，两种横轴）
```

每个子图同时输出 PDF（矢量图，论文所用格式）与 PNG（300 dpi）。脚本会打印完整清单，若有缺图会告警。

> 已验证：在固定版本的环境下，重新生成的 36 个子图与论文中嵌入的图**逐字节相同**（PNG）。
> PDF 仅在嵌入的创建时间戳上不同。

---

## 复现的图

| 论文图 | 设置 | 子图 | 输出文件 |
|---|---|---|---|
| Fig. 1 | 9 节点 / 18 边，ω = 0.1 | 平均损失 vs 通信轮数 / vs 通信比特（a9a、ijcnn1） | `fig1_<ds>_loss_vs_{rounds,bits}` |
| Fig. 2 | 9 节点 / 18 边，ω = 0.5 | 同上 | `fig2_*` |
| Fig. 3 | 9 节点 / 18 边，ω = 0.05 | 同上 | `fig3_*` |
| Fig. 4 | 50 节点 / 100 边，ω = 0.5 | 同上 | `fig4_*` |
| Fig. 5 | 50 节点 / 100 边，ω = 0.1 | 同上 | `fig5_*` |
| Fig. 6 | 50 节点 / 100 边，ω = 0.05 | 同上 | `fig6_*` |
| Fig. 7 | L = 100，K ∈ {2,5,10,20,50}，ω = 0.5 | 仅 CD-FTBL，损失 vs 轮数 / 比特 | `fig7_*` |
| Fig. 8 | L = 100，K ∈ {2,5,10,20,50}，ω = 0.1 | 同上 | `fig8_*` |
| Fig. 9 | L = 100，K ∈ {2,5,10,20,50}，ω = 0.05 | 同上 | `fig9_*` |

所有子图都是 CD-FTGL / CD-FTBL 与 Top-DOGD、Top-DOBD-1、DC-DOGD、DC-DOBD 的对比
（每个配置 5 个随机种子；实线为种子均值，阴影带为最小值—最大值范围）。

---

## 仓库内容

```
.
├── CD_FTGL.py                    # CD-FTGL（full）+ CD-FTBL（bandit）
├── Top_DOGD.py                   # Top-DOGD / Top-DOBD-1（Yang et al., 2026）
├── DC_DOGD.py                    # DC-DOGD / DC-DOBD（Tu et al., 2022）
├── compressor.py                 # top-k / 随机稀疏 / gossip / 量化压缩器 + 比特计数
├── logistic_base.py              # 在线逻辑回归损失与梯度
├── parameter_CDFTGL.py           # 由理论公式给出的超参数（γ、L、h、ε）
├── parameter_TopDOGD.py
├── parameter_DCDOGD.py
├── data/
│   ├── prepare_data.py           # 由原始 LIBSVM 文件构建 .npy 矩阵
│   └── README.md                 # a9a / ijcnn1 的下载地址
├── experiment/
│   ├── reproduce_figures.py      # ← 由仓库内结果一键重新生成全部图
│   ├── grid_search_algorithms.py     # 网格搜索，9 节点
│   ├── grid_search_algorithms_n50.py # 网格搜索，50 节点
│   ├── compare_best_algorithms.py    # 为每个设置挑选最优超参数
│   ├── compare_best_algorithms_Lp.py # Fig. 7-9（L = 100 时改变 K）
│   ├── plot_best_multiple_runs.py    # Fig. 1-6
│   └── result/
│       ├── grid_search_repeated_n9/  # 逐次运行结果（9 节点）
│       ├── grid_search_repeated_n50/ # 逐次运行结果（50 节点）
│       └── repeated_best_algorithms_comparison_{n9,n50}/best_parameters_tables/*.csv
└── figures/                      # 生成结果（不纳入版本管理）
```

### 结果文件

每次运行存为两个 `.npy` 文件（文件名即记录了配置）：

* `*_SR.npy` —— 每个通信轮上**最差节点**的平均累计损失
  （在全局测试集上计算 `max_i (1/t) Σ f(x_i(τ))`）；
* `*_Bits.npy` —— 同一批通信轮上的累计传输比特数。

仓库内附带的结果子集（`experiment/result/**`，约 1,500 个文件、60 MB）**正好**是 36 个子图所读取的那批文件，
完整实验活动中其余的结果均未纳入。

`best_parameters_tables/` 中的 4 个 CSV 是**论文实际使用的超参数**
（h / η / ε 的缩放系数与块大小 L，按算法、数据集、压缩比与设置分别给出）。
`reproduce_figures.py` 直接读取这些表，因此复现时不进行任何调参。

---

## 从零复现实验结果（可选）

上面的图无需训练即可重新生成。若要真正重跑实验：

**1. 数据。** 从 LIBSVM 下载 `a9a` 与 `ijcnn1`（见 `data/README.md`），然后

```bash
python data/prepare_data.py            # 生成 data/data_a9a/A_a9a.npy 等以及 data/data_ijcnn1/*
```

**2. 网格搜索。** 所有脚本都以自身所在目录为基准解析路径，因此下面的命令在仓库根目录（或任意目录）下都能运行，
结果写入 `experiment/result/`。

```bash
python experiment/grid_search_algorithms.py       # n = 9，T = 10000，w ∈ {0.05, 0.1, 0.5}，种子 42/123/456/789/1000
python experiment/grid_search_algorithms_n50.py   # n = 50
```

每个脚本顶部的配置块用于选择数据集 / 压缩器 / 拓扑 / 反馈类型 / 网格；已完成的运行会自动跳过，
因此搜索可以随时中断并续跑（`--start/--end` 可切分任务）。

**3. 选择超参数。**

```bash
python experiment/compare_best_algorithms.py      # 写入 experiment/result/repeated_best_algorithms_comparison_*/best_parameters_tables/*.csv
```

**4. 出图。**

```bash
python experiment/reproduce_figures.py
```

若要从零重跑 K 敏感性实验（Fig. 7-9），在 `grid_search_algorithms.py` 中把
`L_PRIME_VALUES` 设为 `[2, 5, 10, 20, 50]`（默认 `None`，即与其他实验一致取 `K = ⌈L/5⌉`）。

---

## 实验设置（与论文一致）

| 项目 | 取值 |
|---|---|
| 损失 | `f_{t,i}(x) = ln(1 + exp(-b·⟨a, x⟩))`（在线逻辑回归） |
| 决策集 | `K = {x : ‖x‖₂ ≤ τ}`，τ = 10 |
| 数据集 | a9a（32,561 × 123）；ijcnn1（141,691 × 22 = 49,990 训练 + 91,701 测试，与代码一致） |
| 轮数 | T = 10,000，批大小 1 |
| 网络 | n = 9：G(n, 2n) 随机图（NetworkX，seed 42）；n = 50：100 条边的 4-正则图 |
| Gossip 矩阵 | local-degree 权重：`A_ij = 1/max(d_i, d_j)`，`A_ii = 1 − Σ_{j≠i} A_ij` |
| 压缩器 | Top-k，压缩比 ω ∈ {0.05, 0.1, 0.5} |
| Bandit | 单点估计器，探索半径 ε（早期轮次在 {0.1, 1, 10} 中网格搜索） |
| 重复次数 | 5 个随机种子 |

### 实现说明

* 图中画的是每个通信轮上**最差节点**的平均损失（`CD_FTGL.py:max_regret`）；论文中记作固定节点 `i` 的 `AL(t, i)`。
* bandit 设置下每个块只有前 `K = L'` 轮发生通信，因此 CD-FTBL 的曲线覆盖约 `T·K/L ≈ 2,000` 轮，
  而基线覆盖完整的 10,000 轮——这正是论文所报告的通信节省。
* bandit 的决策更新投影到完整的球 `‖x‖ ≤ R` 上（Algorithm 3 中的 `(1 − ε/r)K` 收缩在代码中未实现）；
  在 ε = 1、R = 10 时这对所报告的曲线没有可测量的影响。

---

## 运行环境

Python 3.10，`numpy==2.2.6`、`pandas==2.3.3`、`matplotlib==3.10.7`、`networkx==3.4.2`、`scipy==1.15.3`、`tqdm==4.67.1`。
`scikit-learn` 仅 `data/prepare_data.py` 需要。

论文中的图由 matplotlib 3.10.7 生成；其他 matplotlib 版本可复现同样的曲线，但像素级的渲染细节可能不同。

## 许可证

Apache License 2.0 —— 见 [LICENSE](LICENSE)。

## 引用

```bibtex
@article{zhou2026revisiting,
  title   = {Revisiting Decentralized Online Convex Optimization with Compressed Communication},
  author  = {Zhou, Hao and Wang, Xiaoyu and Yao, Chang and Song, Mingli and Wan, Yuanyu},
  journal = {arXiv preprint arXiv:2607.01665},
  year    = {2026}
}
```
