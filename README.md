# RINGS

This is the official repository for our ICML paper:

**[No Metric to Rule Them All: Toward Principled Evaluations of Graph-Learning Datasets](https://arxiv.org/abs/2502.02379)**

which introduces **RINGS**: a perturbation framework for attributed graphs, designed to enable more principled evaluations of graph-learning benchmarks from first principles.

---

## 🚧 Repository Status

This repository is **under active development**.  
We’re making it public early to invite feedback, discussion, and **transparency** as we move from research prototypes to a stable, user-friendly package.

In the coming weeks, we’ll release updates, architectural notes, and implementation details via a series of pull requests. You're welcome to follow along, open issues, or suggest improvements!

---

## 🚀 Current MVP Release

This initial Minimum Viable Product (MVP) includes:

- A set of **graph perturbation transformations** for manipulating node features and graph structure
- The **SeparabilityFunctor**, which enables statistical comparisons between distributions
- The **ComplementarityFunctor**, which computes mode complementarity between node features and graph structure
- Example scripts demonstrating usage with PyTorch Geometric datasets and toy performance distributions

---

## 💍 Framework Overview

We are developing a community-friendly implementation of the **RINGS** framework introduced in the paper. Our goal is to make it easy for the graph-learning community to:

- Apply dataset perturbations tailored to graph-learning datasets
- Conduct more rigorous and insightful evaluations of both datasets and models
- Promote better dataset practices and evaluation hygiene across the field

If you have feedback on the paper or suggestions for how this package could better integrate with popular frameworks, please feel free to reach out to the authors.

---

## 📦 Installation

RINGS uses [uv](https://github.com/astral-sh/uv) as the package manager, which provides faster dependency resolution and installation.

### Prerequisites

Install `uv` if you don’t have it already:

```bash
pip install uv
```

### Installing RINGS

Clone the repository and install dependencies using `uv`:

```bash
# Clone the repository
git clone https://github.com/aidos-lab/rings.git
cd rings

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

---

## 🔑 Key Components

### Mode Perturbations

RINGS provides several perturbation transforms that can be applied to graph datasets:

**Node Feature Perturbations:**

- `Original`: Keeps node features unchanged (baseline)
- `EmptyFeatures`: Replaces node features with zero vectors
- `RandomFeatures`: Replaces node features with random values
- `CompleteFeatures`: Assigns unique one-hot vectors to nodes

**Graph Structure Perturbations:**

- `EmptyGraph`: Removes all edges from the graph
- `CompleteGraph`: Connects every pair of nodes
- `RandomGraph`: Generates a random graph structure
- `RandomConnectedGraph`: Generates a random graph that is guaranteed to be connected

### SeparabilityFunctor

The `SeparabilityFunctor` computes statistically rigorous comparisons between multiple distributions to determine if they differ significantly. This is useful for:

- Evaluating whether different graph perturbations produce statistically distinct model performances
- Identifying which perturbations most impact model behavior
- Making rigorous, statistically valid claims about distribution separability

It employs statistical tests with permutation testing and built-in correction for multiple hypotheses (Bonferroni correction).

Available comparators include:

- `KSComparator`: Kolmogorov–Smirnov test for comparing distributions
- `WilcoxonComparator`: Wilcoxon signed-rank test for paired comparisons

### ComplementarityFunctor

The `ComplementarityFunctor` measures the alignment between node features and graph structure by comparing their induced metric spaces. It can help you understand:

- Whether node features and graph structure contain complementary information
- How different perturbations affect this complementarity
- The distribution of information content across modalities in graph datasets

---

## 🔍 Example Usage

The repository includes example scripts that demonstrate how to use RINGS to analyze graph datasets.

### Performance Separability

Measure distances between performance distributions to test whether the original dataset statistically outperforms perturbed versions.

```bash
# Run a basic separability analysis with the default KS comparator
python -m examples.separability --comparator ks --alpha 0.05
```

```bash
# Use the Wilcoxon test comparator
python -m examples.separability --comparator wilcoxon --alpha 0.01
```

```bash
# Get help and see all available options
python -m examples.separability --help
```

The script analyzes and compares distributions from synthetic data, showing how to determine if differences are statistically significant.

---

### Mode Complementarity

Assess the complementarity of geometric information in the metric spaces of node features and graph structure.

```bash
# Run the example on the MUTAG dataset with original (unperturbed) graphs
python -m examples.complementarity --dataset MUTAG --perturbation original
```

```bash
# Try different perturbations
python -m examples.complementarity --dataset MUTAG --perturbation random-features
```

```bash
# Get help and see all available options
python -m examples.complementarity --help
```

The script outputs complementarity statistics that measure how well node features align with graph structure in the dataset.

## 📚 Citation

If you use RINGS in your research, please cite our paper:

```bibtex
@inproceedings{
coupette2025metric,
title={No Metric to Rule Them All: Toward Principled Evaluations of Graph-Learning Datasets},
author={Corinna Coupette and Jeremy Wayland and Emily Simons and Bastian Rieck},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=XbmBNwrfG5}
}
```

Stay tuned for more examples and documentation!
