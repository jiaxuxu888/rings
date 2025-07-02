.. RINGS: Relevant Information in Node Features and Graph Structure documentation master file, created by
   sphinx-quickstart on Mon Jun 30 10:20:28 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RINGS: Relevant Information in Node Features and Graph Structure documentation
==============================================================================

.. note::
   🚧 This project is under active development. 🚧

   We've made it public early to invite feedback, discussion, and transparency as we transition from research prototypes to a stable, user-friendly package.
   
   In the coming weeks, we’ll be releasing updates, architectural notes, and implementation details via a series of pull requests. You're welcome to follow along, open issues, or suggest improvements!

|

.. image:: _static/rings-logo.svg
   :width: 150
   :height: 150
   :alt: RINGS logo
   :align: left


Welcome to the official repository for the 2025 ICML paper, `No Metric to Rule Them All: Toward Principled Evaluations of Graph-Learning Datasets <https://doi.org/10.48550/arXiv.2502.02379>`__, which introduces RINGS: a perturbation framework for attributed graphs, designed to facilitate more principled evaluations of graph learning benchmarks from first principles. 

The full repo is available `here <https://github.com/aidos-lab/rings>`__.

|

💍 Framework Overview
----------------------

We are developing a community-friendly implementation of the **RINGS** framework introduced in the paper. Our goal is to make it easy for the graph-learning community to:

- Apply dataset perturbations tailored to graph-learning datasets
- Conduct more rigorous and insightful evaluations of both datasets and models
- Promote better dataset practices and evaluation hygiene across the field

If you have feedback on the paper or suggestions for how this package could better integrate with popular frameworks, please feel free to reach out to the authors.

---

📦 Installation
----------------------

RINGS uses `uv <https://github.com/astral-sh/uv>`_ as the package manager, which provides faster dependency resolution and installation.

Prerequisites
^^^^^^^^^^^^^^^

Install ``uv`` if you don’t have it already::

    pip install uv

Installing RINGS
^^^^^^^^^^^^^^^^^^^^

Clone the repository and install dependencies using ``uv``::

    # Clone the repository
    git clone https://github.com/aidos-lab/rings.git
    cd rings

    # Install dependencies
    uv sync

    # Activate environment
    source .venv/bin/activate

---

🔑 Key Components
----------------------

Mode Perturbations
^^^^^^^^^^^^^^^^^^^^^^

RINGS provides several perturbation transforms that can be applied to graph datasets:

**Node Feature Perturbations:**

- ``Original``: Keeps node features unchanged (baseline)
- ``EmptyFeatures``: Replaces node features with zero vectors
- ``RandomFeatures``: Replaces node features with random values
- ``CompleteFeatures``: Assigns unique one-hot vectors to nodes

**Graph Structure Perturbations:**

- ``EmptyGraph``: Removes all edges from the graph
- ``CompleteGraph``: Connects every pair of nodes
- ``RandomGraph``: Generates a random graph structure
- ``RandomConnectedGraph``: Generates a random graph that is guaranteed to be connected

SeparabilityFunctor
^^^^^^^^^^^^^^^^^^^^^^^

The ``SeparabilityFunctor`` computes statistically rigorous comparisons between multiple distributions to determine if they differ significantly. This is useful for:

- Evaluating whether different graph perturbations produce statistically distinct model performances
- Identifying which perturbations most impact model behavior
- Making rigorous, statistically valid claims about distribution separability

It employs statistical tests with permutation testing and built-in correction for multiple hypotheses (Bonferroni correction).

Available comparators include:

- ``KSComparator``: Kolmogorov–Smirnov test for comparing distributions
- ``WilcoxonComparator``: Wilcoxon signed-rank test for paired comparisons

ComplementarityFunctor
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ComplementarityFunctor`` measures the alignment between node features and graph structure by comparing their induced metric spaces. It can help you understand:

- Whether node features and graph structure contain complementary information
- How different perturbations affect this complementarity
- The distribution of information content across modalities in graph datasets

---

🔍 Example Usage
----------------------

The repository includes example scripts that demonstrate how to use RINGS to analyze graph datasets.

Performance Separability
^^^^^^^^^^^^^^^^^^^^^^^^^^

Measure distances between performance distributions to test whether the original dataset statistically outperforms perturbed versions.

Run a basic separability analysis with the default KS comparator::

    python -m examples.separability --comparator ks --alpha 0.05

Use the Wilcoxon test comparator::

    python -m examples.separability --comparator wilcoxon --alpha 0.01

Get help and see all available options::

    python -m examples.separability --help

The script analyzes and compares distributions from synthetic data, showing how to determine if differences are statistically significant.

Mode Complementarity
^^^^^^^^^^^^^^^^^^^^^^^^^

Assess the complementarity of geometric information in the metric spaces of node features and graph structure.

Run the example on the MUTAG dataset with original (unperturbed) graphs::

    python -m examples.complementarity --dataset MUTAG --perturbation original

Try different perturbations::

    python -m examples.complementarity --dataset MUTAG --perturbation random-features

Get help and see all available options::

    python -m examples.complementarity --help

The script outputs complementarity statistics that measure how well node features align with graph structure in the dataset.



🔍 Table of Contents
---------------------

.. toctree::
   :maxdepth: 2
   :caption: RINGS

   perturbations
   utils

.. toctree::
   :maxdepth: 2
   :caption: 🔑 Performance Separability

   separability/comparator
   separability/functor

.. toctree::
   :maxdepth: 2
   :caption: 🔑 Mode Complementarity

   complementarity/comparator
   complementarity/functor
   complementarity/metrics
   complementarity/utils

.. toctree::
   :maxdepth: 2
   :caption: 🔍 Example Scripts

   examples/complementarity
   examples/separability


📚 Citation
-------------

If you use RINGS in your research, please cite our paper:

.. code-block:: bibtex

   @inproceedings{
   coupette2025metric,
   title={No Metric to Rule Them All: Toward Principled Evaluations of Graph-Learning Datasets},
   author={Corinna Coupette and Jeremy Wayland and Emily Simons and Bastian Rieck},
   booktitle={Forty-second International Conference on Machine Learning},
   year={2025},
   url={https://openreview.net/forum?id=XbmBNwrfG5}
   }




.. image:: _static/aidos_logo.png
   :width: 120
   :height: 120
   :alt: SCOTT logo
   :align: left

|
| **Interested in more of our work?**
|
| See what we are working on at `AIDOS Lab <https://aidos.group>`_ or check out our `GitHub <https://github.com/aidos-lab>`_.
