"""Apply Mode Perturbations and Use the Complementarity Functor"""

import numpy as np
from torch_geometric.datasets import TUDataset

from rings.complementarity import ComplementarityFunctor, MatrixNormComparator
from rings.perturbations import Original, RandomFeatures

if __name__ == "__main__":
    # Load MUTAG dataset
    dataset = TUDataset(root="/tmp/TUDataset", name="MUTAG")

    # Create original and perturbed datasets
    original_transform = Original()
    perturbed_transform = RandomFeatures(
        shuffle=False
    )  # Using RandomFeatures as an example

    # Apply transforms to create two versions of the dataset
    original_dataset = [original_transform(data.clone()) for data in dataset]
    perturbed_dataset = [perturbed_transform(data.clone()) for data in dataset]

    # Set up the complementarity functor with appropriate metrics and comparator
    functor = ComplementarityFunctor(
        feature_metric="euclidean",  # Metric for node features
        graph_metric="shortest_path_distance",  # Metric for graph structure
        comparator=MatrixNormComparator,  # Comparator for metric spaces
        n_jobs=-1,  # Use all available cores
        norm="L11",  # Use L11 norm for the comparator
    )

    # Compute complementarity for each graph in the dataset
    original_scores = []
    rf_scores = []

    for original, perturbed in zip(original_dataset, perturbed_dataset):
        # Create a batch containing both original and perturbed versions

        # Compute complementarity
        original_result = functor(original)
        perturbed_result = functor(perturbed)

        print(original_result)
        import sys

        sys.exit()

        original_scores.append(original_result["complementarity"].item())
        rf_scores.append(perturbed_result["complementarity"].item())

    # Convert scores to numpy array for statistics
    original_scores = np.array(original_scores)
    rf_scores = np.array(rf_scores)

    # Calculate and print statistics
    mean_score = np.mean(original_scores)
    std_score = np.std(original_scores)

    rf_score = np.mean(rf_scores)
    rf_std = np.std(rf_scores)

    print(f"Original Scores Statistics for MUTAG Dataset:")
    print(f"Transform: {original_transform.__class__.__name__}")
    print(f"Mean Complementarity: {mean_score:.4f}")
    print(f"Standard Deviation: {std_score:.4f}")
    print(f"Random Features Scores Statistics for MUTAG Dataset:")
    print(f"Transform: {perturbed_transform.__class__.__name__}")
    print(f"Mean Complementarity: {rf_score:.4f}")
    print(f"Standard Deviation: {rf_std:.4f}")
