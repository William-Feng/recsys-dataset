"""
This module is used to run the Covisitation Matrices on the dataset.
"""


from covis_matrices import generate_covis_matrices
from rerank import rerank


if __name__ == "__main__":
    # Ensure covisitation matrices are already generated and located in the appropriate location

    # Stage 1:
    generate_covis_matrices()

    # Stage 2:
    rerank()
