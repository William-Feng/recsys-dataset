from covis_matrices import generate_covis_matrices
from rerank import rerank


if __name__ == "__main__":
    # stage 1:
    # ensure covis matrices are already generated and located in the appropriate location before this
    generate_covis_matrices()

    # stage 2:
    rerank()
