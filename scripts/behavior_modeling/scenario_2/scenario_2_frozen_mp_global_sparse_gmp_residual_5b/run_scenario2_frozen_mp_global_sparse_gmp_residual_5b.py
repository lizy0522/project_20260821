"""Task entrypoint wrapper for the promoted shared support runner."""

from behavior_modeling.shared.frozen_mp_global_sparse_gmp_residual_support import *  # noqa: F401,F403
from behavior_modeling.shared.frozen_mp_global_sparse_gmp_residual_support import main

if __name__ == "__main__":
    main()
