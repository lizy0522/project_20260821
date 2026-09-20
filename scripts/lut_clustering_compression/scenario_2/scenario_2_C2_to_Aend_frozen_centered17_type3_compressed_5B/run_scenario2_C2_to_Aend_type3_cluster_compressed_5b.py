"""Task entrypoint wrapper for the promoted shared support runner."""

from lut_clustering_compression.shared.type3_retrieval_support import *  # noqa: F401,F403
from lut_clustering_compression.shared.type3_retrieval_support import main

if __name__ == "__main__":
    main()
