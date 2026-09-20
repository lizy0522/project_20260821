"""Task entrypoint wrapper for the promoted shared support runner."""

from retrieval_oriented_model_selection.shared.mp10_k9_retrieval_runner import *  # noqa: F401,F403
from retrieval_oriented_model_selection.shared.mp10_k9_retrieval_runner import main

if __name__ == "__main__":
    main()
