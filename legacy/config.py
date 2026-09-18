"""
config.py

Central configuration for the Neuroimmune Graph Project.

Handles project paths so the codebase runs unchanged in three environments:
  1. Local development (paths resolved relative to this file)
  2. A cloned GitHub repository (same as above)
  3. Google Colab (paths resolved relative to /content, unless overridden)

Every other script in this project imports PROJECT_ROOT and DIRS from here
rather than hardcoding paths, so the project only needs to be configured
once, in one place.
"""

import os


def _running_in_colab() -> bool:
    """Detect whether the current process is running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_project_root() -> str:
    """
    Resolve the project root directory.

    Priority:
      1. PROJECT_ROOT environment variable, if set (works in any environment).
      2. /content/neuroimmune-graph-project, if running in Colab.
      3. The directory containing this file, otherwise.
    """
    env_override = os.environ.get("PROJECT_ROOT")
    if env_override:
        return os.path.abspath(env_override)

    if _running_in_colab():
        return "/content/neuroimmune-graph-project"

    return os.path.dirname(os.path.abspath(__file__))


IN_COLAB = _running_in_colab()
PROJECT_ROOT = _resolve_project_root()

# Subdirectories used throughout the project. Every phase writes its
# outputs into one of these, so downstream phases always know where to
# look for upstream results.
DIRS = {
    "models": os.path.join(PROJECT_ROOT, "models"),
    "sims": os.path.join(PROJECT_ROOT, "sims"),
    "graphs": os.path.join(PROJECT_ROOT, "graphs"),
    "results": os.path.join(PROJECT_ROOT, "results"),
    "notebooks": os.path.join(PROJECT_ROOT, "notebooks"),
}


if __name__ == "__main__":
    # Quick manual sanity check: `python config.py`
    print(f"Running in Colab: {IN_COLAB}")
    print(f"Project root:     {PROJECT_ROOT}")
    for name, path in DIRS.items():
        print(f"  {name:10s} -> {path}")
