"""
setup_environment.py

Phase 0 of the Neuroimmune Graph Project.

Responsibilities:
  1. Verify that required third-party packages are installed.
  2. Create the project's folder structure if it does not already exist.

This script has no dependencies on later phases and can be run on its own,
either from the command line or as the first cell of a notebook:

    Command line:
        python setup_environment.py

    Notebook / Google Colab:
        %run setup_environment.py
        # or, after cloning the repo into the working directory:
        !python setup_environment.py

Nothing in this file or the rest of the project assumes a specific machine
or personal file path — all paths are resolved through config.py, which
adapts automatically to a local checkout, a cloned repo, or Colab.
"""

import importlib

from config import DIRS, IN_COLAB, PROJECT_ROOT

# Package name -> import name, for cases where they differ (none currently,
# kept as a dict so it's easy to extend later, e.g. "scikit-learn": "sklearn").
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "networkx": "networkx",
    "pandas": "pandas",
}


def check_dependencies() -> bool:
    """Check that every required package can be imported. Return True if so."""
    missing = []
    for package_name, import_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print("Missing required packages:")
        for package_name in missing:
            print(f"  - {package_name}")
        print()
        print("Install them with:")
        print(f"    pip install {' '.join(missing)}")
        return False

    print("All required dependencies are installed:")
    for package_name in REQUIRED_PACKAGES:
        print(f"  - {package_name}")
    return True


def create_project_structure() -> None:
    """Create every directory in DIRS if it doesn't already exist."""
    import os

    for name, path in DIRS.items():
        os.makedirs(path, exist_ok=True)

        # A .gitkeep placeholder so empty directories are still tracked by git.
        gitkeep_path = os.path.join(path, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            open(gitkeep_path, "w").close()

        print(f"Ready: {path}")


def main() -> None:
    print(f"Environment: {'Google Colab' if IN_COLAB else 'local'}")
    print(f"Project root: {PROJECT_ROOT}\n")

    print("Checking dependencies...")
    deps_ok = check_dependencies()
    print()

    print("Setting up project folder structure...")
    create_project_structure()
    print()

    if deps_ok:
        print("Phase 0 complete. Ready for Phase 1.")
    else:
        print("Phase 0 complete, but install the missing packages above before starting Phase 1.")


if __name__ == "__main__":
    main()
