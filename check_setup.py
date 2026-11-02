"""
check_setup.py  -  PHASE 1: checks that Python and all libraries are installed.

Run:   python -m utils.check_setup
"""
import importlib
import sys

import config

LIBRARIES = ["numpy", "pandas", "sklearn", "joblib", "matplotlib", "cv2", "PIL",
             "tensorflow", "xgboost", "streamlit", "reportlab"]


def main():
    print(f"Python version: {sys.version.split()[0]}")
    bad = 0
    for lib in LIBRARIES:
        try:
            mod = importlib.import_module(lib)
            print(f"  OK       {lib:<12}{getattr(mod, '__version__', '')}")
        except Exception as e:
            bad += 1
            print(f"  MISSING  {lib:<12}({type(e).__name__})")
    config.ensure_dirs()
    n_raw = sum(1 for _ in config.RAW_DIR.rglob("*.jp*g"))
    print(f"\nImages found in dataset_raw/: {n_raw}")
    print("Project folders are ready." if not bad else
          "\nSome libraries are missing. Run:  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
