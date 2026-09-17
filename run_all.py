#!/usr/bin/env python3
"""Reproduce the topological results of the paper in one go.

    python run_all.py                     # images in ./data/images, results in ./results
    OSTEO_TDA_IMAGES=/path/to/tifs python run_all.py

Steps (each is also runnable on its own):
  1. full_persistence.py    core pipeline, Figures 8, 9, 11, Table 1
  2. stats_tests.py         Mann-Whitney / PERMANOVA / permutation, Figure 10, Table 2
  3. void_size_spectrum.py  Figure 5
  4. diag_largescale.py     Figure 6
  5. state_diagram.py       Figure 7
  6. diag_color.py          Figure 4
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("OSTEO_TDA_IMAGES", os.path.join(HERE, "data", "images"))
os.environ.setdefault("OSTEO_TDA_RESULTS", os.path.join(HERE, "results"))
os.environ.setdefault("OSTEO_TDA_MASK", "fixed")
os.environ.setdefault("OSTEO_TDA_RESCALE", "508x377")
STEPS = ["full_persistence.py", "stats_tests.py", "void_size_spectrum.py",
         "diag_largescale.py", "state_diagram.py", "diag_color.py"]
t0 = time.time()
for s in STEPS:
    print(f"\n===== {s} =====", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE)
    if r.returncode:
        sys.exit(f"{s} failed with code {r.returncode}")
print(f"\nDone in {time.time()-t0:.0f} s. Results in {os.environ['OSTEO_TDA_RESULTS']}")
print(open(os.path.join(os.environ["OSTEO_TDA_RESULTS"], "full_validation_summary.txt")).read())
