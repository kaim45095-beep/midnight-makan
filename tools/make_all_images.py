"""
make_all_images.py
------------------
Runs every image generator script one after another, so the whole
art set of Midnight Makan can be rebuilt with one command:

    python tools/make_all_images.py
"""
import os
import subprocess
import sys
import time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "make_backgrounds.py",
    "make_characters.py",
    "make_food_props.py",
    "make_ui_particles.py",
]


def main():
    results = []
    for name in SCRIPTS:
        path = os.path.join(TOOLS_DIR, name)
        if not os.path.exists(path):
            print(f"-- skipping {name} (file not found)")
            results.append((name, "missing", 0.0))
            continue
        print(f"== running {name}")
        start = time.time()
        finished = subprocess.run([sys.executable, path])
        seconds = time.time() - start
        status = "ok" if finished.returncode == 0 else f"error {finished.returncode}"
        results.append((name, status, seconds))

    print("\nSummary")
    for name, status, seconds in results:
        print(f"  {name:24s} {status:10s} {seconds:6.1f} s")


if __name__ == "__main__":
    main()
