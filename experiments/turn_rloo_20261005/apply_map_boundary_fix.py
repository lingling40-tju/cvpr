"""Guard an isolated VLN-CE Habitat waypoint overlay against edge pixels.

Run on vlnce_server/VLN_CE/habitat_extensions/maps.py in the isolated
ActiveVLN checkout before reproducing the paired val-seen evaluation.
"""
import argparse
from pathlib import Path

OLD = "            if img[r_x, r_y]:\n"
NEW = (
    "            if 0 <= r_x < img.shape[0] and "
    "0 <= r_y < img.shape[1] and img[r_x, r_y]:\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maps_py", type=Path)
    args = ap.parse_args()
    source = args.maps_py.read_text()
    if source.count(NEW) == 1 and OLD not in source:
        print("bounds guard already present")
        return
    if source.count(OLD) != 1 or NEW in source:
        raise ValueError("unexpected map renderer source; inspect before patching")
    args.maps_py.write_text(source.replace(OLD, NEW))
    print("added bounds guard")


if __name__ == "__main__":
    main()
