import numpy as np
import json
from pathlib import Path
import tqdm

SKEL_DIR = Path("./skeleton_canonicalized")
TEXT_DIR = Path("./text")
OUTPUT_DIR = Path("./skeleton_split")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    skel_cache = {}
    for f in sorted(SKEL_DIR.glob("*.npy")):
        skel_cache[f.stem] = np.load(f)

    json_files = sorted(TEXT_DIR.glob("*.json"))
    for json_file in tqdm.tqdm(json_files):
        key = json_file.stem
        with open(json_file) as f:
            meta = json.load(f)

        seq_id = meta["seq_name"].replace("arctic_", "")
        skel = skel_cache[seq_id]
        window = skel[meta["frame_start"]:meta["frame_end"]].copy()
        window = window - window[0, 1, 0]
        np.save(OUTPUT_DIR / f"{key}.npy", window)

    print(f"Done. {len(list(OUTPUT_DIR.glob('*.npy')))} files.")


if __name__ == "__main__":
    main()
