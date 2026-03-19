import numpy as np
from pathlib import Path
import tqdm

INPUT_DIR = Path("./skeleton")
OUTPUT_DIR = Path("./skeleton_canonicalized")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    skeleton_files = sorted(INPUT_DIR.glob("*.npy"))

    for skel_file in tqdm.tqdm(skeleton_files):
        skeleton = np.load(skel_file)
        skeleton[..., 0] = -skeleton[..., 0]
        skeleton[..., 1] = -skeleton[..., 1]
        np.save(OUTPUT_DIR / skel_file.name, skeleton)


if __name__ == "__main__":
    main()
