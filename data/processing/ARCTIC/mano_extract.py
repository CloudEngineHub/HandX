#!/usr/bin/env python3
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

ROOT = Path("./raw_seqs")
OUT_DIR = Path("./temp")


def arctic_to_viz_dict(side_dict):
    rot = np.asarray(side_dict["rot"], dtype=np.float32)
    pose = np.asarray(side_dict["pose"], dtype=np.float32)
    trans = np.asarray(side_dict["trans"], dtype=np.float32)
    betas = np.asarray(side_dict["shape"], dtype=np.float32)
    T = rot.shape[0]
    pose48 = np.concatenate([rot, pose], axis=1).astype(np.float32)
    shapeT = np.tile(betas[None, :], (T, 1)).astype(np.float32)
    return {"pose": pose48, "trans": trans, "shape": shapeT}


def process_file(mano_npy_path, out_dir, prefix):
    d = np.load(mano_npy_path, allow_pickle=True).item()
    left_viz = arctic_to_viz_dict(d["left"])
    right_viz = arctic_to_viz_dict(d["right"])
    out = {"left": left_viz, "right": right_viz}

    base = mano_npy_path.stem.replace(".mano", "")
    stem = f"{prefix}_{base}_viz" if prefix else f"{base}_viz"

    with open(out_dir / f"{stem}.pkl", "wb") as f:
        pickle.dump(out, f, protocol=4)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subs = sorted([p for p in ROOT.glob("s*") if p.is_dir()])

    for sub in subs:
        mano_files = sorted(sub.glob("*.mano.npy"))
        for f in tqdm(mano_files, desc=sub.name):
            process_file(f, OUT_DIR, prefix=sub.name)


if __name__ == "__main__":
    main()
