import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm

HANDX_DIR = Path("../handx")
ARCTIC_DIR = Path("./ARCTIC")
H2O_DIR = Path("./H2O")


def load_extra_data(dataset_dir):
    entries = {"train": [], "test": []}
    text_dir = dataset_dir / "text"
    skel_dir = dataset_dir / "skeleton_split"
    mano_dir = dataset_dir / "mano"

    json_files = sorted(text_dir.glob("*.json"))
    for json_file in tqdm(json_files, desc=f"  Loading {dataset_dir.name}"):
        key = json_file.stem
        with open(json_file) as f:
            meta = json.load(f)

        skel = np.load(skel_dir / f"{key}.npy")
        with open(mano_dir / f"{key}.pkl", "rb") as f:
            mano = pickle.load(f)

        can_entry = {
            "motion": skel,
            "left_annotation": [a["left"] for a in meta["annotations"]],
            "right_annotation": [a["right"] for a in meta["annotations"]],
            "interaction_annotation": [a["two_hands_relation"] for a in meta["annotations"]],
        }

        entries[meta["split"]].append((can_entry, mano))

    return entries


def main():
    print("Loading ARCTIC...")
    arctic = load_extra_data(ARCTIC_DIR)
    print(f"  train: {len(arctic['train'])}, test: {len(arctic['test'])}")

    print("Loading H2O...")
    h2o = load_extra_data(H2O_DIR)
    print(f"  train: {len(h2o['train'])}, test: {len(h2o['test'])}")

    for split in ["train", "test"]:
        print(f"\nProcessing {split}...")
        existing_can = np.load(HANDX_DIR / f"{split}_can_pos_all_wotextfeat.npz", allow_pickle=True)
        existing_mano = np.load(HANDX_DIR / f"{split}_mano.npz", allow_pickle=True)
        n_existing = len(existing_can.files)

        extra = arctic[split] + h2o[split]
        total = n_existing + len(extra)
        print(f"  Existing: {n_existing}, Adding: {len(extra)}, Total: {total}")

        new_can = {k: existing_can[k] for k in existing_can.files}
        new_mano = {k: existing_mano[k] for k in existing_mano.files}

        for j, (can_entry, mano_entry) in enumerate(tqdm(extra, desc=f"  Adding {split}")):
            new_can[str(n_existing + j)] = can_entry
            new_mano[str(n_existing + j)] = mano_entry

        print(f"  Saving...")
        np.savez(HANDX_DIR / f"{split}_can_pos_all_wotextfeat.npz", **new_can)
        np.savez(HANDX_DIR / f"{split}_mano.npz", **new_mano)

    print("\nDone.")


if __name__ == "__main__":
    main()
