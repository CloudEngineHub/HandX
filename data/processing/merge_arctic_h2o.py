import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm

HANDX_DIR = Path("../handx")
ARCTIC_DIR = Path("./ARCTIC")
H2O_DIR = Path("./H2O")
SOURCE_LICENSES = {
    "arctic": "ARCTIC Data & Software Copyright License for non-commercial scientific research purposes",
    "h2o": "H2O Dataset Terms of Use",
}


def make_source_key(source, raw_key):
    return raw_key if raw_key.startswith(f"{source}_") else f"{source}_{raw_key}"


def load_extra_data(dataset_dir):
    entries = {"train": [], "test": []}
    source = dataset_dir.name.lower()
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

        entry = {
            "key": make_source_key(source, key),
            "raw_key": key,
            "source_dataset": dataset_dir.name,
            "meta": meta,
            "can": can_entry,
            "mano": mano,
        }

        entries[meta["split"]].append(entry)

    return entries


def load_metadata():
    path = HANDX_DIR / "source_metadata.json"
    if not path.exists():
        return {"train": {}, "test": {}}
    with open(path) as f:
        metadata = json.load(f)
    metadata.setdefault("train", {})
    metadata.setdefault("test", {})
    return metadata


def metadata_entry(entry):
    source = entry["source_dataset"].lower()
    meta = entry["meta"]
    return {
        "source_dataset": entry["source_dataset"],
        "source_key": entry["raw_key"],
        "source_sequence_id": meta.get("seq_name"),
        "frame_start": meta.get("frame_start"),
        "frame_end": meta.get("frame_end"),
        "license": SOURCE_LICENSES.get(source),
    }


def main():
    print("Loading ARCTIC...")
    arctic = load_extra_data(ARCTIC_DIR)
    print(f"  train: {len(arctic['train'])}, test: {len(arctic['test'])}")

    print("Loading H2O...")
    h2o = load_extra_data(H2O_DIR)
    print(f"  train: {len(h2o['train'])}, test: {len(h2o['test'])}")

    source_metadata = load_metadata()

    for split in ["train", "test"]:
        print(f"\nProcessing {split}...")
        existing_can = np.load(HANDX_DIR / f"{split}_can_pos_all_wotextfeat.npz", allow_pickle=True)
        existing_mano = np.load(HANDX_DIR / f"{split}_mano.npz", allow_pickle=True)
        n_existing = len(existing_can.files)

        extra = arctic[split] + h2o[split]
        print(f"  Existing: {n_existing}, Adding/updating: {len(extra)}")

        new_can = {k: existing_can[k] for k in existing_can.files}
        new_mano = {k: existing_mano[k] for k in existing_mano.files}

        for entry in tqdm(extra, desc=f"  Adding {split}"):
            key = entry["key"]
            new_can[key] = entry["can"]
            new_mano[key] = entry["mano"]
            source_metadata[split][key] = metadata_entry(entry)

        print(f"  Saving...")
        np.savez(HANDX_DIR / f"{split}_can_pos_all_wotextfeat.npz", **new_can)
        np.savez(HANDX_DIR / f"{split}_mano.npz", **new_mano)

    metadata_path = HANDX_DIR / "source_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(source_metadata, f, indent=2)
    print(f"\nSaved source metadata to {metadata_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
