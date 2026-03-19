#!/usr/bin/env python3
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from smplx import MANO

MANO_PKL_DIR = "../../../diffusion/body_models/mano"
INPUT_DIR = "./temp"
OUTPUT_DIR = "./skeleton"
BATCH_SIZE = 512
FINGERTIP_INDICES = [745, 333, 444, 555, 672]


def forward_side(params, mano_model, device):
    pose = np.asarray(params["pose"], dtype=np.float32)
    trans = np.asarray(params["trans"], dtype=np.float32)
    T = pose.shape[0]
    betas = np.asarray(params["shape"], dtype=np.float32)
    if betas.ndim == 1:
        betas = np.tile(betas[None, :], (T, 1))

    out_all = np.zeros((T, 21, 3), dtype=np.float32)

    for s in range(0, T, BATCH_SIZE):
        e = min(T, s + BATCH_SIZE)
        with torch.no_grad():
            out = mano_model(
                betas=torch.from_numpy(betas[s:e]).float().to(device),
                global_orient=torch.from_numpy(pose[s:e, :3]).float().to(device),
                hand_pose=torch.from_numpy(pose[s:e, 3:]).float().to(device),
                transl=torch.from_numpy(trans[s:e]).float().to(device),
            )
            j16 = out.joints.cpu().numpy()
            tips = out.vertices.cpu().numpy()[:, FINGERTIP_INDICES, :]
            out_all[s:e, :16] = j16
            out_all[s:e, 16:] = tips

    return out_all


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    left_mano = MANO(model_path=MANO_PKL_DIR, is_rhand=False, use_pca=False, flat_hand_mean=True).eval().to(device)
    right_mano = MANO(model_path=MANO_PKL_DIR, is_rhand=True, use_pca=False, flat_hand_mean=True).eval().to(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".pkl")])

    for fname in tqdm(pkl_files):
        with open(os.path.join(INPUT_DIR, fname), "rb") as f:
            params = pickle.load(f)

        left_params = params.get("left")
        right_params = params.get("right")

        if left_params is not None:
            T = left_params["pose"].shape[0]
        else:
            T = right_params["pose"].shape[0]

        skeleton = np.zeros((T, 2, 21, 3), dtype=np.float32)

        if left_params is not None:
            skeleton[:, 0] = forward_side(left_params, left_mano, device)
        if right_params is not None:
            skeleton[:, 1] = forward_side(right_params, right_mano, device)

        out_path = os.path.join(OUTPUT_DIR, fname.replace(".pkl", ".npy"))
        np.save(out_path, skeleton)


if __name__ == "__main__":
    main()
