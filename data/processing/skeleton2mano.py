"""
Batch MANO fitting: fit MANO parameters to skeleton joint positions.

Reads skeleton windows from a directory of .npy files (shape: (60, 2, 21, 3))
and outputs per-clip MANO parameters as .pkl files.

Usage:
    python skeleton2mano.py --input_dir H2O/skeleton_split --output_dir H2O/mano
    python skeleton2mano.py --input_dir ARCTIC/skeleton_split --output_dir ARCTIC/mano
"""

import argparse
import glob
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from mano2mesh import left_manomodel, right_manomodel, ModifiedMANO


def fitting_loss(mano_output, target_joints, betas,
                 pose_preserve_loss_weight=2.0,
                 shape_prior_loss_weight=5.0,
                 pose_prior_loss_weight=0.000001,
                 initial_hand_pose=None):
    diff = mano_output.joints - target_joints
    joint_loss = (diff ** 2).sum(dim=0).mean()
    pose_prior_loss = torch.sum(mano_output.hand_pose ** 2)
    shape_prior_loss = torch.sum(betas ** 2)

    if initial_hand_pose is not None:
        pose_preserve_loss = torch.sum((mano_output.hand_pose - initial_hand_pose) ** 2)
    else:
        pose_preserve_loss = 0.0

    total_loss = (joint_loss
                  + shape_prior_loss_weight * shape_prior_loss
                  + pose_preserve_loss_weight * pose_preserve_loss
                  + pose_prior_loss_weight * pose_prior_loss)
    return total_loss


def fit_batch(skeleton_batch: torch.Tensor, mano_model: ModifiedMANO, device='cuda'):
    """
    Fit MANO parameters to a batch of skeleton sequences.

    Args:
        skeleton_batch: (B, 60, 21, 3) single-hand skeleton data on device
        mano_model: MANO model instance

    Returns:
        dict with shape (B, 60, *) arrays: shape, pose, trans
    """
    B, num_frames, J, _ = skeleton_batch.shape
    mano_model.to(device)

    target_wrist_pos_frame0 = skeleton_batch[:, 0, 0:1]  # (B, 1, 3)

    with torch.no_grad():
        zero_output = mano_model(
            global_orient=torch.zeros(B, 3, device=device),
            hand_pose=torch.zeros(B, 45, device=device),
            betas=torch.zeros(B, 10, device=device),
            transl=torch.zeros(B, 3, device=device)
        )
        initial_mano_wrist_pos = zero_output.joints[:, 0:1, :]  # (B, 1, 3)

    initial_translation = target_wrist_pos_frame0 - initial_mano_wrist_pos

    all_poses_list = []
    all_trans_list = []
    all_betas_list = []

    prev_pose = torch.zeros(B, 48, device=device)
    prev_trans = initial_translation[:, 0]  # (B, 3)
    prev_betas = torch.zeros(B, 10, device=device)

    for frame_idx in range(num_frames):
        target_joints = skeleton_batch[:, frame_idx]  # (B, 21, 3)

        init_global_orient = prev_pose[:, :3].clone().detach()
        init_hand_pose = prev_pose[:, 3:].clone().detach()
        init_trans = prev_trans.clone().detach()

        betas = torch.nn.Parameter(prev_betas.clone().detach())
        global_orient = torch.nn.Parameter(init_global_orient)
        hand_pose = torch.nn.Parameter(init_hand_pose)
        trans = torch.nn.Parameter(init_trans)

        # Stage 1: optimize global_orient and trans only
        stage1_optimizer = torch.optim.Adam([global_orient, trans], lr=0.1)
        for _ in range(30):
            mano_output = mano_model(
                global_orient=global_orient, hand_pose=hand_pose.detach(),
                betas=betas.detach(), transl=trans)
            diff = mano_output.joints - target_joints
            loss = (diff ** 2).sum(dim=0).mean()
            stage1_optimizer.zero_grad()
            loss.backward()
            stage1_optimizer.step()

        # Stage 2: optimize all (betas only on frame 0)
        if frame_idx == 0:
            stage2_params = [global_orient, hand_pose, betas, trans]
        else:
            stage2_params = [global_orient, hand_pose, trans]
        stage2_optimizer = torch.optim.Adam(stage2_params, lr=0.05)

        for _ in range(120):
            mano_output = mano_model(
                global_orient=global_orient, hand_pose=hand_pose,
                betas=betas, transl=trans)
            loss = fitting_loss(mano_output, target_joints, betas,
                                initial_hand_pose=init_hand_pose)
            stage2_optimizer.zero_grad()
            loss.backward()
            stage2_optimizer.step()

        prev_pose = torch.cat([global_orient, hand_pose], dim=1).detach()
        prev_trans = trans.detach()
        prev_betas = betas.detach()

        all_poses_list.append(prev_pose.cpu().numpy()[:, None])    # (B, 1, 48)
        all_trans_list.append(prev_trans.cpu().numpy()[:, None])    # (B, 1, 3)
        all_betas_list.append(prev_betas.cpu().numpy()[:, None])   # (B, 1, 10)

    final_poses = np.concatenate(all_poses_list, axis=1)   # (B, 60, 48)
    final_trans = np.concatenate(all_trans_list, axis=1)    # (B, 60, 3)
    final_betas = np.concatenate(all_betas_list, axis=1)   # (B, 60, 10)

    return dict(shape=final_betas, pose=final_poses, trans=final_trans)


def main():
    parser = argparse.ArgumentParser(description="Batch MANO fitting from skeleton data")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory with skeleton .npy files (e.g. H2O/skeleton_split)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for .pkl files (e.g. H2O/mano)')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
    print(f"Found {len(files)} files")

    # Load all skeleton data
    names = []
    skeletons = []
    for f in tqdm(files, desc="Loading"):
        names.append(os.path.splitext(os.path.basename(f))[0])
        skeletons.append(np.load(f))  # (60, 2, 21, 3)
    skeletons = np.stack(skeletons, axis=0)  # (N, 60, 2, 21, 3)

    N = len(names)
    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        batch_names = names[start:end]
        batch_skel = torch.from_numpy(skeletons[start:end]).float().to(device)

        print(f"Processing batch {start}-{end} ({end - start} samples)...")
        left_params = fit_batch(batch_skel[:, :, 0], left_manomodel, device=device)
        right_params = fit_batch(batch_skel[:, :, 1], right_manomodel, device=device)

        for i, name in enumerate(batch_names):
            output = {
                "left_shape": left_params["shape"][i],
                "right_shape": right_params["shape"][i],
                "left_pose": left_params["pose"][i],
                "right_pose": right_params["pose"][i],
                "left_trans": left_params["trans"][i],
                "right_trans": right_params["trans"][i],
            }
            with open(os.path.join(args.output_dir, f"{name}.pkl"), "wb") as f:
                pickle.dump(output, f)

    print(f"Done. Saved {N} files to {args.output_dir}")


if __name__ == "__main__":
    main()
