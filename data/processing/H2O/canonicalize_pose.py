import numpy as np
from pathlib import Path
import tqdm
from collections import Counter


JOINT_NAME_INDEX_MAP = {
    'wrist': 0,
    'thumb_cmc': 13, 'thumb_mcp': 14, 'thumb_ip': 15, 'thumb_tip': 16,
    'index_mcp': 1, 'index_pip': 2, 'index_dip': 3, 'index_tip': 17,
    'middle_mcp': 4, 'middle_pip': 5, 'middle_dip': 6, 'middle_tip': 18,
    'ring_mcp': 10, 'ring_pip': 11, 'ring_dip': 12, 'ring_tip': 19,
    'pinky_mcp': 7, 'pinky_pip': 8, 'pinky_dip': 9, 'pinky_tip': 20,
}


def determine_flip_axes(skeleton):
    """
    Analyze and determine which coordinate axes need to be flipped based on the given skeleton data.
    """
    left_hand_x_mean = skeleton[0, 0, :, 0].mean()
    right_hand_x_mean = skeleton[0, 1, :, 0].mean()
    flip_x = left_hand_x_mean > right_hand_x_mean

    wrist_idx = JOINT_NAME_INDEX_MAP['wrist']
    mcp_idx = JOINT_NAME_INDEX_MAP['middle_mcp']

    wrist_pos_y = skeleton[0, :, wrist_idx, 1]
    mcp_pos_y = skeleton[0, :, mcp_idx, 1]
    finger_direction_y = (mcp_pos_y - wrist_pos_y).mean()

    flip_y = finger_direction_y < 0

    flip_z = flip_x ^ flip_y
    return flip_x, flip_y, flip_z


def apply_flip(points, flip_x, flip_y, flip_z):
    """
    Apply specified axis flipping to a coordinate array with shape (..., 3).
    """
    flipped = points.copy()
    if flip_x:
        flipped[..., 0] = -flipped[..., 0]
    if flip_y:
        flipped[..., 1] = -flipped[..., 1]
    if flip_z:
        flipped[..., 2] = -flipped[..., 2]
    return flipped


def process_data_flexible(input_dir, output_dir):
    """
    Independently analyze and process each skeleton file.
    Flip axes to canonical orientation.
    Note: per-window centering (subtract right wrist at window frame 0)
    is done later when slicing windows, not here.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    skeleton_input = input_dir


    skeleton_output = output_dir
    skeleton_output.mkdir(parents=True, exist_ok=True)

    skeleton_files = sorted(list(skeleton_input.glob('*.npy')))

    transform_counter = Counter()

    print(f"Will independently analyze and process {len(skeleton_files)} data samples based on Skeleton files...")

    for skeleton_file in tqdm.tqdm(skeleton_files, desc="Processing files"):
        try:
            skeleton_data = np.load(skeleton_file)
            flip_x, flip_y, flip_z = determine_flip_axes(skeleton_data)
            transform_counter[(flip_x, flip_y, flip_z)] += 1
            flipped_skeleton = apply_flip(skeleton_data, flip_x, flip_y, flip_z)
            print(f"\nProcessing file: {skeleton_file.name} | Flip axes - X: {flip_x}, Y: {flip_y}, Z: {flip_z}")

            skel_output_path = skeleton_output / skeleton_file.name
            np.save(skel_output_path, flipped_skeleton)

        except Exception as e:
            print(f"\nError occurred while processing file {skeleton_file.name}, skipped. Error message: {e}")



if __name__ == "__main__":
    INPUT_DATA_DIR = "./skeleton"
    OUTPUT_DATA_DIR = "./skeleton_canonicalized"

    print(f"Input directory: {INPUT_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DATA_DIR}")

    process_data_flexible(input_dir=INPUT_DATA_DIR, output_dir=OUTPUT_DATA_DIR)
