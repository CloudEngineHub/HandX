import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

JOINT_NAME_INDEX_MAP = {
    'wrist': 0,
    'thumb_cmc': 13, 'thumb_mcp': 14, 'thumb_ip': 15, 'thumb_tip': 16,
    'index_mcp': 1, 'index_pip': 2, 'index_dip': 3, 'index_tip': 17,
    'middle_mcp': 4, 'middle_pip': 5, 'middle_dip': 6, 'middle_tip': 18,
    'ring_mcp': 10, 'ring_pip': 11, 'ring_dip': 12, 'ring_tip': 19,
    'pinky_mcp': 7, 'pinky_pip': 8, 'pinky_dip': 9, 'pinky_tip': 20,
}

H2O_TO_MY_ORDER_MAP = [
    0,  # 0: wrist       -> H2O[0]
    5,  # 1: index_mcp    -> H2O[5]
    6,  # 2: index_pip    -> H2O[6]
    7,  # 3: index_dip    -> H2O[7]
    9,  # 4: middle_mcp   -> H2O[9]
    10, # 5: middle_pip   -> H2O[10]
    11, # 6: middle_dip   -> H2O[11]
    17, # 7: pinky_mcp    -> H2O[17]
    18, # 8: pinky_pip    -> H2O[18]
    19, # 9: pinky_dip    -> H2O[19]
    13, # 10: ring_mcp    -> H2O[13]
    14, # 11: ring_pip    -> H2O[14]
    15, # 12: ring_dip    -> H2O[15]
    1,  # 13: thumb_cmc   -> H2O[1]
    2,  # 14: thumb_mcp   -> H2O[2]
    3,  # 15: thumb_ip    -> H2O[3]
    4,  # 16: thumb_tip   -> H2O[4]
    8,  # 17: index_tip   -> H2O[8]
    12, # 18: middle_tip  -> H2O[12]
    16, # 19: ring_tip    -> H2O[16]
    20  # 20: pinky_tip   -> H2O[20]
]


def parse_hand_pose_line(line):
    """
    Parse a line of hand_pose data
    Format: confidence x1 y1 z1 x2 y2 z2 ... (for 21 joints of left and right hands)
    Returns: left_joints (21, 3), right_joints (21, 3), left_conf, right_conf
    """
    values = np.array(line.strip().split(), dtype=np.float32)
    left_conf = values[0]
    left_coords = values[1:64].reshape(21, 3)
    right_conf = values[64]
    right_coords = values[65:128].reshape(21, 3)
    return left_coords, right_coords, left_conf, right_conf

def extract_sequence_data(sequence_path, output_skeleton_path):
    """
    Extract data from a sequence
    """
    hand_pose_dir = os.path.join(sequence_path, 'hand_pose')
    pose_files = sorted([f for f in os.listdir(hand_pose_dir) if f.endswith('.txt')])

    num_frames = len(pose_files)
    skeleton_data = np.zeros((num_frames, 2, 21, 3), dtype=np.float32)

    for i, pose_file in enumerate(pose_files):
        with open(os.path.join(hand_pose_dir, pose_file), 'r') as f:
            line = f.readline()
            left_joints, right_joints, _, _ = parse_hand_pose_line(line)
            skeleton_data[i, 0] = left_joints
            skeleton_data[i, 1] = right_joints

   
    skeleton_data = skeleton_data[:, :, H2O_TO_MY_ORDER_MAP, :]

    np.save(output_skeleton_path, skeleton_data)

    return num_frames


def build_subject_jobs(h2o_base, subjects):
    jobs_by_subject = {}
    global_index = 0

    for subject in subjects:
        subject_path = os.path.join(h2o_base, subject)
        categories = sorted([d for d in os.listdir(subject_path)
                             if os.path.isdir(os.path.join(subject_path, d))])
        subject_jobs = []
        for category in categories:
            category_path = os.path.join(subject_path, category)
            sequences = sorted([d for d in os.listdir(category_path)
                                if os.path.isdir(os.path.join(category_path, d))],
                               key=lambda x: int(x))
            for seq_id in sequences:
                seq_path = os.path.join(category_path, seq_id, 'cam4')
                subject_jobs.append({
                    'subject': subject,
                    'category': category,
                    'seq_id': seq_id,
                    'seq_path': seq_path,
                    'output_index': global_index,
                })
                global_index += 1
        jobs_by_subject[subject] = subject_jobs
    return jobs_by_subject


def process_subject_jobs(subject, subject_jobs, output_skeleton_dir):
    success_count = 0
    error_count = 0

    for job in tqdm(subject_jobs, desc=f"Processing {subject}"):
        seq_path = job['seq_path']
        output_index = job['output_index']

        if not os.path.exists(seq_path):
            print(f"Warning: {seq_path} does not exist")
            error_count += 1
            continue

        output_skeleton_file = os.path.join(output_skeleton_dir, f'{output_index:04d}.npy')
        try:
            num_frames = extract_sequence_data(seq_path, output_skeleton_file)
            print(
                f"Extracted {job['subject']}/{job['category']}/{job['seq_id']} "
                f"-> {output_index:04d} ({num_frames} frames)"
            )
            success_count += 1
        except Exception as e:
            print(f"Error processing {job['subject']}/{job['category']}/{job['seq_id']}: {e}")
            error_count += 1

    return success_count, error_count


def main():
    h2o_base = './raw/'
    output_skeleton_dir = './skeleton/'

    os.makedirs(output_skeleton_dir, exist_ok=True)

    subjects = sorted([d for d in os.listdir(h2o_base)
                       if os.path.isdir(os.path.join(h2o_base, d)) and d.startswith('subject')])
    jobs_by_subject = build_subject_jobs(h2o_base, subjects)

    sequence_count = 0
    error_count = 0
    max_workers = min(4, len(subjects))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_subject_jobs, subject, jobs_by_subject[subject], output_skeleton_dir)
            for subject in subjects
        ]
        for future in as_completed(futures):
            succ, err = future.result()
            sequence_count += succ
            error_count += err

    print(f"\nTotal sequences extracted: {sequence_count}")
    print(f"Total sequences failed/skipped: {error_count}")

if __name__ == '__main__':
    main()
