#!/usr/bin/env python3
"""
Bimanual Contact Quality Evaluation

This module computes contact quality metrics for bimanual hand motion sequences.
It evaluates three core metrics:
- Metric 1: Contact Ratio - proportion of frames with hand-hand contact
- Metric 3: Average Contact Duration - mean length of contact segments
- Metric 5: Contact Frequency - number of contact events per second

These metrics are combined into an overall contact quality score.
"""

import numpy as np
from scipy.spatial.distance import cdist

# Contact threshold parameters
FINGER_FINGER_CONTACT = 0.020   # Finger-finger contact threshold (meters)
FINGER_PALM_CONTACT = 0.025     # Finger-palm contact threshold (meters)

# Fingertip joint indices
FINGER_TIP_INDICES = [16, 17, 18, 19, 20]  # thumb, index, middle, ring, pinky

# Palm keypoint indices (all joints except fingertips)
PALM_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def compute_contact_metrics(skeleton_data: np.ndarray, fps: int = 30):
    """
    Compute three core contact-related metrics for bimanual hand motion.

    Analyzes the skeleton sequence to detect contact events between hands
    and computes statistics about contact patterns.

    Args:
        skeleton_data: (T, 2, 21, 3) - Bimanual skeleton data
                       T=frames, 2=hands, 21=joints, 3=xyz coordinates
        fps: Frame rate of the motion sequence

    Returns:
        dict: Contains contact_ratio, avg_contact_duration, contact_frequency,
              num_contact_segments, overall_contact_score, and metadata
    """
    T = skeleton_data.shape[0]
    hand_left = skeleton_data[:, 0, :, :]   # (T, 21, 3)
    hand_right = skeleton_data[:, 1, :, :]  # (T, 21, 3)

    # ========== Pre-compute distances ==========
    # 1. Distance between fingertips
    fingertip_distances = np.zeros(T)
    for t in range(T):
        left_tips = hand_left[t, FINGER_TIP_INDICES, :]
        right_tips = hand_right[t, FINGER_TIP_INDICES, :]
        dist_matrix = cdist(left_tips, right_tips)
        fingertip_distances[t] = np.min(dist_matrix)

    # 2. Finger-palm distance
    finger_palm_distances = np.zeros(T)
    for t in range(T):
        # Left fingertips to right palm
        min_dists_lr = []
        for tip_idx in FINGER_TIP_INDICES:
            tip_pos = hand_left[t, tip_idx, :]
            palm_points = hand_right[t, PALM_INDICES, :]
            dists = np.linalg.norm(palm_points - tip_pos, axis=1)
            min_dists_lr.append(np.min(dists))

        # Right fingertips to left palm
        min_dists_rl = []
        for tip_idx in FINGER_TIP_INDICES:
            tip_pos = hand_right[t, tip_idx, :]
            palm_points = hand_left[t, PALM_INDICES, :]
            dists = np.linalg.norm(palm_points - tip_pos, axis=1)
            min_dists_rl.append(np.min(dists))

        finger_palm_distances[t] = min(min(min_dists_lr), min(min_dists_rl))

    # ========== Metric 1: Contact Ratio ==========
    finger_finger_contact = fingertip_distances < FINGER_FINGER_CONTACT
    finger_palm_contact = finger_palm_distances < FINGER_PALM_CONTACT

    interaction_mask = finger_finger_contact | finger_palm_contact
    contact_ratio = np.sum(interaction_mask) / T

    # ========== Metric 3: Avg Contact Duration (seconds) ==========
    # Find continuous contact segments
    segments = []
    in_contact = False
    start = 0

    for t in range(T):
        if interaction_mask[t] and not in_contact:
            start = t
            in_contact = True
        elif not interaction_mask[t] and in_contact:
            segments.append((start, t-1))
            in_contact = False

    if in_contact:
        segments.append((start, T-1))

    durations = [end - start + 1 for start, end in segments]
    avg_duration = np.mean(durations) / fps if durations else 0.0
    num_contacts = len(segments)

    # ========== Metric 5: Contact Frequency (events per second) ==========
    contact_frequency = num_contacts / (T / fps) if T > 0 else 0.0

    # ========== Overall Contact Score ==========
    # Combined score: normalize and weight-combine the three metrics
    # Using weighted average instead of geometric mean to handle zero values

    # Normalization approach:
    # Contact_Ratio: already in [0, 1] range
    # Avg_Duration: normalize assuming max value of 5 seconds
    # Contact_Frequency: normalize assuming max value of 5 events/second

    norm_ratio = contact_ratio  # [0, 1]
    norm_duration = min(avg_duration / 5.0, 1.0)  # [0, 1]
    norm_frequency = min(contact_frequency / 5.0, 1.0)  # [0, 1]

    # Weighted average: contact ratio 40%, duration 30%, frequency 30%
    overall_score = 0.4 * norm_ratio + 0.3 * norm_duration + 0.3 * norm_frequency

    return {
        'contact_ratio': float(contact_ratio),
        'avg_contact_duration': float(avg_duration),
        'contact_frequency': float(contact_frequency),
        'num_contact_segments': int(num_contacts),
        'overall_contact_score': float(overall_score),
        'metadata': {
            'frames': T,
            'duration_sec': T / fps,
            'fps': fps
        }
    }


def evaluate_npy_file(npy_path: str, fps: int = 30, verbose: bool = True):
    """
    Evaluate a single npy file containing bimanual skeleton data.

    Loads the skeleton data from file, validates the format,
    and computes all contact metrics.

    Args:
        npy_path: Path to the npy file
        fps: Frame rate of the motion
        verbose: Whether to print detailed information

    Returns:
        dict: Evaluation results containing all contact metrics
    """
    # Load data
    skeleton_data = np.load(npy_path)

    if verbose:
        print(f"File: {npy_path}")
        print(f"Shape: {skeleton_data.shape}")

    # Validate data format
    if len(skeleton_data.shape) != 4 or skeleton_data.shape[1] != 2 or skeleton_data.shape[2] != 21 or skeleton_data.shape[3] != 3:
        raise ValueError(f"Data format error! Expected (T, 2, 21, 3), got {skeleton_data.shape}")

    # Compute metrics
    result = compute_contact_metrics(skeleton_data, fps=fps)

    if verbose:
        print("\n" + "="*60)
        print("Contact Quality Evaluation Results")
        print("="*60)
        print(f"Metric 1 - Contact Ratio:        {result['contact_ratio']:.4f}")
        print(f"Metric 3 - Avg Contact Duration: {result['avg_contact_duration']:.4f} sec")
        print(f"Metric 5 - Contact Frequency:    {result['contact_frequency']:.4f} events/sec")
        print(f"           Contact Segments:      {result['num_contact_segments']} events")
        print("-"*60)
        print(f"Overall Contact Score:            {result['overall_contact_score']:.4f}")
        print("="*60)

    return result


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python compute_contact_metric.py <npy_file_path> [fps]")
        print("\nExamples:")
        print("  python compute_contact_metric.py /path/to/motion.npy")
        print("  python compute_contact_metric.py /path/to/motion.npy 30")
        sys.exit(1)

    npy_path = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    try:
        result = evaluate_npy_file(npy_path, fps=fps, verbose=True)

        # Optional: Save as JSON
        # output_json = npy_path.replace('.npy', '_contact_metrics.json')
        # with open(output_json, 'w') as f:
        #     json.dump(result, f, indent=2)
        # print(f"\nResults saved to: {output_json}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
