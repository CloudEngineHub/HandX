#!/usr/bin/env python3
"""
Standalone Evaluation Script

This script runs evaluation metrics on pre-generated motion samples.
Use this when you already have generated PKL files and want to compute metrics only.

Usage:
    python run_evaluation.py --output_dir /path/to/pkl/files

"""

import sys
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add script directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from tma.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from tma.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
from eval_t2m_utils import (
    calculate_activation_statistics,
    calculate_frechet_distance,
    calculate_R_precision,
    euclidean_distance_matrix,
    calculate_diversity,
    calculate_multimodality,
    calculate_mpjpe
)

from interaction import (
    compute_intra_metric,
    compute_inter_metric,
    compute_metric
)


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # Encoder parameters
    'encoder': {
        'checkpoint_path': str(SCRIPT_DIR / 'checkpoints/epoch=269.ckpt'),
        'mean_path': str(SCRIPT_DIR / 'checkpoints/mean_can_pos.npy'),
        'std_path': str(SCRIPT_DIR / 'checkpoints/std_can_pos.npy'),
        'text_model': 'distilbert-base-uncased',
        'latent_dim': 256,
        'ff_size': 1024,
        'num_layers': 4,
    },

    # Evaluation parameters
    'evaluation': {
        'batch_size': 64,
        'seed': 42,
    },

    # Interaction metrics parameters
    'interaction': {
        'intra_threshold': 0.020,  # 2cm for finger-finger contact within same hand
        'inter_threshold': 0.025,  # 2.5cm for finger-palm contact across hands
        'min_duration': 1,         # minimum frames for valid contact
    },

    # Output
    'output': {
        'results_file': 'evaluation_results.json',
        'embeddings_file': 'embeddings.npz',
    }
}


# ============================================================================
# Helper function for batch encoding
# ============================================================================

def encode_texts_in_batches(textencoder, texts, batch_size=32):
    """Encode texts in batches to avoid OOM errors."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = textencoder(batch_texts).loc
        all_embeddings.append(batch_embeddings)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(all_embeddings, dim=0)


# ============================================================================
# Load encoder models
# ============================================================================

def load_encoders(config, device):
    """Load text and motion encoders"""
    print("\n" + "="*80)
    print("Loading encoder models")
    print("="*80)

    enc_config = config['encoder']

    # Load checkpoint
    print(f"Loading checkpoint: {enc_config['checkpoint_path']}")
    checkpoint = torch.load(enc_config['checkpoint_path'], weights_only=False, map_location=device)
    state_dict = checkpoint['state_dict']

    # Filter state dict
    text_dict = {k[12:]: v for k, v in state_dict.items() if k.startswith("textencoder")}
    motion_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("motionencoder")}

    # Create models
    print("Creating text encoder...")
    textencoder = DistilbertActorAgnosticEncoder(
        enc_config['text_model'],
        latent_dim=enc_config['latent_dim'],
        ff_size=enc_config['ff_size'],
        num_layers=enc_config['num_layers']
    ).to(device)
    textencoder.load_state_dict(text_dict)
    textencoder.eval()

    print("Creating motion encoder...")
    motionencoder = ActorAgnosticEncoder(
        nfeats=21*3*2,  # 2 hands, 21 joints, 3 coords
        vae=True,
        latent_dim=enc_config['latent_dim'],
        ff_size=enc_config['ff_size'],
        num_layers=enc_config['num_layers']
    ).to(device)
    motionencoder.load_state_dict(motion_dict)
    motionencoder.eval()

    # Load normalization stats
    print(f"Loading normalization stats...")
    mean_enc = torch.from_numpy(np.load(enc_config['mean_path'])).to(device)
    std_enc = torch.from_numpy(np.load(enc_config['std_path'])).to(device)

    print("Encoders loaded successfully\n")

    return textencoder, motionencoder, mean_enc, std_enc


# ============================================================================
# Load generated samples from PKL files
# ============================================================================

def load_generated_samples(output_dir, delete_after_load=False):
    """Load all PKL files from specified directory"""
    print("\n" + "="*80)
    print("Loading generated samples")
    print("="*80)

    output_dir = Path(output_dir)
    print(f"Loading from: {output_dir}")

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Find all PKL files
    pkl_files = list(output_dir.glob("val_sample_*.pkl"))
    print(f"Found {len(pkl_files)} PKL files")

    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No PKL files found in {output_dir}")

    # Sort by validation index
    def extract_val_idx(pkl_path):
        filename = pkl_path.stem
        if '_idx' in filename:
            return int(filename.split('_idx')[1])
        else:
            return int(filename.split('_')[2])

    pkl_files = sorted(pkl_files, key=extract_val_idx)
    print(f"Sorted PKL files by validation index")

    # Load all samples
    samples = []
    for pkl_file in tqdm(pkl_files, desc="Loading PKL files"):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            samples.append(data)

    print(f"Loaded {len(samples)} samples")

    # Optionally delete PKL files after loading
    if delete_after_load:
        print("Cleaning up PKL files after loading...")
        for pkl_file in pkl_files:
            pkl_file.unlink()
        print(f"Deleted {len(pkl_files)} PKL files\n")

    return samples, output_dir


# ============================================================================
# Convert PKL data to encoder input format
# ============================================================================

def prepare_data_for_encoding(samples, device):
    """Convert PKL data to format needed for encoding."""
    print("\n" + "="*80)
    print("Preparing data for encoding")
    print("="*80)

    # Unique GT data (one per text)
    unique_gt_motions = []
    unique_texts = []

    # Grouped prediction data
    num_generated = samples[0]['generated_real'].shape[0]
    pred_motions_groups = [[] for _ in range(num_generated)]
    pred_texts_groups = [[] for _ in range(num_generated)]

    print("Processing samples...")
    for sample in tqdm(samples, desc="Converting data"):
        text_prompt = sample['text_prompt']
        gt_motion = sample['gt_motion_real']  # (60, 168)
        generated = sample['generated_real']  # (num_generated, 60, 168)

        # Convert GT from (60, 168) to (60, 2, 21, 3)
        gt_motion_3d = gt_motion.reshape(60, 2, 21, 4)[:, :, :, :3]

        # Format text
        text_str = f"<extra_id0> {text_prompt['left']} <extra_id1> {text_prompt['right']} <extra_id2> {text_prompt['two_hands_relation']} <extra_id3>"

        unique_gt_motions.append(gt_motion_3d)
        unique_texts.append(text_str)

        for i in range(num_generated):
            gen_motion = generated[i]  # (60, 168)
            gen_motion_3d = gen_motion.reshape(60, 2, 21, 4)[:, :, :, :3]
            pred_motions_groups[i].append(gen_motion_3d)
            pred_texts_groups[i].append(text_str)

    # Convert to tensors
    unique_gt = torch.from_numpy(np.stack(unique_gt_motions)).float().to(device)

    pred_groups = []
    for i in range(num_generated):
        group_tensor = torch.from_numpy(np.stack(pred_motions_groups[i])).float().to(device)
        pred_groups.append(group_tensor)

    print(f"Unique texts: {len(unique_texts)}")
    print(f"Unique GT motions: {unique_gt.shape}")
    print(f"Number of generation groups: {num_generated}")
    print(f"Each group shape: {pred_groups[0].shape}")

    return unique_gt, unique_texts, pred_groups, pred_texts_groups


# ============================================================================
# Encode and evaluate
# ============================================================================

def encode_and_evaluate(unique_gt, unique_texts, pred_groups, pred_texts_groups,
                        textencoder, motionencoder, mean_enc, std_enc, config):
    """Encode motions and texts, then calculate metrics"""
    print("\n" + "="*80)
    print("Encoding and evaluation")
    print("="*80)

    num_groups = len(pred_groups)
    num_unique = len(unique_texts)
    batch_size = config['evaluation']['batch_size']
    interaction_cfg = config['interaction']

    print(f"Number of generation groups: {num_groups}")
    print(f"Unique texts: {num_unique}")
    print(f"Samples per group: {pred_groups[0].shape[0]}")

    # Process GT data
    print("\nProcessing GT data...")
    unique_gt_canon = unique_gt - unique_gt[:, :, 1:2, 0:1]
    unique_gt_for_mpjpe = unique_gt_canon.clone()
    unique_gt_reshaped = unique_gt_canon.reshape(unique_gt_canon.shape[0], -1, 2*21*3)
    unique_gt_normalized = (unique_gt_reshaped - mean_enc) / std_enc
    gt_lengths = [unique_gt_normalized.shape[1] for _ in range(unique_gt_normalized.shape[0])]

    with torch.no_grad():
        print("  Encoding GT motions...")
        em_gt = motionencoder(unique_gt_normalized.float(), gt_lengths).loc

        print("  Encoding GT texts...")
        et_gt = encode_texts_in_batches(textencoder, unique_texts, batch_size=32)

        print(f"  GT motion embeddings: {em_gt.shape}")
        print(f"  GT text embeddings: {et_gt.shape}")

        # GT diversity
        print("\nCalculating GT diversity...")
        gt_mu, gt_cov = calculate_activation_statistics(em_gt.detach().cpu().numpy())

        if num_unique > 1:
            diversity_times_gt = min(num_unique - 1, 300)
            diversity_real = calculate_diversity(em_gt.detach().cpu().numpy(), diversity_times_gt)
        else:
            diversity_real = 0.0

        # Metric accumulators
        fid_list = []
        diversity_list = []
        mpjpe_list = []
        intra_precision_list, intra_recall_list, intra_f1_list = [], [], []
        inter_precision_list, inter_recall_list, inter_f1_list = [], [], []
        R_pred_batch_list = []
        match_pred_batch_list = []
        all_group_embeddings = []
        all_group_texts_for_embeddings = []

        # Process each generation group
        print(f"\nProcessing {num_groups} generation groups...")

        for group_idx in range(num_groups):
            print(f"\n{'='*80}")
            print(f"GROUP {group_idx + 1}/{num_groups}")
            print(f"{'='*80}")

            group_motions = pred_groups[group_idx]
            group_texts = pred_texts_groups[group_idx]

            # Canonicalize and normalize
            print("  Canonicalizing and normalizing...")
            group_canon = group_motions - group_motions[:, :, 1:2, 0:1]
            group_for_mpjpe = group_canon.clone()
            group_reshaped = group_canon.reshape(group_canon.shape[0], -1, 2*21*3)
            group_normalized = (group_reshaped - mean_enc) / std_enc
            group_lengths = [group_normalized.shape[1] for _ in range(group_normalized.shape[0])]

            # Encode
            print("  Encoding...")
            em_group = motionencoder(group_normalized.float(), group_lengths).loc
            et_group = encode_texts_in_batches(textencoder, group_texts, batch_size=32)

            all_group_embeddings.append(em_group.detach().cpu().numpy())
            all_group_texts_for_embeddings.append(et_group.detach().cpu().numpy())

            # FID
            print("  - FID")
            group_mu, group_cov = calculate_activation_statistics(em_group.detach().cpu().numpy())
            group_fid = calculate_frechet_distance(gt_mu, gt_cov, group_mu, group_cov)
            fid_list.append(group_fid)
            print(f"    FID: {group_fid:.4f}")

            # Diversity
            print("  - Diversity")
            if num_unique > 1:
                diversity_times = min(num_unique - 1, 300)
                group_diversity = calculate_diversity(em_group.detach().cpu().numpy(), diversity_times)
                diversity_list.append(group_diversity)
                print(f"    Diversity: {group_diversity:.4f}")
            else:
                diversity_list.append(0.0)

            # MPJPE
            print("  - MPJPE")
            unique_gt_joints = unique_gt_for_mpjpe.reshape(num_unique, 60, 42, 3)
            group_joints = group_for_mpjpe.reshape(num_unique, 60, 42, 3)
            gt_flat = unique_gt_joints.reshape(-1, 42, 3)
            pred_flat = group_joints.reshape(-1, 42, 3)
            mpjpe_per_frame = calculate_mpjpe(gt_flat, pred_flat)
            group_mpjpe = mpjpe_per_frame.mean().item() * 1000
            mpjpe_list.append(group_mpjpe)
            print(f"    MPJPE: {group_mpjpe:.2f} mm")

            # R-precision and Matching Score
            print("  - R-precision and Matching Score")
            num_batches = num_unique // batch_size

            if num_batches == 0:
                batch_R = calculate_R_precision(
                    et_group.detach().cpu().numpy(),
                    em_group.detach().cpu().numpy(),
                    top_k=3, sum_all=True
                ) / num_unique
                batch_match = euclidean_distance_matrix(
                    et_group.detach().cpu().numpy(),
                    em_group.detach().cpu().numpy()
                ).trace() / num_unique
                R_pred_batch_list.append(batch_R)
                match_pred_batch_list.append(batch_match)
            else:
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size
                    batch_R = calculate_R_precision(
                        et_group[start_idx:end_idx].detach().cpu().numpy(),
                        em_group[start_idx:end_idx].detach().cpu().numpy(),
                        top_k=3, sum_all=True
                    ) / batch_size
                    batch_match = euclidean_distance_matrix(
                        et_group[start_idx:end_idx].detach().cpu().numpy(),
                        em_group[start_idx:end_idx].detach().cpu().numpy()
                    ).trace()
                    R_pred_batch_list.append(batch_R)
                    match_pred_batch_list.append(batch_match)

            # Interaction metrics
            print("  - Interaction Metrics")
            intra_tp_total, intra_fp_total, intra_fn_total = 0, 0, 0
            inter_tp_total, inter_fp_total, inter_fn_total = 0, 0, 0

            for i in tqdm(range(num_unique), desc="    Processing samples", leave=False):
                gt_motion_bihand = unique_gt_for_mpjpe[i].cpu().numpy()
                pred_motion_bihand = group_for_mpjpe[i].cpu().numpy()

                intra_tp, intra_fp, intra_fn = compute_intra_metric(
                    gt_motion=gt_motion_bihand,
                    pred_motion=pred_motion_bihand,
                    threshold=interaction_cfg['intra_threshold'],
                    min_duration=interaction_cfg['min_duration']
                )
                inter_tp, inter_fp, inter_fn = compute_inter_metric(
                    gt_motion=gt_motion_bihand,
                    pred_motion=pred_motion_bihand,
                    threshold=interaction_cfg['inter_threshold'],
                    min_duration=interaction_cfg['min_duration']
                )

                intra_tp_total += intra_tp
                intra_fp_total += intra_fp
                intra_fn_total += intra_fn
                inter_tp_total += inter_tp
                inter_fp_total += inter_fp
                inter_fn_total += inter_fn

            intra_p, intra_r, intra_f = compute_metric(intra_tp_total, intra_fp_total, intra_fn_total)
            inter_p, inter_r, inter_f = compute_metric(inter_tp_total, inter_fp_total, inter_fn_total)

            intra_precision_list.append(intra_p)
            intra_recall_list.append(intra_r)
            intra_f1_list.append(intra_f)
            inter_precision_list.append(inter_p)
            inter_recall_list.append(inter_r)
            inter_f1_list.append(inter_f)

            print(f"    Intra: P={intra_p:.4f}, R={intra_r:.4f}, F1={intra_f:.4f}")
            print(f"    Inter: P={inter_p:.4f}, R={inter_r:.4f}, F1={inter_f:.4f}")

        # Aggregate metrics
        print(f"\n{'='*80}")
        print("AGGREGATING METRICS ACROSS GROUPS")
        print(f"{'='*80}")

        # GT R-precision and Matching Score
        print("\nGT R-precision and Matching Score...")
        num_batches_real = num_unique // batch_size

        if num_batches_real == 0:
            temp_R_real = calculate_R_precision(
                et_gt.detach().cpu().numpy(),
                em_gt.detach().cpu().numpy(),
                top_k=3, sum_all=True
            ) / num_unique
            temp_R_real_std = np.zeros(3)
            temp_match_real = euclidean_distance_matrix(
                et_gt.detach().cpu().numpy(),
                em_gt.detach().cpu().numpy()
            ).trace() / num_unique
            temp_match_real_std = 0.0
        else:
            R_real_list = []
            match_real_list = []
            for batch_idx in range(num_batches_real):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_R_real = calculate_R_precision(
                    et_gt[start_idx:end_idx].detach().cpu().numpy(),
                    em_gt[start_idx:end_idx].detach().cpu().numpy(),
                    top_k=3, sum_all=True
                ) / batch_size
                batch_match_real = euclidean_distance_matrix(
                    et_gt[start_idx:end_idx].detach().cpu().numpy(),
                    em_gt[start_idx:end_idx].detach().cpu().numpy()
                ).trace()
                R_real_list.append(batch_R_real)
                match_real_list.append(batch_match_real)

            R_real_array = np.array(R_real_list)
            temp_R_real = np.mean(R_real_array, axis=0)
            temp_R_real_std = np.std(R_real_array, axis=0)
            match_real_array = np.array(match_real_list)
            temp_match_real = np.sum(match_real_array) / num_unique
            temp_match_real_std = np.std(match_real_array / batch_size)

        # Aggregate Pred metrics
        print("\nPred R-precision and Matching Score (across all groups)...")
        R_pred_batch_array = np.array(R_pred_batch_list)
        match_pred_batch_array = np.array(match_pred_batch_list)

        temp_R_pred = np.mean(R_pred_batch_array, axis=0)
        temp_R_pred_std = np.std(R_pred_batch_array, axis=0)
        temp_match_pred = np.sum(match_pred_batch_array) / (num_groups * num_unique)
        temp_match_pred_std = np.std(match_pred_batch_array / batch_size)

        # Multimodality
        print("\nMultimodality...")
        all_em_pred = np.stack(all_group_embeddings, axis=0)
        all_em_pred_reshaped = np.transpose(all_em_pred, (1, 0, 2))

        if num_groups > 1:
            multimodality_times = min(num_groups - 1, 20)
            multimodality = calculate_multimodality(all_em_pred_reshaped, multimodality_times)
            print(f"  Multimodality: {multimodality:.4f}")
        else:
            multimodality = 0.0

        # Aggregate non-batchified metrics
        fid_mean, fid_std = np.mean(fid_list), np.std(fid_list)
        diversity_mean, diversity_std = np.mean(diversity_list), np.std(diversity_list)
        mpjpe_mean, mpjpe_std = np.mean(mpjpe_list), np.std(mpjpe_list)

        intra_precision_mean, intra_precision_std = np.mean(intra_precision_list), np.std(intra_precision_list)
        intra_recall_mean, intra_recall_std = np.mean(intra_recall_list), np.std(intra_recall_list)
        intra_f1_mean, intra_f1_std = np.mean(intra_f1_list), np.std(intra_f1_list)
        inter_precision_mean, inter_precision_std = np.mean(inter_precision_list), np.std(inter_precision_list)
        inter_recall_mean, inter_recall_std = np.mean(inter_recall_list), np.std(inter_recall_list)
        inter_f1_mean, inter_f1_std = np.mean(inter_f1_list), np.std(inter_f1_list)

        # Compile results
        results = {
            'num_unique_texts': num_unique,
            'num_groups': num_groups,
            'num_predictions_per_group': num_unique,
            'total_predictions': num_unique * num_groups,
            'num_batches_gt': num_batches_real if num_batches_real > 0 else 1,
            'num_batches_pred_total': len(R_pred_batch_list),
            'batch_size': batch_size,
            'num_generated_per_text': num_groups,

            'fid': float(fid_mean),
            'fid_std': float(fid_std),
            'diversity_real': float(diversity_real),
            'diversity_pred': float(diversity_mean),
            'diversity_pred_std': float(diversity_std),
            'multimodality': float(multimodality),
            'mpjpe_mm': float(mpjpe_mean),
            'mpjpe_mm_std': float(mpjpe_std),

            'r_precision_real': {
                'top1': float(temp_R_real[0]),
                'top2': float(temp_R_real[1]),
                'top3': float(temp_R_real[2]),
                'top1_std': float(temp_R_real_std[0]),
                'top2_std': float(temp_R_real_std[1]),
                'top3_std': float(temp_R_real_std[2])
            },
            'r_precision_pred': {
                'top1': float(temp_R_pred[0]),
                'top2': float(temp_R_pred[1]),
                'top3': float(temp_R_pred[2]),
                'top1_std': float(temp_R_pred_std[0]),
                'top2_std': float(temp_R_pred_std[1]),
                'top3_std': float(temp_R_pred_std[2])
            },
            'matching_score_real': float(temp_match_real),
            'matching_score_real_std': float(temp_match_real_std),
            'matching_score_pred': float(temp_match_pred),
            'matching_score_pred_std': float(temp_match_pred_std),

            'interaction_intra_hand': {
                'precision': float(intra_precision_mean),
                'precision_std': float(intra_precision_std),
                'recall': float(intra_recall_mean),
                'recall_std': float(intra_recall_std),
                'f1': float(intra_f1_mean),
                'f1_std': float(intra_f1_std)
            },
            'interaction_inter_hand': {
                'precision': float(inter_precision_mean),
                'precision_std': float(inter_precision_std),
                'recall': float(inter_recall_mean),
                'recall_std': float(inter_recall_std),
                'f1': float(inter_f1_mean),
                'f1_std': float(inter_f1_std)
            },
        }

        embeddings = {
            'motion_pred_group0': all_group_embeddings[0],
            'motion_gt': em_gt.cpu().numpy(),
            'text_gt': et_gt.cpu().numpy(),
            'text_pred_group0': all_group_texts_for_embeddings[0],
        }

    return results, embeddings


# ============================================================================
# Save results
# ============================================================================

def save_results(results, embeddings, output_dir, config):
    """Save evaluation results and embeddings"""
    print("\n" + "="*80)
    print("Saving results")
    print("="*80)

    output_cfg = config['output']

    # Save metrics as JSON
    results_path = output_dir / output_cfg['results_file']
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to: {results_path}")

    # Save embeddings as NPZ
    embeddings_path = output_dir / output_cfg['embeddings_file']
    np.savez(embeddings_path, **embeddings)
    print(f"Saved embeddings to: {embeddings_path}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Unique texts (GT): {results['num_unique_texts']}")
    print(f"Number of generation groups: {results['num_groups']}")
    print(f"Total predictions: {results['total_predictions']}")

    print(f"\n{'='*80}")
    print("QUALITY METRICS")
    print("="*80)
    print(f"MPJPE: {results['mpjpe_mm']:.2f} +/- {results['mpjpe_mm_std']:.2f} mm")
    print(f"FID: {results['fid']:.4f} +/- {results['fid_std']:.4f}")

    print(f"\n{'='*80}")
    print("DIVERSITY METRICS")
    print("="*80)
    print(f"Diversity (Real): {results['diversity_real']:.4f}")
    print(f"Diversity (Pred): {results['diversity_pred']:.4f} +/- {results['diversity_pred_std']:.4f}")
    print(f"Multimodality: {results['multimodality']:.4f}")

    print(f"\n{'='*80}")
    print("ALIGNMENT METRICS")
    print("="*80)
    print(f"\nR-precision (Real):")
    print(f"  Top-1: {results['r_precision_real']['top1']:.4f}")
    print(f"  Top-2: {results['r_precision_real']['top2']:.4f}")
    print(f"  Top-3: {results['r_precision_real']['top3']:.4f}")
    print(f"\nR-precision (Pred):")
    print(f"  Top-1: {results['r_precision_pred']['top1']:.4f} +/- {results['r_precision_pred']['top1_std']:.4f}")
    print(f"  Top-2: {results['r_precision_pred']['top2']:.4f} +/- {results['r_precision_pred']['top2_std']:.4f}")
    print(f"  Top-3: {results['r_precision_pred']['top3']:.4f} +/- {results['r_precision_pred']['top3_std']:.4f}")
    print(f"\nMatching Score:")
    print(f"  Real: {results['matching_score_real']:.4f}")
    print(f"  Pred: {results['matching_score_pred']:.4f} +/- {results['matching_score_pred_std']:.4f}")

    print(f"\n{'='*80}")
    print("INTERACTION METRICS")
    print("="*80)
    print(f"\nIntra-hand Contact:")
    print(f"  Precision: {results['interaction_intra_hand']['precision']:.4f}")
    print(f"  Recall:    {results['interaction_intra_hand']['recall']:.4f}")
    print(f"  F1 Score:  {results['interaction_intra_hand']['f1']:.4f}")
    print(f"\nInter-hand Contact:")
    print(f"  Precision: {results['interaction_inter_hand']['precision']:.4f}")
    print(f"  Recall:    {results['interaction_inter_hand']['recall']:.4f}")
    print(f"  F1 Score:  {results['interaction_inter_hand']['f1']:.4f}")
    print("="*80)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Standalone evaluation script for pre-generated motion samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluate_only.py --output_dir /path/to/pkl/files
  python run_evaluate_only.py --output_dir /path/to/pkl/files --batch_size 32
  python run_evaluate_only.py --output_dir /path/to/pkl/files --delete_pkl
        """
    )
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing generated PKL files')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save results (default: current directory)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for R-precision calculation (default: 32)')
    parser.add_argument('--delete_pkl', action='store_true',
                        help='Delete PKL files after loading to save disk space')
    parser.add_argument('--results_file', type=str, default='evaluation_results.json',
                        help='Output JSON file name (default: evaluation_results.json)')
    args = parser.parse_args()

    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    config['evaluation']['batch_size'] = args.batch_size
    config['output']['results_file'] = args.results_file

    # Resolve save_dir to absolute path
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("STANDALONE EVALUATION PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  PKL dir: {args.output_dir}")
    print(f"  Save dir: {save_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Delete PKL after load: {args.delete_pkl}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load encoders
    textencoder, motionencoder, mean_enc, std_enc = load_encoders(config, device)

    # Load generated samples
    samples, output_dir = load_generated_samples(args.output_dir, delete_after_load=args.delete_pkl)

    # Prepare data
    unique_gt, unique_texts, pred_groups, pred_texts_groups = prepare_data_for_encoding(samples, device)

    # Encode and evaluate
    results, embeddings = encode_and_evaluate(
        unique_gt, unique_texts, pred_groups, pred_texts_groups,
        textencoder, motionencoder, mean_enc, std_enc,
        config
    )

    # Save results to save_dir (current directory by default)
    save_results(results, embeddings, save_dir, config)

    print("\n" + "="*80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    main()
