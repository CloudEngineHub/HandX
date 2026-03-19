import os
import random
import pickle
import codecs as cs

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from os.path import join as pjoin
from tqdm import tqdm

import utils.paramUtil as paramUtil
import clip


##############################################
# Collate utilities
##############################################

def collate_tensors(batch, padding_value=0):
    """
    Generic, fast-ish tensor collator.

    - If 1D tensors: uses pad_sequence.
    - If >=2D tensors: pads to max size along each dim.
    """
    if not isinstance(batch[0], torch.Tensor):
        raise TypeError("collate_tensors expects a list of tensors.")

    # 1D: use pad_sequence (efficient and simple)
    if batch[0].dim() == 1:
        return pad_sequence(batch, batch_first=True, padding_value=padding_value)

    # nD: pad each dimension to max
    dims = batch[0].dim()
    max_sizes = [0] * dims
    for t in batch:
        for d in range(dims):
            if t.size(d) > max_sizes[d]:
                max_sizes[d] = t.size(d)

    out_size = (len(batch),) + tuple(max_sizes)
    out = batch[0].new_full(out_size, padding_value)

    for i, t in enumerate(batch):
        slices = tuple(slice(0, s) for s in t.size())
        out[(i,) + slices] = t

    return out


def make_collate_fn(mot_pad_idx):
    """
    Returns a collate_fn that:
      - Batches captions as list[str]
      - Pads motion token sequences with mot_pad_idx
      - Returns lengths (before padding)
    """

    def collate_fn(batch):
        captions = [b[0] for b in batch]
        m_tokens_list = [b[1] for b in batch]

        lengths = [len(x) for x in m_tokens_list]
        max_len = max(lengths)

        padded = torch.full(
            (len(batch), max_len),
            fill_value=mot_pad_idx,
            dtype=torch.long,
        )

        for i, seq in enumerate(m_tokens_list):
            L = seq.size(0)
            padded[i, :L] = seq

        m_tokens_len = torch.tensor(lengths, dtype=torch.long)

        return captions, padded, m_tokens_len

    return collate_fn


##############################################
# Dataset
##############################################

class Text2MotionDataset_motionmillion(data.Dataset):
    """
    Refactored dataset:
      - NO model calls here
      - NO .to(device) here
      - Only CPU tensors & captions
    """

    def __init__(
        self,
        dataset_name,
        split,
        codebook_size,
        tokenizer_name,
        motion_type=None,
        text_type=None,
        version=None,
        unit_length=4,
        text_encode="clip",
        text_sum_way="cls",
        debug=False,
        meta_dir='./data',
    ):
        self.meta_dir = meta_dir
        self.pointer = 0
        self.dataset_name = dataset_name
        self.motion_type = motion_type
        self.text_type = text_type
        self.version = version

        self.unit_length = unit_length
        self.mot_end_idx = codebook_size          # end token id
        self.mot_pad_idx = codebook_size + 1      # pad token id

        self.tokenizer_name = tokenizer_name
        self.text_encode = text_encode
        self.text_sum_way = text_sum_way

        # ---------------- Dataset-specific config ----------------
        self.data_root = './dataset/HandX'
        self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)
        self.text_dir = pjoin(self.data_root, self.text_type)
        self.joints_num = 22
        self.max_motion_length = 281
        self.max_text_length = 250
        dim_pose = 272
        kinematic_chain = paramUtil.t2m_kinematic_chain
        split_file = os.path.join(self.meta_dir, f'{split}_full_valid.txt')

        # ---------------- Load tokenized data ----------------
        with open(os.path.join(self.data_root, self.tokenizer_name + '.pkl'), "rb") as f:
            all_data = pickle.load(f)

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    id_list.append(line)

        if debug:
            id_list = id_list[:1000]

        data_dict = {}
        new_name_list = []

        # all_data["code_data"][name] : list/array of motion token sequences
        # all_data["text_data"][name] : dict with left/right/interaction_annotation
        for name in tqdm(id_list, desc=f"Loading {dataset_name}-{split}"):
            if name not in all_data["code_data"] or name not in all_data["text_data"]:
                continue

            code_data_ref = all_data["code_data"][name]
            text_data_ref = all_data["text_data"][name]

            if len(code_data_ref) == 0:
                continue

            data_dict[name] = {
                "m_token_list": code_data_ref,
                "text": text_data_ref,
            }
            new_name_list.append(name)

        self.data_dict = data_dict
        self.name_list = new_name_list

        print(f"[Dataset] Loaded {len(self.data_dict)} items.")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        entry = self.data_dict[name]
        m_token_list = entry["m_token_list"]
        text_dict = entry["text"]

        # ---- sample motion tokens ----
        m_tokens = random.choice(m_token_list)      # e.g. np.array or list
        m_tokens = m_tokens.astype(np.int64) #dtype=torch.long)  # keep on CPU
        # m_tokens here are ONLY the motion part, no pads, no end token.

        # ---- build caption from annotations ----
        # use last two annotations (matching original behavior)
        l_text_list = text_dict['left_annotation'][:]
        r_text_list = text_dict['right_annotation'][:]
        i_text_list = text_dict['interaction_annotation'][:]

        # fallback: if lists are empty for some reason
        # if len(l_text_list) == 0:
        #     l_text_list = [""]
        # if len(r_text_list) == 0:
        #     r_text_list = [""]
        # if len(i_text_list) == 0:
        #     i_text_list = [""]

        ltext = random.choice(l_text_list)
        rtext = random.choice(r_text_list)
        itext = random.choice(i_text_list)

        caption = f"<extra_id_0> {ltext} <extra_id_1> {rtext} <extra_id_2> {itext}"

        return caption, m_tokens, np.array(m_tokens.shape[0])


##############################################
# DataLoader factory
##############################################

def DATALoader(
    dataset_name,
    batch_size,
    codebook_size,
    tokenizer_name,
    split,
    text_encode,
    text_sum_way,
    motion_type=None,
    text_type=None,
    version=None,
    unit_length=4,
    num_workers=4,
    debug=False,
):
    dataset = Text2MotionDataset_motionmillion(
        dataset_name=dataset_name,
        split=split,
        codebook_size=codebook_size,
        tokenizer_name=tokenizer_name,
        motion_type=motion_type,
        text_type=text_type,
        version=version,
        unit_length=unit_length,
        text_encode=text_encode,
        text_sum_way=text_sum_way,
        debug=debug,
    )

    # collate_fn = make_collate_fn(dataset.mot_pad_idx)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # collate_fn=collate_fn,
        # drop_last=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    # print('LOADING')
    return loader


##############################################
# How to process batches after loading
##############################################

def encode_text_batch(
    captions,
    text_encode,
    text_sum_way,
    clip_model=None,
    hf_tokenizer=None,
    hf_model=None,
    device="cuda",
    max_text_length=250,
):
    """
    Returns:
      feat_clip_text: (B, T_text, D) in bfloat16
      y_mask:        (B, T_text)    int/bool
      text_tokens_len: (B,)         long
    """

    if text_encode == "clip":
        # CLIP: outputs a single pooled embedding per text
        with torch.no_grad():
            text_tokens = clip.tokenize(captions, truncate=True).to(device, non_blocking=True)
            feats = clip_model.encode_text(text_tokens).to(torch.bfloat16)  # (B, D)

        # treat as a single "token"
        feat_clip_text = feats.unsqueeze(1)                        # (B, 1, D)
        y_mask = torch.ones((feats.size(0), 1), device=device, dtype=torch.long)
        text_tokens_len = torch.ones((feats.size(0),), device=device, dtype=torch.long)

        return feat_clip_text, y_mask, text_tokens_len

    elif text_encode in ["flan-t5-xl", "flan-t5-xxl"]:
        # HuggingFace encoder-decoder / encoder model
        assert hf_tokenizer is not None and hf_model is not None, \
            "hf_tokenizer and hf_model must be provided for T5."

        with torch.no_grad():
            enc = hf_tokenizer(
                list(captions),
                padding=True,
                truncation=True,
                max_length=max_text_length,
                return_tensors="pt",
            )
            input_ids = enc.input_ids.to(device, non_blocking=True)
            attn_mask = enc.attention_mask.to(device, non_blocking=True)

            outputs = hf_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=False,
            )
            hidden = outputs.last_hidden_state.to(torch.bfloat16)  # (B, T, D)

        y_mask = attn_mask  # (B, T)
        text_tokens_len = y_mask.sum(dim=1)  # (B,)

        # reduce according to text_sum_way; keep sequence dim = 1 for compatibility
        if text_sum_way == "cls":
            # take first token
            pooled = hidden[:, 0, :]                      # (B, D)
            feat_clip_text = pooled.unsqueeze(1)          # (B, 1, D)
            text_tokens_len = torch.ones_like(text_tokens_len)  # treat as 1

        elif text_sum_way == "mean":
            # mask-aware mean
            lengths = text_tokens_len.clamp(min=1).unsqueeze(-1)  # (B, 1)
            pooled = (hidden * y_mask.unsqueeze(-1)).sum(dim=1) / lengths
            feat_clip_text = pooled.unsqueeze(1)          # (B, 1, D)
            text_tokens_len = torch.ones_like(text_tokens_len)

        elif text_sum_way == "sum":
            pooled = (hidden * y_mask.unsqueeze(-1)).sum(dim=1)   # (B, D)
            feat_clip_text = pooled.unsqueeze(1)                  # (B, 1, D)
            text_tokens_len = torch.ones_like(text_tokens_len)

        else:
            # keep full sequence
            feat_clip_text = hidden

        return feat_clip_text, y_mask, text_tokens_len

    else:
        raise ValueError(f"Unknown text encoder: {text_encode}")


def build_motion_tokens_with_text_offset(
    raw_m_tokens,
    raw_m_tokens_len,
    text_tokens_len,
    max_motion_length,
    mot_pad_idx,
    mot_end_idx,
    device,
):
    """
    Replicates the original logic:
      final_seq = [PAD x text_len] + m_tokens + [END] + [PAD ...]  (length = max_motion_length)

    Arguments:
      raw_m_tokens:    (B, L_raw_max) long, padded with mot_pad_idx
      raw_m_tokens_len:(B,) lengths before that padding
      text_tokens_len: (B,) number of text tokens (or 1 if pooled)
    """
    B = raw_m_tokens.size(0)
    final_tokens = torch.full(
        (B, max_motion_length),
        fill_value=mot_pad_idx,
        dtype=torch.long,
        device=device,
    )

    for i in range(B):
        tlen = int(text_tokens_len[i].item())
        mlen = int(raw_m_tokens_len[i].item())

        # where motion tokens can start
        start = tlen
        # leave space for END token
        max_motion_space = max_motion_length - start - 1
        if max_motion_space <= 0:
            # no room; just put END at last position
            final_tokens[i, -1] = mot_end_idx
            continue

        use_len = min(mlen, max_motion_space)
        if use_len > 0:
            final_tokens[i, start:start + use_len] = raw_m_tokens[i, :use_len].to(device)

        end_pos = start + use_len
        if end_pos < max_motion_length:
            final_tokens[i, end_pos] = mot_end_idx

    return final_tokens


##############################################
# Example usage in training loop
##############################################

def example_training_loop_step(
    batch,
    dataset,
    text_encode,
    text_sum_way,
    clip_model=None,
    hf_tokenizer=None,
    hf_model=None,
    device="cuda",
):
    """
    batch from DataLoader:
      captions:       list[str]
      raw_m_tokens:   (B, L_raw_max) long on CPU
      raw_m_tokens_len:(B,) long on CPU
    """

    captions, raw_m_tokens, raw_m_tokens_len = batch

    # 1) Move motion tokens to device (batched)
    raw_m_tokens = raw_m_tokens.to(device, non_blocking=True)
    raw_m_tokens_len = raw_m_tokens_len.to(device, non_blocking=True)

    # 2) Encode captions as a batch
    feat_clip_text, y_mask, text_tokens_len = encode_text_batch(
        captions=captions,
        text_encode=text_encode,
        text_sum_way=text_sum_way,
        clip_model=clip_model,
        hf_tokenizer=hf_tokenizer,
        hf_model=hf_model,
        device=device,
        max_text_length=dataset.max_text_length,
    )

    # 3) Build final motion token sequences that depend on text length
    final_m_tokens = build_motion_tokens_with_text_offset(
        raw_m_tokens=raw_m_tokens,
        raw_m_tokens_len=raw_m_tokens_len,
        text_tokens_len=text_tokens_len,
        max_motion_length=dataset.max_motion_length,
        mot_pad_idx=dataset.mot_pad_idx,
        mot_end_idx=dataset.mot_end_idx,
        device=device,
    )

    # Now you have:
    #   final_m_tokens:  (B, max_motion_length)
    #   feat_clip_text:  (B, T_text, D)
    #   y_mask:          (B, T_text) or None (for CLIP pooled)
    #   text_tokens_len: (B,)
    #
    # Feed these to your model:
    # loss = model(final_m_tokens, feat_clip_text, y_mask, ...)
    # loss.backward()
    # optimizer.step()

    return final_m_tokens, feat_clip_text, y_mask, text_tokens_len
