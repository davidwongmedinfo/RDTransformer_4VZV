import os
import sys
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from torch.utils.data import Dataset
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
import torch


class SequenceDataset(Dataset):
    """
    A Dataset that reads sequences from a FASTA file and converts them to k-mer indices.

    Assumptions:
    - FASTA description line format: ">sequence_id|wb_label|elisa_label"
    - When used with is_train=False, a pre-built kmer_to_idx dict must be supplied so vocabulary is shared with pretraining or fulltraining.
    """
    def __init__(self, fasta_file, kmer_size=3, max_length=2048, is_train=True, kmer_to_idx=None, label_encoder=None, compute_class_weights=False, reference_sequence=None, label_type=None):
        self.kmer_size = kmer_size
        self.max_length = max_length
        self.sequences = []
        self.labels = []
        self.ids = []  
        self.compute_class_weights = compute_class_weights
        self.class_weights = None
        self.gap_token = '-'
        
        # Read FASTA and extract sequence, label and id
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq)
            # Description format: "sequence_id|wb_label|elisa_label"
            parts = record.description.split('|')
            seq_id = parts[0].strip()
            lt = label_type if label_type is not None else 'wb'
            if lt == 'wb':
                if len(parts) < 2:
                    raise ValueError(f"FASTA description missing wb label: {record.description}")
                label = parts[1].strip()
            elif lt == 'elisa':
                # Guard missing third field
                if len(parts) < 3:
                    raise ValueError(f"FASTA description missing elisa label: {record.description}")
                label = parts[2].strip()
            else:
                raise ValueError(f"Invalid label type: '{lt}'. Must be 'wb' or 'elisa'")

            self.sequences.append(seq)
            self.labels.append(label)
            self.ids.append(seq_id)

        # Fit or use provided LabelEncoder
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.labels)
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(self.labels)

        # Compute class weights for training set if requested (optional)
        if compute_class_weights:  
            self.class_weights = self._compute_class_weights()
        
        # Build k-mer vocabulary if training; otherwise use provided mapping
        if is_train:
            self.build_vocab()
        else:
            if kmer_to_idx is None:
                raise ValueError("kmer_to_idx must be provided for validation/test sets")
            self.kmer_to_idx = kmer_to_idx
            self.vocab_size = len(kmer_to_idx) + 1  # +1 for padding index 0
        
        # Precompute k-mer index sequences
        self.kmer_indices = [self.sequence_to_kmers(seq) for seq in self.sequences]

        # Prepare reference sequence indices (optional)
        if reference_sequence:
            self.reference_indices = self.sequence_to_kmers(reference_sequence)
    
    def get_reference_indices(self):
        """Return reference sequence k-mer index list if provided, else None."""
        return getattr(self, "reference_indices", None)

    def _compute_class_weights(self):
        """Compute balanced class weights and return dict mapping class_index -> weight."""
        unique_classes = np.unique(self.labels)
        weights = compute_class_weight('balanced', classes=unique_classes, y=self.labels)
        class_weights = {cls: weight for cls, weight in zip(unique_classes, weights)}
        return class_weights
    
    def build_vocab(self):
        """Build k-mer vocabulary from training sequences. Reserve index 0 for padding."""
        all_kmers = []
        for seq in self.sequences:
            kmers = self.extract_kmers(seq)
            all_kmers.extend(kmers)
        
        kmer_counter = collections.Counter(all_kmers)
        sorted_kmers = sorted(kmer_counter.items(), key=lambda x: x[1], reverse=True)
        # Indices start at 1; 0 is reserved for padding
        self.kmer_to_idx = {kmer: idx+1 for idx, (kmer, _) in enumerate(sorted_kmers)}  # index 0 reserved for padding
        self.vocab_size = len(self.kmer_to_idx) + 1  # +1 for padding index 0
    
    def extract_kmers(self, sequence):
        """Return overlapping k-mers from a raw sequence string."""
        kmers = []
        for i in range(len(sequence) - self.kmer_size + 1):
            kmer = sequence[i:i+self.kmer_size]
            kmers.append(kmer)
        return kmers
    
    def sequence_to_kmers(self, sequence):
        """Convert sequence to fixed-length k-mer index list (padding or truncation)."""
        kmers = self.extract_kmers(sequence)
        indices = []
        for kmer in kmers:
            # Treat any k-mer containing gap_token as padding (skip)
            if self.gap_token is not None and self.gap_token in kmer:
                indices.append(0)   
            else:
                indices.append(self.kmer_to_idx.get(kmer, 0))
        # Padding or truncation to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices += [0] * (self.max_length - len(indices))
        return indices
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Return (kmer_index_tensor, label_tensor, id_str)."""
        return torch.tensor(self.kmer_indices[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long), self.ids[idx]

def validate_and_normalize_config(config, yaml_cfg=None):
    """
    Cast common config fields to the expected Python types (int/float/bool).
    Call this after applying YAML values and CLI overrides, before using config.
    """
    # cast ints
    int_fields = ['batch_size', 'num_epochs', 'num_workers', 'n_boot', 'seed',
                  'kmer_size', 'max_seq_length', 'num_layers', 'pretrain_num_classes']
    for k in int_fields:
        if hasattr(config, k):
            v = getattr(config, k)
            if v is not None:
                try:
                    setattr(config, k, int(v))
                except Exception:
                    raise ValueError(f"Config.{k} must be int-like (got {v!r})")

    # cast floats
    float_fields = ['learning_rate', 'weight_decay', 'lr_min', 'start_factor', 'end_factor', 'max_grad_norm', 'dropout_rate']
    for k in float_fields:
        if hasattr(config, k):
            v = getattr(config, k)
            if v is not None:
                try:
                    setattr(config, k, float(v))
                except Exception:
                    raise ValueError(f"Config.{k} must be float-like (got {v!r})")

    # bools from strings (e.g. "False" -> False)
    bool_fields = ['use_class_weights']
    for k in bool_fields:
        if hasattr(config, k):
            v = getattr(config, k)
            if isinstance(v, str):
                lower = v.strip().lower()
                if lower in ('true', '1', 'yes'):
                    setattr(config, k, True)
                elif lower in ('false', '0', 'no', ''):
                    setattr(config, k, False)
                else:
                    raise ValueError(f"Config.{k} must be boolean-like (got {v!r})")

    return config

def save_config(config_obj, output_folder, save_name):
    """
    Save config object to JSON. Works if config is a dataclass or a plain object.
    Uses default=str in json.dump to guard against non-serializable types.
    """
    config_path = os.path.join(output_folder, save_name)
    try:
        if is_dataclass(config_obj):
            config_dict = asdict(config_obj)
        else:
            config_dict = {k: v for k, v in vars(config_obj).items() if not k.startswith("_") and not callable(v)}
    except Exception:
        config_dict = {k: str(v) for k, v in vars(config_obj).items() if not k.startswith("_")}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4, default=str)
    print(f"Training configuration saved to: {config_path}")

def safe_index_or_max(arr, idx):
    """
    Safely return arr[idx] if available and not NaN, otherwise return max(arr).
    This helper avoids index or empty-array errors when extracting 'best epoch' metrics.
    """
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return np.nan
    if idx is not None and 0 <= idx < arr.size and not np.isnan(arr[int(idx)]):
        return float(arr[int(idx)])
    try:
        return float(np.nanmax(arr))
    except ValueError:
        return np.nan

def expected_calibration_error(true_binary, prob_pos, n_bins=8, strategy='quantile'):
    """
    Compute ECE (expected calibration error).
    Supports two strategies for bin edges:
    - 'uniform' : equal-width bins on [0,1]
    - 'quantile' : quantile-based bins so each bin has ~equal number of samples
    """
    true_binary = np.asarray(true_binary)
    prob_pos = np.asarray(prob_pos)
    # ​​Return NaN for empty predictions (safeguard)​
    if prob_pos.size == 0:
        return float(np.nan)

    # Choose bin edges
    if strategy == 'quantile':
        try:
            bins = np.quantile(prob_pos, np.linspace(0.0, 1.0, n_bins + 1))
            # If quantile edges collapsed (e.g. constant probabilities), fallback to uniform bins
            if np.any(np.diff(bins) <= 0):
                bins = np.linspace(0.0, 1.0, n_bins + 1)
        except Exception:
            bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        # default: uniform / equal-width bins
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    # Map probabilities to bin ids (0..n_bins-1)
    binids = np.digitize(prob_pos, bins) - 1
    # Clip to valid range in case of edge values
    binids = np.clip(binids, 0, n_bins - 1)

    ece = 0.0
    total = len(prob_pos)
    for b in range(n_bins):
        mask = binids == b
        if mask.sum() > 0:
            acc = true_binary[mask].mean()
            conf = prob_pos[mask].mean()
            ece += (mask.sum() / total) * abs(acc - conf)
    return float(ece)

def bootstrap_ci(y_true, y_score, metric=None, n_bootstrap=None, seed=None):
    """
    Compute bootstrap 95% CI for a given metric on (y_true, y_score).
    Returns: (point_estimate, lower_95, upper_95, n_valid_boots)
    metric is one of 'auroc', 'auprc', 'brier'.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    # Return NaNs if no samples
    if n == 0:
        return np.nan, np.nan, np.nan, 0

    # point estimate  
    try:
        if metric == 'auroc':
            pt = float(roc_auc_score(y_true, y_score))
        elif metric == 'auprc':
            pt = float(average_precision_score(y_true, y_score))
        elif metric == 'brier':
            pt = float(brier_score_loss(y_true, y_score))
        else:
            raise ValueError("metric must be 'auroc'|'auprc'|'brier'")
    except Exception:
        pt = np.nan

    boots = []
    n_boot = int(n_bootstrap)
    # Generate bootstrap samples with replacement
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)  # Uniform random sampling 
        yt = y_true[idx]; ys = y_score[idx]
        try:
            if metric == 'auroc':
                val = float(roc_auc_score(yt, ys))
            elif metric == 'auprc':
                val = float(average_precision_score(yt, ys))
            else:
                val = float(brier_score_loss(yt, ys))
        except Exception:
            val = np.nan
        boots.append(val)
    
    # Exclude bootstrap samples that resulted in NaN  
    boots = np.array(boots, dtype=float)
    boots = boots[~np.isnan(boots)]
    # Return NaNs if no valid bootstrap samples
    if boots.size == 0:
        return pt, np.nan, np.nan, 0
    # Compute 95% CI via percentile method
    lower = np.percentile(boots, 2.5)
    upper = np.percentile(boots, 97.5)
    return pt, lower, upper, boots.size

def _align_histories(hist_list, key):
    """
    Pad variable-length history arrays to uniform length with NaNs.
    
    Args:
        hist_list: List of history dicts
        key: Key to extract from each history
    
    Returns:
        np.ndarray: NaN-padded 2D array (n_histories x max_length)
        Empty array (0,0) if input is empty
    """
    # Return empty 2D array when no histories
    if not hist_list:
        return np.empty((0, 0), dtype=float)
    
    arrays = []
    for h in hist_list:
        # Safely convert missing entries to empty 1-D arrays
        arr = np.array(h.get(key, []), dtype=float)
        arrays.append(arr)
    
    # Determine maximum length among arrays (0 if all arrays empty)
    max_len = max((a.shape[0] for a in arrays), default=0)

    # Create output array filled with NaN and copy per-row values
    out = np.full((len(arrays), max_len), np.nan, dtype=float)
    for i, a in enumerate(arrays):
        out[i, :a.shape[0]] = a
    return out