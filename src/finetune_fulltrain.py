import os
import sys
import json
import math
import random
import collections
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import yaml
from dataclasses import dataclass, is_dataclass, asdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, average_precision_score, precision_recall_curve, roc_curve, auc as sk_auc, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from Bio import SeqIO
from models import SinusoidalPositionalEncoding, TransformerEncoderModel_Finetune, Head_SingleLogit
from utils import SequenceDataset, validate_and_normalize_config, save_config, safe_index_or_max, expected_calibration_error, bootstrap_ci, _align_histories
import warnings
warnings.filterwarnings('ignore')


# -----------------------------
# Reproducibility  
# -----------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# -----------------------------
# Matplotlib style (global)
# -----------------------------
plt.rcParams.update({
    'figure.dpi': 300,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'font.size': 18,
})
font_text = 18

ROC_COLOR = '#00A163'   
PR_COLOR = '#00A163'   
CALIBRATION_COLOR = '#00A163'   
REFERENCE_COLOR = 'grey'   

# -----------------------------
# Configuration 
# -----------------------------
@dataclass
class Config:
    # Data / preprocessing (must match the pretrained model)
    kmer_size: int = 3  #  
    pretrain_num_classes: int = 6   
    max_seq_length: int = 2048   
    # Model architecture
    embedding_dim: int = 64  # must match the pretrained model
    num_heads: int = 4  # must match the pretrained model
    num_layers: int = 2  # must match the pretrained model
    hidden_dim: int = 192  # must match the pretrained model
    dropout_rate: float = 0.2
    hidden_dim_for_binary_classifier: int = 64
    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 3e-4
    num_epochs: int = 36  # Median of optimal epoch counts across CV folds
    use_class_weights: bool = False
    weight_decay: float = 8e-4
    max_grad_norm: float = 1.0
    # LR scheduling
    start_factor: float = 0.01
    end_factor: float = 1.0
    warmup_epochs: int = 12 
    lr_min: float = 1e-4
    cosine_cycle_epochs: int = 55
    # OOF pooled Bootstrap for final metrics
    n_boot: int = 500
    # Device 
    device: str = "cuda"
    # DataLoader workers
    num_workers: int = 0
    # Seed  
    seed: int = SEED
    # Paths for pretrain checkpoint, kmer vocab and finetune fasta
    pretrained_ckpt_path: str = "pretrain_save_20250923_1727/pretrained_best_model_epoch436.pth"
    pretrained_kmer_json: str = "pretrain_save_20250923_1727/pretrained_kmer_to_idx.json"
    finetune_combined_fasta: str = "../data/preprocessed/finetune/wb_splits/wb_finetune_trainval_set.fasta"
    reference_fasta: str = "../data/raw/mRNA_045_WT.fasta"
    # Label type
    label: str = "wb"  # indicate wb or elisa

# -----------------------------
# Training loop
# ----------------------------- 
def train_model(model, train_loader, criterion, optimizer, config, class_names=None, output_folder=None):
    """
    Train the model on the fulltrain set and record metrics per epoch.
    - Saves a final checkpoint for held-out testing.
    - Scheduler uses step-level LinearLR warmup followed by CosineAnnealingLR (wrapped in SequentialLR).
    - Metrics recorded: loss, weighted/macro F1, AUROC, AUPRC, Brier, ECE (for debugging purposes).
    """
    # Create output folder with timestamp if not provided
    if output_folder is None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M')
        output_folder = f"{config.label}_fulltrain_save_{current_time}"
    os.makedirs(output_folder, exist_ok=True)

    # Save config snapshot
    save_config(config, output_folder, 'fulltrain_config.json') 

    # Bookkeeping
    train_losses, lrs = [], []
    
    # Per-epoch metrics lists
    train_weighted_f1_scores = []
    train_macro_f1_scores = []
    train_auroc_scores = []
    train_auprc_scores = []
    train_brier_scores = []
    train_ece_scores = []
    
    best_train_auc = -np.inf
    best_epoch = None

    # Scheduler: convert epoch-based warmup to step-based iter count
    num_batches_per_epoch = len(train_loader)
    total_steps = int(config.cosine_cycle_epochs * num_batches_per_epoch)
    total_warmup_steps = max(1, int(config.warmup_epochs * num_batches_per_epoch))
    if total_steps <= total_warmup_steps:
        total_steps = total_warmup_steps + 1
    remaining_steps = total_steps - total_warmup_steps  # >=1

    # Per-step linear warmup and cosine annealing on steps
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(config.start_factor),
        end_factor=float(config.end_factor),
        total_iters=total_warmup_steps
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=remaining_steps,
        eta_min=float(config.lr_min)
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[total_warmup_steps]
    )

        # Initialize global step counter
    global_step = 0
    print(f"\nWarmup will last for {config.warmup_epochs} epochs ({total_warmup_steps} steps).")
 
    # ==================== Training stage ====================
    print(f"Starting training for {config.num_epochs} epochs, running on {config.device}...")
    
    for epoch in range(config.num_epochs):
        model.train()

        running_loss = 0.0
        train_preds, train_labels, train_probs = [], [], []
        
        for inputs, labels, ids in train_loader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            # Using BCEWithLogitsLoss: outputs expected to be single-logit per sample for binary setup
            loss = criterion(outputs, labels.float().unsqueeze(1))  # BCEwithloss
            loss.backward()

            # Gradient clipping only on trainable params
            if getattr(config, "max_grad_norm", None) is not None and config.max_grad_norm > 0:
                params_to_clip = [p for p in model.parameters() if p.requires_grad]
                clip_grad_norm_(params_to_clip, config.max_grad_norm)
                    
            optimizer.step()

            # Invalidate cached reference embedding after optimizer update to ensure it's up-to-date if weights changed
            if hasattr(model, "_reference_embed"):
                model._reference_embed = None
            
            running_loss += loss.item()
                
            # Single-logit binary: convert to probability with sigmoid and threshold at 0.5 for preds
            probs_pos = torch.sigmoid(outputs).squeeze(1)   # shape (B,)
            preds = (probs_pos >= 0.5).long()

            train_preds.extend(preds.cpu().numpy().tolist())
            train_labels.extend(labels.cpu().numpy().tolist())
            train_probs.extend(probs_pos.detach().cpu().numpy().tolist())

            # Per-step scheduler stepping
            global_step += 1
            if global_step <= total_steps:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                old_lr = optimizer.param_groups[0]['lr']
                current_lr = float(config.lr_min)
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
            lrs.append(current_lr)

            # Periodic LR logging
            if global_step % 100 == 0:
                if global_step <= total_warmup_steps:
                    print(f"Step {global_step}/{total_warmup_steps}: Warmup LR {old_lr:.2e} -> {current_lr:.2e}")
                elif global_step <= total_steps:
                    step_in_cosine = global_step - total_warmup_steps
                    print(f"Step {step_in_cosine}/{remaining_steps}: Cosine LR {old_lr:.2e} -> {current_lr:.2e}")
                else:
                    print(f"Step {global_step}: Fixed LR {old_lr:.2e} -> {current_lr:.2e}")

        # Epoch-level bookkeeping
        epoch_loss = running_loss / max(1, len(train_loader))
        train_losses.append(epoch_loss)

        # Compute training metrics 
        train_labels_arr = np.array(train_labels)
        train_preds_arr = np.array(train_preds)
        train_probs_arr = np.array(train_probs)
        if train_probs_arr.size == 0:
            pos_prob_train = np.array([])
        else:
            pos_prob_train = train_probs_arr.ravel()   

        try:
            train_auc = float(roc_auc_score(train_labels_arr, pos_prob_train))
        except Exception as e:
            print(f"Train AUC error: {e}")
            train_auc = 0.0
        
        try:
            train_auprc = float(average_precision_score(train_labels_arr, pos_prob_train))
        except Exception as e:
            print(f"Train AUPRC error: {e}")
            train_auprc = 0.0  

        try:
            train_brier = float(brier_score_loss(train_labels_arr, pos_prob_train))
        except Exception as e:
            print(f"Train Brier error: {e}")
            train_brier = 0.0

        try:
            train_ece = expected_calibration_error(train_labels_arr, pos_prob_train, n_bins=8, strategy='quantile')
        except Exception as e:
            print(f"Train ECE error: {e}")
            train_ece = 0.0
        
        try:
            train_weighted_f1 = float(f1_score(train_labels_arr, train_preds_arr, average='weighted', zero_division=0))
        except Exception:
            train_weighted_f1 = float(np.nan)

        try:
            train_macro_f1 = float(f1_score(train_labels_arr, train_preds_arr, average='macro', zero_division=0))
        except Exception:
            train_macro_f1 = float(np.nan)

        train_auroc_scores.append(train_auc)
        train_auprc_scores.append(train_auprc)
        train_brier_scores.append(train_brier)
        train_ece_scores.append(train_ece)
        train_weighted_f1_scores.append(train_weighted_f1)
        train_macro_f1_scores.append(train_macro_f1)

        # Print epoch summary
        print(f"\nEpoch {epoch+1:2d}/{config.num_epochs} | "
            f"Trn Loss: {epoch_loss:7.4f} | "
            f"Trn F1(w): {train_weighted_f1:7.4f} | "
            f"Trn F1(m): {train_macro_f1:7.4f} | "
            f"Trn AUROC: {train_auc:7.4f} | "
            f"Trn AUPRC: {train_auprc:7.4f} | "
            f"Trn Brier: {train_brier:7.4f} | "
            f"Trn ECE: {train_ece:.4f} | ")

        # Save best model based on training AUROC (for debugging purposes)
        if not (np.isnan(train_auc)):
            if train_auc > best_train_auc:
                best_train_auc = train_auc
                best_epoch = epoch
                best_model_path = os.path.join(output_folder, 'best_model_by_train.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() else None,
                    'best_train_auc': best_train_auc
                }, best_model_path)
                print(f"Epoch {epoch+1}/{config.num_epochs}: [BEST_BY_TRAIN] Train AUROC: {best_train_auc:.4f}")

    # Training finished  
    final_model_path = os.path.join(output_folder, 'final_model.pth')
    torch.save({
        'epoch': config.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() else None,
        'best_train_auc': best_train_auc
    }, final_model_path)
    print(f"\nTraining completed. Best train AUROC: {best_train_auc:.4f}")
    print(f"Saved final checkpoint: {final_model_path}")

    # Plot training curves (loss, weighted f1, macro f1, auroc, auprc, brier, ece, lr) for debugging purposes
    epoch_numbers = list(range(1, len(train_losses) + 1))

    # 1. Loss curve 
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'loss_curve.pdf'))
    plt.close()

    # 2. Weighted F1 curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_weighted_f1_scores, label='Train Weighted F1')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted F1')
    plt.title('Training Weighted F1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'f1_curve.pdf'))
    plt.close()

    # 3. Macro F1 curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_macro_f1_scores, label='Train Macro F1')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1')
    plt.title('Training Macro F1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'macro_f1_curve.pdf'))
    plt.close()

    # 4. AUROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_auroc_scores, label='Train AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title('Training AUROC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'auroc_curve.pdf'))
    plt.close()

    # 5. AUPRC curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_auprc_scores, label='Train AUPRC')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.title('Training AUPRC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'auprc_curve.pdf'))
    plt.close()

    # 6. Brier curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_brier_scores, label='Train Brier')
    plt.xlabel('Epoch')
    plt.ylabel('Brier')
    plt.title('Training Brier')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'brier_curve.pdf'))
    plt.close()

    # 7. ECE curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_ece_scores, label='Train ECE')
    plt.xlabel('Epoch')
    plt.ylabel('ECE')
    plt.title('Training ECE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'ECE_curve.pdf'))
    plt.close()
    print("Training curves saved: 'loss_curve.pdf', 'f1_curve', 'macro_f1_curve', 'auroc_curve.pdf', 'auprc_curve.pdf', 'brier_curve.pdf', 'ECE_curve.pdf'")

    # LR curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(lrs)), lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'lr_curve.pdf'))
    plt.close()
    print("Learning rate curve saved: 'lr_curve.pdf'")
    plt.close()

    # Save training history to CSV
    history_for_df = {
        'epoch': epoch_numbers,
        'train_losses': train_losses,
        'train_weighted_f1_scores': train_weighted_f1_scores,
        'train_macro_f1_scores': train_macro_f1_scores,
        'train_auroc_scores': train_auroc_scores,
        'train_auprc_scores': train_auprc_scores,
        'train_brier_scores': train_brier_scores,
        'train_ece_scores': train_ece_scores,
        'best_epoch': best_epoch,
        'best_train_auc': best_train_auc
    }

    history_df = pd.DataFrame(history_for_df)
    history_path = os.path.join(output_folder, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to '{history_path}'")

    return {
        'train_losses': train_losses,
        'train_weighted_f1_scores': train_weighted_f1_scores,
        'train_macro_f1_scores': train_macro_f1_scores,
        'train_auroc_scores': train_auroc_scores,
        'train_auprc_scores': train_auprc_scores,
        'train_brier_scores': train_brier_scores,
        'train_ece_scores': train_ece_scores,
        'lrs': lrs,
        'best_epoch': best_epoch,
        'best_train_auc': best_train_auc,
        'output_folder': output_folder
    }


# -----------------------------
# Main fulltrain  
# ----------------------------
def main_fulltrain(config):
    """
    Run full-data fine-tuning using a pretrained checkpoint and pretrained k-mer vocabulary.
    - Loads reference sequence, pretrained vocab and checkpoint.
    - Builds full dataset (using pretrained kmer_to_idx) and replaces classifier with Head_SingleLogit.
    - Saves vocab, label mapping and a final checkpoint for held-out testing.  
    """
    # Read paths from config  
    pretrained_ckpt_path = config.pretrained_ckpt_path
    pretrained_kmer_json = config.pretrained_kmer_json
    finetune_combined_fasta = config.finetune_combined_fasta
    reference_fasta = config.reference_fasta

    # Read reference (wild-type) sequence (take last record if multiple)
    try:
        wt_seq = None
        for record in SeqIO.parse(reference_fasta, "fasta"):
            wt_seq = str(record.seq)
            print(f"Loaded reference (wild-type) sequence: {record.id}")
        if wt_seq is None:
            raise ValueError(f"Reference fasta contains no sequences: {reference_fasta}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference fasta not found: {reference_fasta}")
    except Exception as e:
        raise RuntimeError(f"Failed reading reference fasta {reference_fasta}: {e}")

    # Load pretrained k-mer vocabulary
    try:
        with open(pretrained_kmer_json, 'r') as f:
            pretrained_kmer_to_idx = json.load(f)
        print(f"Loaded pretrained vocab: {len(pretrained_kmer_to_idx)} k-mers")
    except FileNotFoundError:
        raise FileNotFoundError(f"Pretrained vocab file not found: {pretrained_kmer_json}")
    except Exception as e:
        raise RuntimeError(f"Failed loading pretrained_kmer_json {pretrained_kmer_json}: {e}")

    # Build full dataset for fulltrain (use pretrained kmer_to_idx)
    full_dataset = SequenceDataset(
        fasta_file=finetune_combined_fasta,
        kmer_size=config.kmer_size,
        max_length=config.max_seq_length,
        is_train=False,
        kmer_to_idx=pretrained_kmer_to_idx,
        label_encoder=None,
        compute_class_weights=False,
        reference_sequence=wt_seq,
        label_type=config.label
    )
    print("Loaded fulltrain dataset.")
    n_samples = len(full_dataset)
    print(f"Full dataset size: {n_samples}")
    print("Label classes (finetune):", full_dataset.label_encoder.classes_)

    # DataLoader for full dataset
    train_loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # Initialize model and load pretrained checkpoint
    ref_idx = full_dataset.get_reference_indices()
    model = TransformerEncoderModel_Finetune(
        vocab_size=len(pretrained_kmer_to_idx) + 1,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        num_classes=config.pretrain_num_classes,
        dropout_rate=0.0,
        max_length=config.max_seq_length,
        reference_indices=ref_idx if ref_idx is not None else None
    ).to(config.device)

    if os.path.exists(pretrained_ckpt_path):
        state = torch.load(pretrained_ckpt_path, map_location=config.device)
        sd = state.get('model_state_dict', state) if isinstance(state, dict) else state
        try:
            model.load_state_dict(sd, strict=True)
            print("Loaded pretrained checkpoint (strict=True).")
        except Exception as e:
            model.load_state_dict(sd, strict=False)
            print("Loaded pretrained checkpoint with strict=False (classifier may have mismatch).\n")
            # sys.exit(1)
    else:
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_ckpt_path}")
    
    # -----------------------------------------
    # Freeze / unfreeze logic
    # -----------------------------------------
    # Current implementation: leave all params trainable (full fine-tuning)
    for param in model.parameters():
        param.requires_grad = True
    for p in model.transformer_encoder.layers[-1].parameters():
        p.requires_grad = True

    # Replace classifier head with a single-logit head for binary BCEWithLogitsLoss
    model.classifier = Head_SingleLogit(
        in_dim=config.embedding_dim, 
        hidden_dim=config.hidden_dim_for_binary_classifier, 
        dropout=config.dropout_rate 
        ).to(config.device)
    # Ensure classifier parameters require gradients
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Print trainable param summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} ({trainable/1e6:.3f}M)")
    # trainable_params = [n for n,p in model.named_parameters() if p.requires_grad]
    # print("Trainable params (names):", trainable_params)
    # print(model)

    # Sanity check: expect binary labels for the single-logit + BCE setup
    num_classes = len(full_dataset.label_encoder.classes_)
    if num_classes != 2:
        raise ValueError(f"Expected binary finetune labels (2 classes) for BCE single-logit setup, got {num_classes} classes.")

    # Print label mapping (index -> original label string)
    print("Label mapping (index -> class_name):", dict(enumerate(full_dataset.label_encoder.classes_)))

    # Count positives/negatives in fulltrain set  
    labels_all = np.array(full_dataset.labels)
    n_pos = int((labels_all == 1).sum())
    n_neg = int((labels_all == 0).sum())
    print(f"Full-data counts: pos={n_pos}, neg={n_neg}")

    # Loss: optionally use pos_weight to balance BCE
    if config.use_class_weights:
        if n_pos == 0:
            raise ValueError("No positive samples in train fold; can't compute pos_weight.")
        pos_weight_scalar = float(n_neg) / float(n_pos)
        pos_weight_tensor = torch.tensor(pos_weight_scalar, dtype=torch.float32, device=config.device)   
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"Using BCEWithLogitsLoss with pos_weight = n_neg/n_pos = {pos_weight_scalar:.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using standard BCEWithLogitsLoss (no pos_weight).")

    # Optimizer: only update parameters with requires_grad=True
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=getattr(config, "weight_decay", 0.0)
    )
    print("Optimizer configured.")
 
    # Base output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_folder = f"{config.label}_fulltrain_save_{timestamp}"
    os.makedirs(out_folder, exist_ok=True)

    # Save label encoder classes
    le_classes_file = os.path.join(out_folder, 'label_encoder_classes.json')
    try:
        with open(le_classes_file, 'w') as f:
            json.dump(list(full_dataset.label_encoder.classes_), f)
        print(f"Saved label encoder classes to: {le_classes_file}")
    except Exception as e:
        print("Warning: failed to save label encoder classes:", e)

    # Save k-mer vocabulary
    kmer_save_path = os.path.join(out_folder, 'kmer_to_idx.json')
    try:
        # Prefer dataset-generated vocabulary if available, or fallback to pre-trained vocabulary
        if hasattr(full_dataset, 'kmer_to_idx') and full_dataset.kmer_to_idx:
            kmer_to_save = full_dataset.kmer_to_idx
        else:
            kmer_to_save = pretrained_kmer_to_idx
            # sys.exit(1)
        with open(kmer_save_path, 'w') as f:
            json.dump(kmer_to_save, f, indent=2, ensure_ascii=False)
        print(f"Saved k-mer vocab to: {kmer_save_path} (vocab size: {len(kmer_to_save)})")
    except Exception as e:
        print("Warning: failed to save k-mer vocab:", e)
        
    # Run fulltrain 
    history = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        config,
        # class_names=list(full_dataset.label_encoder.classes_),
        output_folder=out_folder
    )
    print("Training on full dataset finished. Outputs saved to:", out_folder)
    return history


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # -----------------------------
    # YAML config loader
    # -----------------------------
    parser = argparse.ArgumentParser(description="Fine-tune full-training script")
    parser.add_argument('--config', type=str, default='configs/finetune.yaml', help='Path to YAML config file (optional); set "" to skip')
    parser.add_argument('--override', nargs='*', default=None, help='Optional overrides, e.g. --override learning_rate=1e-4 batch_size=8')
    args, _ = parser.parse_known_args()

    # Start from dataclass defaults
    config = Config()
    loaded_yaml = False

    # Load yaml if provided and exists (allow user to pass empty string to skip)
    if args.config:
        cfg_path = args.config
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                yaml_cfg = yaml.safe_load(f) or {}
            if not isinstance(yaml_cfg, dict):
                print(f"Warning: YAML at {cfg_path} did not produce a mapping (dict). Ignored.")
                yaml_cfg = {}
            for k, v in yaml_cfg.items():
                if hasattr(config, k):
                    setattr(config, k, v)
                else:
                    print(f"Warning: unknown config key in YAML: {k} (ignored)")
            loaded_yaml = True
        else:
            print(f"YAML config not found at {cfg_path}; using dataclass defaults (you can create it or pass --config path).")
    else:
        print("No --config passed (empty); using dataclass defaults and overrides only if provided.")

    # Overrides in key=value format
    if args.override:
        for item in args.override:
            if '=' not in item:
                print(f"Warning: invalid override '{item}', expected key=value (ignored)")
                continue
            k, v = item.split('=', 1)
            if not hasattr(config, k):
                print(f"Warning: unknown override key: {k} (ignored)")
                continue
            # parse simple literals safely (numbers, bools, lists, dicts, null)
            try:
                parsed = yaml.safe_load(v)
            except Exception:
                parsed = v
            setattr(config, k, parsed)

    # -----------------------------
    # Validate numeric config 
    # -----------------------------
    config = validate_and_normalize_config(config, yaml_cfg if 'yaml_cfg' in locals() else None)
    # Sanity print 
    # print("\nDEBUG: config types after normalization:")
    # print("  learning_rate ->", type(config.learning_rate), config.learning_rate)
    # print("  weight_decay  ->", type(config.weight_decay), config.weight_decay)
    # print("  batch_size    ->", type(config.batch_size), config.batch_size)
    # print("  device        ->", type(config.device), config.device)
    # print("  seed          ->", type(config.seed), config.seed)

    # Device parsing
    if isinstance(config.device, str):
        dev_str = config.device.strip()
        # allow user to specify 'cpu', 'cuda', 'cuda:0', etc.
        if dev_str.startswith('cuda') and not torch.cuda.is_available():
            print("Warning: cuda requested but unavailable -> falling back to cpu.")
            config.device = torch.device('cpu')
        else:
            try:
                config.device = torch.device(dev_str)
            except Exception:
                print(f"Warning: failed to parse device '{dev_str}' -> falling back to cpu.")
                config.device = torch.device('cpu')

    # Re-seed using config.seed to guarantee reproducibility from config
    SEED = int(getattr(config, 'seed', SEED))
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ---- Read required paths from config ----
    pretrained_ckpt_path = getattr(config, 'pretrained_ckpt_path', None)
    pretrained_kmer_json = getattr(config, 'pretrained_kmer_json', None)
    finetune_combined_fasta = getattr(config, 'finetune_combined_fasta', None)
    reference_fasta = getattr(config, 'reference_fasta', None)

    # Check paths 
    required_paths = {
        'pretrained_ckpt_path': pretrained_ckpt_path,
        'pretrained_kmer_json': pretrained_kmer_json,
        'finetune_combined_fasta': finetune_combined_fasta,
        'reference_fasta': reference_fasta
    }
    for name, p in required_paths.items():
        if p is None:
            raise ValueError(f"Missing required config.{name}. Please set it in the Config dataclass or in your YAML.")
        # optional: check existence now
        if not os.path.exists(p):
            raise FileNotFoundError(f"Path for config.{name} not found: {p}")

    # Final logging
    src = f"YAML: {args.config}" if loaded_yaml else "dataclass defaults"
    print(f"Config source: {src}; overrides: {args.override}")
    print("Final config (after YAML/overrides):", config, "\n")

    main_fulltrain(config)