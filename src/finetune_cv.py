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
    num_epochs: int = 55
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
def train_model(model, train_loader, val_loader, criterion_train, criterion_val, optimizer, config, class_names=None, output_folder=None):
    """
    Train the model and record metrics per epoch.
    - Saves best checkpoint (by validation AUROC), saves OOF CSV for best epoch, and generates diagnostic plots.
    - Scheduler uses step-level LinearLR warmup followed by CosineAnnealingLR (wrapped in SequentialLR).
    - Metrics recorded: loss, weighted/macro F1, AUROC, AUPRC, Brier, ECE.
    """
    # Create output folder with timestamp if not provided
    if output_folder is None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M')
        output_folder = f"{config.label}_cv_save_{current_time}"
    os.makedirs(output_folder, exist_ok=True)

    # Save config snapshot
    save_config(config, output_folder, 'cv_config.json')   

    # Bookkeeping
    best_val_auc = -np.inf
    train_losses, val_losses, lrs = [], [], []
    
    # Per-epoch metrics lists
    train_weighted_f1_scores = []
    train_macro_f1_scores = []
    train_auroc_scores = []
    train_auprc_scores = []
    train_brier_scores = []
    train_ece_scores = []
    
    val_weighted_f1_scores = []
    val_macro_f1_scores = []
    val_auc_scores = []
    val_auprc_scores = []
    val_brier_scores = []
    val_ece_scores = []

    best_epoch_preds = best_epoch_labels = best_epoch_probs = best_epoch_ids = None
    best_epoch = None
    cm = None

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
            loss = criterion_train(outputs, labels.float().unsqueeze(1))  # BCEwithloss
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
        
        # ==================== Validation stage ====================
        model.eval()
        val_loss = 0.0
        val_preds, val_labels, val_probs, val_ids = [], [], [], []
        
        with torch.no_grad():
            for inputs, labels, ids in val_loader:
                inputs, labels = inputs.to(config.device), labels.to(config.device) 
                outputs = model(inputs)
                loss = criterion_val(outputs, labels.float().unsqueeze(1))  
                val_loss += loss.item()
                
                probs_pos = torch.sigmoid(outputs).squeeze(1)
                preds = (probs_pos >= 0.5).long()

                val_preds.extend(preds.cpu().numpy().tolist())
                val_labels.extend(labels.cpu().numpy().tolist())
                val_probs.extend(probs_pos.detach().cpu().numpy().tolist())
                val_ids.extend(ids)
        
        val_loss /= max(1, len(val_loader))
        val_losses.append(val_loss)
        
        # Compute validation metrics
        val_labels_arr = np.array(val_labels)
        val_preds_arr = np.array(val_preds)
        val_probs_arr = np.array(val_probs)
        if val_probs_arr.size == 0:
            pos_prob_val = np.array([])
        else:
            pos_prob_val = val_probs_arr.ravel()

        try:
            val_auc = roc_auc_score(val_labels_arr, pos_prob_val)
        except Exception as e:
            print(f"Error calculating validation AUC: {e}")
            val_auc = np.nan
        
        try:
            val_auprc = float(average_precision_score(val_labels_arr, pos_prob_val))
        except Exception as e:
            print(f"Val AUPRC error: {e}")
            val_auprc = 0.0

        try:
            val_brier = float(brier_score_loss(val_labels_arr, pos_prob_val))
        except Exception as e:
            print(f"Val Brier error: {e}")
            val_brier = 0.0

        try:
            val_ece = expected_calibration_error(val_labels_arr, pos_prob_val, n_bins=8, strategy='quantile') if pos_prob_val.size>0 else 0.0
        except Exception as e:
            print(f"Val ECE error: {e}")
            val_ece = 0.0

        try:
            val_weighted_f1 = float(f1_score(val_labels_arr, val_preds_arr, average='weighted', zero_division=0))
        except Exception:
            val_weighted_f1 = float(np.nan)

        try:
            val_macro_f1 = float(f1_score(val_labels_arr, val_preds_arr, average='macro', zero_division=0))
        except Exception:
            val_macro_f1 = float(np.nan)
        
        val_auc_scores.append(val_auc)
        val_auprc_scores.append(val_auprc)
        val_brier_scores.append(val_brier)
        val_ece_scores.append(val_ece)
        val_weighted_f1_scores.append(val_weighted_f1)
        val_macro_f1_scores.append(val_macro_f1)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1:2d}/{config.num_epochs} | "
            f"Trn Loss: {epoch_loss:7.4f} | "
            f"Trn F1(w): {train_weighted_f1:7.4f} | "
            f"Trn F1(m): {train_macro_f1:7.4f} | "
            f"Trn AUROC: {train_auc:7.4f} | "
            f"Trn AUPRC: {train_auprc:7.4f} | "
            f"Trn Brier: {train_brier:7.4f} | "
            f"Trn ECE: {train_ece:.4f} | ")

        print(f"Epoch {epoch+1:2d}/{config.num_epochs} | "
            f"Val Loss: {val_loss:7.4f} | "
            f"Val F1(w): {val_weighted_f1:7.4f} | "
            f"Val F1(m): {val_macro_f1:7.4f} | "
            f"Val AUROC: {val_auc:7.4f} | "
            f"Val AUPRC: {val_auprc:7.4f} | "
            f"Val Brier: {val_brier:7.4f} | "
            f"Val ECE: {val_ece:.4f} | ")      

        # Save best model by val AUROC (no early stopping in current training loop) 
        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_model_path = os.path.join(output_folder, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': best_val_auc
            }, best_model_path)

            best_epoch_preds = val_preds_arr
            best_epoch_labels = val_labels_arr
            best_epoch_probs = val_probs_arr
            best_epoch_ids = list(val_ids)

            print(f"Epoch {epoch+1}/{config.num_epochs}: [BEST] Val AUROC: {best_val_auc:.4f}")
        else:
            print(f"Epoch {epoch+1}: Val AUROC did not improve (current: {val_auc:.4f}, best: {best_val_auc:.4f})")

    # Training finished  
    if best_epoch is None:
        print(f"\nTraining completed. No best epoch found (best_val_auc={best_val_auc:.4f}).")
    else:
        print(f"\nTraining completed. Best epoch: {best_epoch+1}/{config.num_epochs}, Best Val AUROC: {best_val_auc:.4f}")

    # Confusion matrix for best epoch (if available)
    if best_epoch_preds is not None and best_epoch_labels is not None and best_epoch_probs is not None:
        cm = confusion_matrix(best_epoch_labels, best_epoch_preds)

    # Save OOF best-epoch records (id, true, pred, prob_0, prob_1) if available
    if best_epoch_preds is not None and best_epoch_labels is not None and best_epoch_probs is not None and best_epoch_ids is not None:
        try:
            oof_records = []
            probs_arr = np.array(best_epoch_probs)
            if probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
                prob1_col = probs_arr[:, 1]
            else:
                prob1_col = probs_arr.ravel()

            for i in range(len(best_epoch_labels)):
                pid = best_epoch_ids[i] if i < len(best_epoch_ids) else f"idx_{i}"
                true = int(best_epoch_labels[i])
                pred = int(best_epoch_preds[i])
                p1 = float(prob1_col[i]) if prob1_col.size > i else 0.0
                p0 = float(1.0 - p1)
                oof_records.append({'id': pid, 'true': true, 'pred': pred, 'prob_0': p0, 'prob_1': p1})
            oof_df = pd.DataFrame(oof_records)
            oof_path = os.path.join(output_folder, 'oof_best_epoch.csv')
            oof_df.to_csv(oof_path, index=False)
            print(f"Saved OOF (best epoch) predictions to: {oof_path}")
        except Exception as e:
            print(f"Failed saving OOF csv: {e}")

        # Plot confusion matrix (best epoch)
        plt.figure(figsize=(10, 8))
        if class_names is None:
            class_names = ['0','1']
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Epoch {best_epoch+1})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'confusion_matrix.pdf'))
        plt.close()
        print(f"Confusion matrix saved: 'confusion_matrix.pdf'")

    # Plot training curves (loss, weighted f1, macro f1, auroc, auprc, brier, ece, lr)
    epoch_numbers = list(range(1, len(train_losses) + 1))

    # 1. Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_losses, label='Train Loss')
    plt.plot(epoch_numbers, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'loss_curve.pdf'))
    plt.close()

    # 2. Weighted F1 curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_weighted_f1_scores, label='Train Weighted F1')
    plt.plot(epoch_numbers, val_weighted_f1_scores, label='Val Weighted F1')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted F1')
    plt.title('Training and Validation Weighted F1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'f1_curve.pdf'))
    plt.close()

    # 3. Macro F1 curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_macro_f1_scores, label='Train Macro F1')
    plt.plot(epoch_numbers, val_macro_f1_scores, label='Val Macro F1')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1')
    plt.title('Training and Validation Macro F1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'macro_f1_curve.pdf'))
    plt.close()

    # 4. AUROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_auroc_scores, label='Train AUROC')
    plt.plot(epoch_numbers, val_auc_scores, label='Val AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title('Training and Validation AUROC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'auroc_curve.pdf'))
    plt.close()

    # 5. AUPRC curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_auprc_scores, label='Train AUPRC')
    plt.plot(epoch_numbers, val_auprc_scores, label='Val AUPRC')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.title('Training and Validation AUPRC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'auprc_curve.pdf'))
    plt.close()

    # 6. Brier curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_brier_scores, label='Train Brier')
    plt.plot(epoch_numbers, val_brier_scores, label='Val Brier')
    plt.xlabel('Epoch')
    plt.ylabel('Brier')
    plt.title('Training and Validation Brier')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'brier_curve.pdf'))
    plt.close()

    # 7. ECE curve
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_ece_scores, label='Train ECE')
    plt.plot(epoch_numbers, val_ece_scores, label='Val ECE')
    plt.xlabel('Epoch')
    plt.ylabel('ECE')
    plt.title('Training and Validation ECE')
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

    # ROC / PR / Calibration plot for best epoch (if available)
    try:
        y_true = np.array(best_epoch_labels)
        probs_arr = np.array(best_epoch_probs)
        if probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
            y_score = probs_arr[:,1]
        else:
            y_score = probs_arr.ravel()

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = sk_auc(fpr, tpr)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
        plt.plot([0,1],[0,1], linestyle='--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC (best)'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(output_folder, 'roc_best.pdf')); plt.close()

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = sk_auc(recall, precision)
        plt.figure(figsize=(6,6))
        plt.plot(recall, precision, label=f"AUPRC={pr_auc:.4f}")
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR (best)'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(output_folder, 'pr_best.pdf')); plt.close()

        # Calibration
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=8, strategy='quantile')
        plt.figure(figsize=(6,6))
        plt.plot(prob_pred, prob_true, marker='o', label='Calibration')
        plt.plot([0,1],[0,1], linestyle='--', label='Perfect')
        plt.xlabel('Mean predicted prob'); plt.ylabel('Fraction positives'); plt.title('Calibration (best)')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'calibration_best.pdf')); plt.close()

    except Exception as e:
            print(f"Failed plotting best-epoch curves: {e}")

    # Save training history to CSV
    history_for_df = {
        'epoch': epoch_numbers,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_weighted_f1_scores': train_weighted_f1_scores,
        'val_weighted_f1_scores': val_weighted_f1_scores,
        'train_macro_f1_scores': train_macro_f1_scores,
        'val_macro_f1_scores': val_macro_f1_scores,
        'train_auroc_scores': train_auroc_scores,
        'val_auroc_scores': val_auc_scores,
        'train_auprc_scores': train_auprc_scores,
        'val_auprc_scores': val_auprc_scores,
        'train_brier_scores': train_brier_scores,
        'val_brier_scores': val_brier_scores,
        'train_ece_scores': train_ece_scores,
        'val_ece_scores': val_ece_scores,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc
    }

    history_df = pd.DataFrame(history_for_df)
    history_path = os.path.join(output_folder, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to '{history_path}'")

    # Save confusion matrix numeric data
    if 'cm' in locals() and cm is not None:
        confusion_df = pd.DataFrame(cm)
        confusion_path = os.path.join(output_folder, 'confusion_matrix.csv')
        confusion_df.to_csv(confusion_path, index=False)
        print(f"Confusion matrix data saved to '{confusion_path}'")

    # Prepare OOF prob1 list for downstream aggregation
    if best_epoch_probs is None:
        oof_prob1_list = []
    else:
        tmp = np.array(best_epoch_probs)
        if tmp.ndim == 2 and tmp.shape[1] >= 2:
            oof_prob1_list = list(tmp[:,1])
        else:
            oof_prob1_list = list(tmp.ravel())
    
    # Return comprehensive training history dictionary
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_weighted_f1_scores': train_weighted_f1_scores,
        'val_weighted_f1_scores': val_weighted_f1_scores,
        'train_macro_f1_scores': train_macro_f1_scores,
        'val_macro_f1_scores': val_macro_f1_scores,
        'train_auroc_scores': train_auroc_scores,
        'val_auroc_scores': val_auc_scores,
        'train_auprc_scores': train_auprc_scores,
        'val_auprc_scores': val_auprc_scores,
        'train_brier_scores': train_brier_scores,
        'val_brier_scores': val_brier_scores,
        'train_ece_scores': train_ece_scores,
        'val_ece_scores': val_ece_scores,
        'lrs': lrs,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'confusion_matrix': cm if 'cm' in locals() else None,
        'output_folder': output_folder,
        'oof_ids': list(best_epoch_ids) if best_epoch_ids is not None else [],
        'oof_true': list(best_epoch_labels) if best_epoch_labels is not None else [],
        'oof_pred': list(best_epoch_preds) if best_epoch_preds is not None else [],
        'oof_prob1': oof_prob1_list
    }


# -----------------------------
# Main fine-tuning 
# -----------------------------
def main_finetune(config):
    """
    Main function to run k-fold cross-validated fine-tuning using a pretrained checkpoint and pretrained k-mer vocabulary.
    - Loads pretrained vocab (kmer_to_idx) and checkpoint.
    - Builds full dataset (using pretrained vocabulary) and runs Stratified K-Fold CV.
    - Replaces classifier with Head_SingleLogit and optionally sets which parameters to train.
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

    # 1. Load pretrained k-mer vocabulary
    try:
        with open(pretrained_kmer_json, 'r') as f:
            pretrained_kmer_to_idx = json.load(f)
        print(f"Loaded pretrained vocab: {len(pretrained_kmer_to_idx)} k-mers")
    except FileNotFoundError:
        raise FileNotFoundError(f"Pretrained vocab file not found: {pretrained_kmer_json}")
    except Exception as e:
        raise RuntimeError(f"Failed loading pretrained_kmer_json {pretrained_kmer_json}: {e}")

    # 2. Build full dataset for K-fold CV (use pretrained kmer_to_idx)
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
    print("Loaded full dataset for 4-fold CV.")
    n_samples = len(full_dataset)
    print(f"Full dataset size: {n_samples}")
    print("Label classes (finetune):", full_dataset.label_encoder.classes_)

    # 3. Prepare indices and labels for StratifiedKFold
    all_indices = np.arange(n_samples)
    labels_array = np.array(full_dataset.labels)  

    # 4. Stratified K-Fold split
    n_splits = 4
    seed = int(getattr(config, 'seed', 42))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Base output directory for all folds
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    base_out_dir = f"{config.label}_cv_save_{timestamp}"
    os.makedirs(base_out_dir, exist_ok=True)

    fold_results = []
    histories = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, labels_array), start=1):
        print(f"\n===== Fold {fold_idx}/{n_splits} =====")
        fold_out = os.path.join(base_out_dir, f"fold_{fold_idx}")
        os.makedirs(fold_out, exist_ok=True)

        # Create Subset objects and DataLoaders
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

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

        # Count positives/negatives in training fold (used if using pos_weight)
        train_labels_fold = labels_array[train_idx]
        n_pos = int((train_labels_fold == 1).sum())
        n_neg = int((train_labels_fold == 0).sum())
        print(f"Fold {fold_idx} train counts: pos={n_pos}, neg={n_neg}")

        # Loss: optionally use pos_weight to balance BCE
        if config.use_class_weights:
            if n_pos == 0:
                raise ValueError("No positive samples in training fold; can't compute pos_weight.")
            pos_weight_scalar = float(n_neg) / float(n_pos)
            pos_weight_tensor = torch.tensor(pos_weight_scalar, dtype=torch.float32, device=config.device)   
            criterion_train = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            print(f"Using BCEWithLogitsLoss with pos_weight = n_neg/n_pos = {pos_weight_scalar:.4f}")
        else:
            criterion_train = nn.BCEWithLogitsLoss()
            print("Using standard BCEWithLogitsLoss (no pos_weight).")

        criterion_val = nn.BCEWithLogitsLoss()

        # Optimizer: only update parameters with requires_grad=True
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=getattr(config, "weight_decay", 0.0)
        )
        print("Optimizer configured (fold).")

        # Run training for this fold, saving outputs under fold_out
        history = train_model(
            model,
            train_loader,
            val_loader,
            criterion_train,
            criterion_val,
            optimizer,
            config,
            class_names=list(full_dataset.label_encoder.classes_),
            output_folder=fold_out
        )
        histories.append(history)

        # Collect best metrics from this fold history
        best_val_auc = float(history.get('best_val_auc', np.nan))
        best_epoch = history.get('best_epoch', None)
        train_aucs = np.array(history.get('train_auroc_scores', []), dtype=float)
        train_auprcs = np.array(history.get('train_auprc_scores', []), dtype=float)
        train_briers = np.array(history.get('train_brier_scores', []), dtype=float)
        val_auprcs = np.array(history.get('val_auprc_scores', []), dtype=float)
        val_briers = np.array(history.get('val_brier_scores', []), dtype=float)

        best_train_auc = safe_index_or_max(train_aucs, best_epoch)
        best_train_auprc = safe_index_or_max(train_auprcs, best_epoch)
        best_train_brier = safe_index_or_max(train_briers, best_epoch)
        best_val_auprc = safe_index_or_max(val_auprcs, best_epoch)
        best_val_brier = safe_index_or_max(val_briers, best_epoch)

        fold_results.append({
            'fold': fold_idx,
            'best_train_auc': best_train_auc,
            'best_train_auprc': best_train_auprc,
            'best_train_brier': best_train_brier,
            'best_val_auc': best_val_auc,
            'best_val_auprc': best_val_auprc,
            'best_val_brier': best_val_brier,
            'output_folder': fold_out
        })

        # Cleanup to free GPU memory
        del model, optimizer, criterion_train, criterion_val
        torch.cuda.empty_cache()

    # Summarize CV results
    mean_train_auc = np.mean([r['best_train_auc'] for r in fold_results])
    std_train_auc = np.std([r['best_train_auc'] for r in fold_results])   
    mean_val_auc = np.mean([r['best_val_auc'] for r in fold_results])  
    std_val_auc = np.std([r['best_val_auc'] for r in fold_results])
    mean_train_auprc = np.mean([r['best_train_auprc'] for r in fold_results])
    std_train_auprc = np.std([r['best_train_auprc'] for r in fold_results])  
    mean_val_auprc = np.mean([r['best_val_auprc'] for r in fold_results])  
    std_val_auprc = np.std([r['best_val_auprc'] for r in fold_results])
    mean_train_brier = np.mean([r['best_train_brier'] for r in fold_results])
    std_train_brier = np.std([r['best_train_brier'] for r in fold_results])  
    mean_val_brier = np.mean([r['best_val_brier'] for r in fold_results])  
    std_val_brier = np.std([r['best_val_brier'] for r in fold_results])

    print("\n[Summary] Per-fold best-epoch metrics:")
    print(f"Train AUROC: mean = {mean_train_auc:.4f}, std = {std_train_auc:.4f}")
    print(f"Validation AUROC: mean = {mean_val_auc:.4f}, std = {std_val_auc:.4f}")
    print(f"Train AUPRC: mean = {mean_train_auprc:.4f}, std = {std_train_auprc:.4f}")
    print(f"Validation AUPRC: mean = {mean_val_auprc:.4f}, std = {std_val_auprc:.4f}")
    print(f"Train Brier: mean = {mean_train_brier:.4f}, std = {std_train_brier:.4f}")
    print(f"Validation Brier: mean = {mean_val_brier:.4f}, std = {std_val_brier:.4f}")
    for r in fold_results:
        print(f"Fold {r['fold']}: Best AUROC = {r['best_val_auc']:.4f} (folder: {r['output_folder']})")
    print(f"All fold outputs saved under: {base_out_dir}")

    # ----------------------------
    # Per-fold OOF aggregation
    # ----------------------------
    # Collect per-fold OOF predictions (best-epoch) and compute per-fold metrics.
    oof_auc_list = []
    oof_auprc_list = []
    oof_brier_list = []
    # Keep per-fold OOF counts for debug and sanity checks
    oof_counts = []

    print("\n[Summary] Per-fold OOF performance:")
    for i, h in enumerate(histories, start=1):
        # Each history is expected to include 'oof_true' and 'oof_prob1' for the best epoch of that fold
        oof_true = np.array(h.get('oof_true', []), dtype=float)
        oof_prob = np.array(h.get('oof_prob1', []), dtype=float)
        
        # Skip folds that did not produce OOF data (safeguard)
        if oof_true.size == 0 or oof_prob.size == 0:
            print(f"Fold {i}: no OOF data found in history (skipping).")
            continue

        # Compute metrics with protection against exceptions (e.g., single-class y_true)
        try:
            auc = float(roc_auc_score(oof_true, oof_prob))
        except Exception as e:
            print(f"Fold {i} AUROC error: {e}")
            auc = np.nan

        try:
            auprc = float(average_precision_score(oof_true, oof_prob))
        except Exception as e:
            print(f"Fold {i} AUPRC error: {e}")
            auprc = np.nan

        try:
            brier = float(brier_score_loss(oof_true, oof_prob))
        except Exception as e:
            print(f"Fold {i} Brier error: {e}")
            brier = np.nan

        oof_auc_list.append(auc)
        oof_auprc_list.append(auprc)
        oof_brier_list.append(brier)
        oof_counts.append(oof_true.size)
        print(f"OOF Fold {i}: n={oof_true.size}, AUROC={auc:.4f}, AUPRC={auprc:.4f}, Brier={brier:.4f}")

    # Compute mean/std across folds while ignoring NaNs
    auc_arr = np.array(oof_auc_list, dtype=float)
    auprc_arr = np.array(oof_auprc_list, dtype=float)
    brier_arr = np.array(oof_brier_list, dtype=float)

    mean_auc = np.nanmean(auc_arr) if auc_arr.size>0 else np.nan
    std_auc  = np.nanstd(auc_arr)  if auc_arr.size>0 else np.nan
    mean_auprc = np.nanmean(auprc_arr) if auprc_arr.size>0 else np.nan
    std_auprc  = np.nanstd(auprc_arr)  if auprc_arr.size>0 else np.nan
    mean_brier = np.nanmean(brier_arr) if brier_arr.size>0 else np.nan
    std_brier  = np.nanstd(brier_arr)  if brier_arr.size>0 else np.nan

    print(f"AUROC: mean={mean_auc:.4f}, std={std_auc:.4f}")
    print(f"AUPRC: mean={mean_auprc:.4f}, std={std_auprc:.4f}")
    print(f"Brier: mean={mean_brier:.4f}, std={std_brier:.4f}")

    # ----------------------------
    # Pooled OOF concatenation
    # ----------------------------
    pooled_true_list = [np.array(h.get('oof_true', []), dtype=float) for h in histories if len(h.get('oof_true', []))>0]
    pooled_prob_list = [np.array(h.get('oof_prob1', []), dtype=float) for h in histories if len(h.get('oof_prob1', []))>0]

    pooled_true = np.concatenate(pooled_true_list) if len(pooled_true_list) > 0 else np.array([])
    pooled_prob = np.concatenate(pooled_prob_list) if len(pooled_prob_list) > 0 else np.array([])

    # pooled_auc = pooled_auprc = pooled_brier = np.nan
    # if pooled_true.size>0 and pooled_prob.size>0:
    #     try:
    #         pooled_auc = float(roc_auc_score(pooled_true, pooled_prob))
    #     except Exception as e:
    #         print(f"Pooled AUROC error: {e}")
    #     try:
    #         pooled_auprc = float(average_precision_score(pooled_true, pooled_prob))
    #     except Exception as e:
    #         print(f"Pooled AUPRC error: {e}")
    #     try:
    #         pooled_brier = float(brier_score_loss(pooled_true, pooled_prob))
    #     except Exception as e:
    #         print(f"Pooled Brier error: {e}")

    # Quick pooled metrics (no CI)
    # print(f"\nOOF pooled (concatenated across folds): AUROC={pooled_auc:.4f}, AUPRC={pooled_auprc:.4f}, Brier={pooled_brier:.4f}")

    # ----------------------------
    # Plot pooled ROC / PR / Calibration with 95% CI via bootstrap
    # ----------------------------
    if pooled_true.size > 0 and pooled_prob.size > 0:
        # Compute ROC / PR curves from pooled OOF
        try:
            fpr, tpr, _ = roc_curve(pooled_true, pooled_prob)
        except Exception as e:
            print(f"Pooled ROC computation failed: {e}")
            fpr = tpr = np.array([])

        try:
            precision, recall, _ = precision_recall_curve(pooled_true, pooled_prob)
        except Exception as e:
            print(f"Pooled PR computation failed: {e}")
            precision = recall = np.array([])

        # Compute AUC / AUPRC / Brier and their 95% CI via bootstrap
        pt_auc, l_auc, u_auc, n_auc_boots = bootstrap_ci(pooled_true, pooled_prob, metric='auroc', n_bootstrap=config.n_boot, seed=config.seed)
        pt_auprc, l_auprc, u_auprc, n_auprc_boots = bootstrap_ci(pooled_true, pooled_prob, metric='auprc', n_bootstrap=config.n_boot, seed=config.seed+1)
        pt_brier, l_brier, u_brier, n_brier_boots = bootstrap_ci(pooled_true, pooled_prob, metric='brier', n_bootstrap=config.n_boot, seed=config.seed+2)

        auc_text = f"AUROC = {pt_auc:.3f}\n95% CI [{l_auc:.3f}, {u_auc:.3f}]"
        auprc_text = f"AUPRC = {pt_auprc:.3f}\n95% CI [{l_auprc:.3f}, {u_auprc:.3f}]"
        brier_text = f"Brier = {pt_brier:.3f}\n95% CI [{l_brier:.3f}, {u_brier:.3f}]"
        print("\n[Summary] Pooled OOF performance (across folds):")
        print(f"Pooled AUROC = {pt_auc:.4f}, 95% CI ({l_auc:.4f} to {u_auc:.4f})")
        print(f"Pooled AUPRC = {pt_auprc:.4f}, 95% CI ({l_auprc:.4f} to {u_auprc:.4f})")
        print(f"Pooled Brier = {pt_brier:.4f}, 95% CI ({l_brier:.4f} to {u_brier:.4f})")

        # Calibration curve 
        try:
            prob_true_bins, prob_pred_bins = calibration_curve(pooled_true, pooled_prob, n_bins=8, strategy='quantile')
        except Exception:
            print(f"Calibration curve computation failed: {e}")
            prob_true_bins, prob_pred_bins = np.array([]), np.array([])

        # Draw 1x3 figure (ROC | PR | Calibration)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

        # ---- ROC panel ----
        ax = axes[0]
        ax.plot(fpr, tpr, lw=3, label='Pooled ROC',color=ROC_COLOR)
        ax.plot([0,1], [0,1], linestyle='--', lw=2, color=REFERENCE_COLOR)
        ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('Pooled ROC')
        ax.text(0.98, 0.02, auc_text, ha='right', transform=ax.transAxes, fontsize=font_text,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax.grid(True)

        # ---- PR panel ----
        # precision_recall_curve returns precision for decreasing thresholds; align recall to be increasing for plotting
        if recall[0] > recall[-1]:
            rec_plot = recall[::-1]; prec_plot = precision[::-1]
        else:
            rec_plot = recall; prec_plot = precision
        ax = axes[1]
        ax.step(rec_plot, prec_plot, where='post', lw=3, label='Pooled PR', color=PR_COLOR)
        ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])   
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('Pooled PR')
        auprc_text = f"AUPRC = {pt_auprc:.3f}\n95% CI [{l_auprc:.3f}, {u_auprc:.3f}]"
        ax.text(0.02, 0.02, auprc_text, transform=ax.transAxes, fontsize=font_text,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax.grid(True)

        # ---- Calibration panel ----
        ax = axes[2]
        if prob_pred_bins.size > 0:
            ax.plot(prob_pred_bins, prob_true_bins, marker='o', lw=3, label='Calibration', color=CALIBRATION_COLOR, markersize=8)
            ax.plot([0,1], [0,1], linestyle='--', lw=2, color=REFERENCE_COLOR)
            ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])   
            ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction positives')
            ax.set_title('Pooled Calibration')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "Calibration data unavailable", ha='center', va='center')
            ax.set_axis_off()

        # Brier annotation in calibration panel  
        ax.text(0.98, 0.02, brier_text, ha='right', transform=ax.transAxes, fontsize=font_text,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # Finalize and save
        plt.tight_layout()
        outpath = os.path.join(base_out_dir, 'pooled_roc_pr_calibration.pdf')
        plt.savefig(outpath, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"\nSaved pooled ROC/PR/Calibration figure: {outpath}")

        # Save individual figures (ROC / PR / Calibration) 
        try:
            # ROC only
            plt.figure(figsize=(5,5), dpi=300)
            plt.plot(fpr, tpr, lw=3, color=ROC_COLOR)
            plt.plot([0,1], [0,1], linestyle='--', lw=2, color=REFERENCE_COLOR)
            plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.01])
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Pooled ROC')
            plt.text(0.98, 0.02, auc_text, ha='right', transform=plt.gca().transAxes, fontsize=font_text, bbox=dict(facecolor='white', alpha=0.6))
            plt.grid(True)
            plt.tight_layout()
            p = os.path.join(base_out_dir, 'pooled_roc.pdf'); plt.savefig(p, dpi=300); plt.close()

            # PR only
            plt.figure(figsize=(5,5), dpi=300)
            plt.step(rec_plot, prec_plot, where='post', lw=3, color=PR_COLOR)
            plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.01])
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])   
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Pooled PR')
            plt.text(0.02, 0.02, auprc_text, transform=plt.gca().transAxes, fontsize=font_text, bbox=dict(facecolor='white', alpha=0.6))
            plt.grid(True)
            plt.tight_layout()
            p = os.path.join(base_out_dir, 'pooled_pr.pdf'); plt.savefig(p, dpi=300); plt.close()

            # Calibration only
            plt.figure(figsize=(5,5), dpi=300)
            if prob_pred_bins.size > 0:
                plt.plot(prob_pred_bins, prob_true_bins, marker='o', lw=3, color=CALIBRATION_COLOR)
                plt.plot([0,1], [0,1], linestyle='--', lw=2, color=REFERENCE_COLOR)
                plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  
                plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction positives'); plt.title('Pooled Calibration')
                plt.text(0.98, 0.02, brier_text, ha='right', transform=plt.gca().transAxes, fontsize=font_text, bbox=dict(facecolor='white', alpha=0.6))
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, "Calibration data unavailable", ha='center', va='center')
            plt.tight_layout()
            p = os.path.join(base_out_dir, 'pooled_calibration.pdf'); plt.savefig(p, dpi=300); plt.close()

            print(f"Also saved separate plots: pooled_roc.pdf, pooled_pr.pdf, pooled_calibration.pdf")
        except Exception as e:
            print(f"Failed saving separate pooled subplots: {e}")
    else:
        print("pooled_true/pooled_prob empty — skipped pooled ROC/PR/Calibration plotting.")

    # -----------------------------
    # Metrics to plot (train vs val)
    # Each tuple: (train_key, val_key, pretty_title)
    # -----------------------------
    metrics_pairs = [
        ('train_losses', 'val_losses', 'Loss'),
        ('train_weighted_f1_scores', 'val_weighted_f1_scores', 'Weighted F1'),
        ('train_macro_f1_scores', 'val_macro_f1_scores', 'Macro F1'),
        ('train_auroc_scores', 'val_auroc_scores', 'AUROC'),
        ('train_auprc_scores', 'val_auprc_scores', 'AUPRC'),
        ('train_brier_scores', 'val_brier_scores', 'Brier Score'),
        ('train_ece_scores', 'val_ece_scores', 'ECE'),
    ]

    os.makedirs(base_out_dir, exist_ok=True)

    # For each metric pair, align histories across folds and plot mean ± std
    for train_key, val_key, title in metrics_pairs:
        train_arr = _align_histories(histories, train_key)  # shape (n_folds, max_epochs_train)
        val_arr   = _align_histories(histories, val_key)  # shape (n_folds, max_epochs_val)

        # Compute epoch lengths  
        epochs_train = train_arr.shape[1] if train_arr.size>0 else 0
        epochs_val   = val_arr.shape[1] if val_arr.size>0 else 0
        max_epochs = max(epochs_train, epochs_val)
        if max_epochs == 0:
            print(f"Skipping aggregated plot for {title}: no data across folds.")
            continue

        x = np.arange(1, max_epochs+1)

        # Compute mean/std across folds while ignoring NaNs
        if train_arr.size > 0:
            train_mean = np.nanmean(train_arr, axis=0)
            train_std  = np.nanstd(train_arr, axis=0)
        else:
            train_mean = np.full((max_epochs,), np.nan)
            train_std  = np.full((max_epochs,), np.nan)

        if val_arr.size > 0:
            val_mean = np.nanmean(val_arr, axis=0)
            val_std  = np.nanstd(val_arr, axis=0)
        else:
            val_mean = np.full((max_epochs,), np.nan)
            val_std  = np.full((max_epochs,), np.nan)

        plt.figure(figsize=(8,6))

        # Optionally draw each fold as a faint line (helps visualize variance)
        for i in range(train_arr.shape[0]):
            plt.plot(np.arange(1, train_arr.shape[1]+1), train_arr[i], color='gray', alpha=0.2, linewidth=0.8)
        for i in range(val_arr.shape[0]):
            plt.plot(np.arange(1, val_arr.shape[1]+1), val_arr[i], color='lightcoral', alpha=0.12, linewidth=0.8)

        # Plot mean ± std  
        if not np.all(np.isnan(train_mean)):
            plt.plot(np.arange(1, train_mean.shape[0]+1), train_mean, label=f'Train {title} (mean)', linewidth=2)
            plt.fill_between(np.arange(1, train_mean.shape[0]+1), train_mean - train_std, train_mean + train_std, alpha=0.25)
        if not np.all(np.isnan(val_mean)):
            plt.plot(np.arange(1, val_mean.shape[0]+1), val_mean, label=f'Val {title} (mean)', linewidth=2)
            plt.fill_between(np.arange(1, val_mean.shape[0]+1), val_mean - val_std, val_mean + val_std, alpha=0.25)

        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(f'{title} across folds (mean ± std)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(base_out_dir, f'aggregated_{train_key}_vs_{val_key}.pdf')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved aggregated plot: {save_path}")

    # Save fold-level summary (CSV)
    summary_path = os.path.join(base_out_dir, 'cv_fold_summary.csv')
    pd.DataFrame(fold_results).to_csv(summary_path, index=False)
    print(f"Saved CV summary to {summary_path}")


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # -----------------------------
    # YAML config loader
    # -----------------------------
    parser = argparse.ArgumentParser(description="Fine-tune cross-validation script")
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

    main_finetune(config)
