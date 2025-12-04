# Copyright (c) 2025 Beijing Youcare Kechuang Pharmaceutical Technology Co. Ltd., All rights reserved.
# Author: Dawei Wang
# This code may not be used, modified, or distributed without prior written consent from Beijing Youcare Kechuang Pharmaceutical Technology Co. Ltd.

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
from utils import SequenceDataset, validate_and_normalize_config, save_config, expected_calibration_error, bootstrap_ci
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

ROC_COLOR = '#FF0066'  
PR_COLOR = '#FF0066'  
CALIBRATION_COLOR = '#FF0066'  
REFERENCE_COLOR = 'grey'   

# -----------------------------
# Configuration 
# -----------------------------
@dataclass
class Config:
    # Data / preprocessing (must match the fulltrained model)
    kmer_size: int = 3
    pretrain_num_classes: int = 6   
    max_seq_length: int = 2048   
    # Model architecture
    embedding_dim: int = 64  # must match the fulltrained model
    num_heads: int = 4  # must match the fulltrained model
    num_layers: int = 2  # must match the fulltrained model
    hidden_dim: int = 192  # must match the fulltrained model
    dropout_rate: float = 0.2
    hidden_dim_for_binary_classifier: int = 64
    # Batch size
    batch_size: int = 4
    # Bootstrap for final metrics
    n_boot: int = 500
    # Device 
    device: str = "cuda"
    # DataLoader workers
    num_workers: int = 0
    # Seed  
    seed: int = SEED
    # Paths for fulltrained checkpoint, kmer vocab, label encoder class, test fasta and reference fasta
    fulltrained_ckpt_path: str = "wb_fulltrain_save_20251028_1702/final_model.pth"
    fulltrained_kmer_json: str = "wb_fulltrain_save_20251028_1702/kmer_to_idx.json"
    fulltrained_le_classes_path: str = "wb_fulltrain_save_20251028_1702/label_encoder_classes.json"
    finetune_test_fasta: str = "../data/preprocessed/finetune/wb_splits/wb_finetune_test_set.fasta"
    reference_fasta: str = "../data/raw/mRNA_045_WT.fasta"
    # Label type
    label: str = "wb"  # indicate wb or elisa   


# -----------------------------
# Testing
# ----------------------------- 
def test_model(model, test_loader, config, class_names=None, output_folder=None):
    """
    Evaluate model on held-out test set and record metrics.
    - Load the final checkpoint from full-training.
    - Metrics recorded: loss, weighted/macro F1, AUROC, AUPRC, Brier, ECE.
    """
    # Create output folder with timestamp if not provided
    if output_folder is None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M')
        output_folder = f"{config.label}_test_save_{current_time}"
    os.makedirs(output_folder, exist_ok=True)

    # Save config snapshot
    save_config(config, output_folder, 'test_config.json') 

    # Evaluate model
    model.eval()
    all_ids, all_labels, all_preds, all_probs = [], [], [], []

    with torch.no_grad():
        for inputs, labels, ids in test_loader:
            inputs = inputs.to(config.device)
            outputs = model(inputs)
            probs_pos = torch.sigmoid(outputs).squeeze(1)   
            preds = (probs_pos >= 0.5).long()

            all_ids.extend(ids)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs_pos.detach().cpu().numpy().tolist())

    # Compute metrics
    all_labels_arr = np.array(all_labels, dtype=int)
    all_preds_arr = np.array(all_preds, dtype=int)
    all_probs_arr = np.array(all_probs, dtype=float)
    if all_probs_arr.size == 0:
        pos_prob = np.array([])
    else:
        pos_prob = all_probs_arr.ravel()   

    try:
        auroc = float(roc_auc_score(all_labels_arr, pos_prob))
    except Exception as e:
        print("AUROC error:", e); auroc = np.nan
    try:
        auprc = float(average_precision_score(all_labels_arr, pos_prob))
    except Exception as e:
        print("AUPRC error:", e); auprc = np.nan
    try:
        brier = float(brier_score_loss(all_labels_arr, pos_prob))
    except Exception as e:
        print("Brier error:", e); brier = np.nan
    try:
        ece = expected_calibration_error(all_labels_arr, pos_prob, n_bins=8, strategy='quantile') if pos_prob.size>0 else np.nan
    except Exception as e:
        print("ECE error:", e); ece = np.nan
    try:
        weighted_f1 = float(f1_score(all_labels_arr, all_preds_arr, average='weighted', zero_division=0))
    except Exception as e:
        weighted_f1 = np.nan
    try:
        macro_f1 = float(f1_score(all_labels_arr, all_preds_arr, average='macro', zero_division=0))
    except Exception as e:
        macro_f1 = np.nan
    
    print("\nHeld-out test metrics:")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"Brier: {brier:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    # Compute confidence intervals of selected metrics
    if pos_prob.size > 0 and len(np.unique(all_labels_arr)) >= 2:
        auroc_pt, auroc_lo, auroc_hi, auroc_n = bootstrap_ci(all_labels_arr, pos_prob, metric='auroc', n_bootstrap=config.n_boot, seed=config.seed)
        auprc_pt, auprc_lo, auprc_hi, auprc_n = bootstrap_ci(all_labels_arr, pos_prob, metric='auprc', n_bootstrap=config.n_boot, seed=config.seed+1)
        brier_pt, brier_lo, brier_hi, brier_n = bootstrap_ci(all_labels_arr, pos_prob, metric='brier', n_bootstrap=config.n_boot, seed=config.seed+2)

    else:
        if pos_prob.size == 0:
            print("[evaluate] pos_prob is empty -> skipping bootstrap.")
        else:
            print("[evaluate] only one class present in all_labels_arr -> skipping bootstrap.")
        auroc_pt = auroc_lo = auroc_hi = np.nan
        auprc_pt = auprc_lo = auprc_hi = np.nan
        brier_pt = brier_lo = brier_hi = np.nan
        auroc_n = auprc_n = brier_n = 0

    print("\nHeld-out test metrics (with bootstrap 95% CI):")
    if not np.isnan(auroc_pt):
        if not np.isnan(auroc_lo):
            print(f"AUROC: {auroc_pt:.4f} (95% CI {auroc_lo:.4f} - {auroc_hi:.4f}, n_boots={auroc_n})")
        else:
            print(f"AUROC: {auroc_pt:.4f} (95% CI unavailable)")
    else:
        print("AUROC: unavailable")

    if not np.isnan(auprc_pt):
        if not np.isnan(auprc_lo):
            print(f"AUPRC: {auprc_pt:.4f} (95% CI {auprc_lo:.4f} - {auprc_hi:.4f}, n_boots={auprc_n})")
        else:
            print(f"AUPRC: {auprc_pt:.4f} (95% CI unavailable)")
    else:
        print("AUPRC: unavailable")

    if not np.isnan(brier_pt):
        if not np.isnan(brier_lo):
            print(f"Brier: {brier_pt:.4f} (95% CI {brier_lo:.4f} - {brier_hi:.4f}, n_boots={brier_n})")
        else:
            print(f"Brier: {brier_pt:.4f} (95% CI unavailable)")
    else:
        print("Brier: unavailable")
 
    # Save held-out predictions (id, true, pred, prob_0, prob_1) 
    assert len(all_ids) == len(all_labels) == len(all_preds) == len(all_probs), "Length mismatch in outputs"
    pred_records = []
    for i in range(len(all_labels)):
        pid = all_ids[i] if i < len(all_ids) else f"idx_{i}"
        true = int(all_labels[i])
        pred = int(all_preds[i])
        if all_probs_arr.size == 0:
            prob1 = 0.0
            prob0 = 1.0
        else:
            prob1 = float(all_probs[i])  # single-logit -> one value per sample
            prob0 = 1.0 - prob1
        pred_records.append({'id': pid, 'true': true, 'pred': pred, 'prob_0': prob0, 'prob_1': prob1})
    pred_df = pd.DataFrame(pred_records)
    pred_path = os.path.join(output_folder, 'test_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved held-out test predictions: {pred_path}")

    # Plot confusion matrix
    try:
        cm = confusion_matrix(all_labels, all_preds)
        if class_names is None:
            class_names = ['0','1']
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        plt.figure(figsize=(10,8))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Held-out)')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_png = os.path.join(output_folder, 'confusion_matrix.pdf'); plt.savefig(cm_png); plt.close()
        cm_csv = os.path.join(output_folder, 'confusion_matrix.csv'); df_cm.to_csv(cm_csv, index=True)
        print(f"Saved confusion matrix pdf/csv: {cm_png}, {cm_csv}")
    except Exception as e:
        print("Failed to compute/save confusion matrix:", e)

    # Draw 1x3 figure (ROC | PR | Calibration)
    if pos_prob.size == 0 or len(np.unique(all_labels_arr)) < 2:
        print("Not enough data to draw ROC/PR/Calibration (empty probabilities or single class).")
    else:
        try: 
            fpr, tpr, _ = roc_curve(all_labels_arr, pos_prob)
            precision, recall, _ = precision_recall_curve(all_labels_arr, pos_prob)
            prob_true_bins, prob_pred_bins = calibration_curve(all_labels_arr, pos_prob, n_bins=8, strategy='quantile')

            # 1x3 figure
            fig, axes = plt.subplots(1, 3, figsize=(15,5), dpi=300)

            # ---- ROC panel ----
            ax = axes[0]
            ax.plot(fpr, tpr, lw=3, label='ROC (Held-out)', color=ROC_COLOR)
            ax.plot([0,1],[0,1], linestyle='--', lw=2, color=REFERENCE_COLOR)
            ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  
            ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.set_title('ROC (Held-out)')
            auc_text = f"AUROC = {auroc_pt:.3f}\n95% CI [{auroc_lo:.3f}, {auroc_hi:.3f}]"
            ax.text(0.98, 0.02, auc_text, ha='right', transform=ax.transAxes, fontsize=font_text, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            ax.grid(True)

            # ---- PR panel ----
            if recall[0] > recall[-1]:
                rec_plot = recall[::-1]; prec_plot = precision[::-1]
            else:
                rec_plot = recall; prec_plot = precision
            ax = axes[1]
            ax.step(rec_plot, prec_plot, where='post', lw=3, label='PR (Held-out)', color=PR_COLOR)
            ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.01])
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])   
            ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('PR (Held-out)')
            auprc_text = f"AUPRC = {auprc_pt:.3f}\n95% CI [{auprc_lo:.3f}, {auprc_hi:.3f}]"
            ax.text(0.02, 0.02, auprc_text, transform=ax.transAxes, fontsize=font_text, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            ax.grid(True)

            # ---- Calibration panel ----
            ax = axes[2]
            if prob_pred_bins.size > 0:
                ax.plot(prob_pred_bins, prob_true_bins, marker='o', lw=3, label='Calibration (Held-out)', color=CALIBRATION_COLOR, markersize=8)
                ax.plot([0,1],[0,1], linestyle='--', lw=2, color=REFERENCE_COLOR)
                ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.01])
                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])   
                ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction positives'); ax.set_title('Calibration (Held-out)')
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, "Calibration data unavailable", ha='center', va='center')
                ax.set_axis_off()

            # Brier annotation in calibration panel  
            brier_text = f"Brier = {brier_pt:.3f}\n95% CI [{brier_lo:.3f}, {brier_hi:.3f}]"
            ax.text(0.98, 0.02, brier_text, ha='right', transform=ax.transAxes, fontsize=font_text, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

            # Finalize and save
            plt.tight_layout()
            combined_png = os.path.join(output_folder, 'test_roc_pr_calibration.pdf')
            plt.savefig(combined_png, dpi=300)
            plt.close()
            print(f"Saved combined ROC/PR/Calibration figure: {combined_png}")

        except Exception as e:
            print("Failed plotting ROC/PR/Calibration:", e)

        # Save individual figures (ROC / PR / Calibration)  (optional)
        try:
            # ROC only
            plt.figure(figsize=(5,5), dpi=300)
            plt.plot(fpr, tpr, lw=3, color=ROC_COLOR)
            plt.plot([0,1],[0,1], linestyle='--', lw=2, color=REFERENCE_COLOR)
            plt.xlim([-0.01,1.01]); plt.ylim([-0.01,1.01])
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC (Held-out)')
            plt.text(0.98, 0.02, auc_text, ha='right', transform=plt.gca().transAxes, fontsize=font_text, bbox=dict(facecolor='white', alpha=0.6))
            plt.grid(True); plt.tight_layout()
            p = os.path.join(output_folder, 'test_roc.pdf'); plt.savefig(p, dpi=300); plt.close()

            # PR only
            plt.figure(figsize=(5,5), dpi=300)
            plt.step(rec_plot, prec_plot, where='post', lw=3, color=PR_COLOR)
            plt.xlim([-0.01,1.01]); plt.ylim([-0.01,1.01])
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])   
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR (Held-out)')
            plt.text(0.02, 0.02, auprc_text, transform=plt.gca().transAxes, fontsize=font_text, bbox=dict(facecolor='white', alpha=0.6))
            plt.grid(True); plt.tight_layout()
            p = os.path.join(output_folder, 'test_pr.pdf'); plt.savefig(p, dpi=300); plt.close()

            # Calibration only
            plt.figure(figsize=(5,5), dpi=300)
            if prob_pred_bins.size > 0:
                plt.plot(prob_pred_bins, prob_true_bins, marker='o', lw=3, color=CALIBRATION_COLOR)
                plt.plot([0,1],[0,1], linestyle='--', lw=2, color=REFERENCE_COLOR)
                plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  
                plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction positives'); plt.title('Calibration (Held-out)')
                plt.text(0.98, 0.02, brier_text, ha='right', transform=plt.gca().transAxes, fontsize=font_text, bbox=dict(facecolor='white', alpha=0.6))
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, "Calibration data unavailable", ha='center', va='center')
            plt.tight_layout()
            p = os.path.join(output_folder, 'test_calibration.pdf'); plt.savefig(p, dpi=300); plt.close()

            print("Saved individual ROC/PR/Calibration plots.")
        except Exception as e:
            print("Failed plotting ROC/PR/Calibration:", e)

    # Save test results summary to csv
    summary = {
        'n_samples': int(len(all_labels)),
        'auroc': float(auroc) if not np.isnan(auroc) else None,
        'auroc_ci_lower': float(auroc_lo) if not np.isnan(auroc_lo) else None,
        'auroc_ci_upper': float(auroc_hi) if not np.isnan(auroc_hi) else None,
        'auprc': float(auprc) if not np.isnan(auprc) else None,
        'auprc_ci_lower': float(auprc_lo) if not np.isnan(auprc_lo) else None,
        'auprc_ci_upper': float(auprc_hi) if not np.isnan(auprc_hi) else None,
        'brier': float(brier) if not np.isnan(brier) else None,
        'brier_ci_lower': float(brier_lo) if not np.isnan(brier_lo) else None,
        'brier_ci_upper': float(brier_hi) if not np.isnan(brier_hi) else None,
        'ece': float(ece) if not np.isnan(ece) else None,
        'weighted_f1': float(weighted_f1) if not np.isnan(weighted_f1) else None,
        'macro_f1': float(macro_f1) if not np.isnan(macro_f1) else None
    }
    summary_csv_path = os.path.join(output_folder, 'test_summary.csv')
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
    print(f"Saved held-out test summary csv: {summary_csv_path}")

    return {
        'output_folder': output_folder,
        'pred_path': pred_path,
        'summary_csv_path': summary_csv_path
    }

# -----------------------------
# Main evaluate model on held-out test set
# ----------------------------- 
def main_test(config):
    """
    Run testing using a fulltrained label encoder classes, k-mer vocabulary, and checkpoint.
    - Loads reference sequence, fulltrained label encoder classes, vocab and checkpoint.
    - Builds test dataset (using fulltrained kmer_to_idx) and replaces classifier with Head_SingleLogit.
    """
    # Read paths from config  
    fulltrained_ckpt_path = config.fulltrained_ckpt_path
    fulltrained_kmer_json = config.fulltrained_kmer_json
    fulltrained_le_classes_path = config.fulltrained_le_classes_path
    finetune_test_fasta = config.finetune_test_fasta
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
   
    # Load fulltrained label encoder classes
    if os.path.exists(fulltrained_le_classes_path):
        with open(fulltrained_le_classes_path, 'r') as f:
            le_classes = json.load(f)
        le = LabelEncoder()
        le.classes_ = np.array(le_classes)
        print("Loaded label encoder classes from:", fulltrained_le_classes_path)
    else:
        le = None
        print("Label encoder classes file not found; dataset will fit its own encoder (risky).")
        # sys.exit(1)

    # Load fulltrained k-mer vocabulary
    try:
        with open(fulltrained_kmer_json, 'r') as f:
            fulltrained_kmer_to_idx = json.load(f)
        print(f"Loaded fulltrained vocab: {len(fulltrained_kmer_to_idx)} k-mers")
    except FileNotFoundError:
        raise FileNotFoundError(f"Fulltrained vocab file not found: {fulltrained_kmer_json}")
    except Exception as e:
        raise RuntimeError(f"Failed loading fulltrained kmer json {fulltrained_kmer_json}: {e}")

    # Build test dataset (use fulltrained kmer_to_idx & label_encoder_classes)
    test_dataset = SequenceDataset(
        fasta_file=finetune_test_fasta,
        kmer_size=config.kmer_size,
        max_length=config.max_seq_length,
        is_train=False,
        kmer_to_idx=fulltrained_kmer_to_idx,
        label_encoder=le,
        compute_class_weights=False,
        reference_sequence=wt_seq,
        label_type=config.label
    )
    print("Loaded test dataset.")
    if len(test_dataset.label_encoder.classes_) != 2:
        raise ValueError("Label encoder does not represent a binary task; check labels and model head.")
    # Print label mapping (index -> original label string)
    print("Label mapping (index -> class_name):", dict(enumerate(test_dataset.label_encoder.classes_)))
    # Count positives/negatives in test set  
    labels_all = np.array(test_dataset.labels)
    n_pos = int((labels_all == 1).sum())
    n_neg = int((labels_all == 0).sum())
    print(f"Held-out dataset size: {len(labels_all)}, pos={n_pos}, neg={n_neg}")
 
    # DataLoader for test dataset
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Initialize model  
    ref_idx = test_dataset.get_reference_indices()
    model = TransformerEncoderModel_Finetune(
        vocab_size=len(fulltrained_kmer_to_idx) + 1,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        num_classes=config.pretrain_num_classes,  # placeholder - will overwrite classifier
        dropout_rate=0.0,
        max_length=config.max_seq_length,
        reference_indices=ref_idx if ref_idx is not None else None
    ).to(config.device)

    # Replace classifier head with a single-logit head for binary BCEWithLogitsLoss
    model.classifier = Head_SingleLogit(
        in_dim=config.embedding_dim, 
        hidden_dim=config.hidden_dim_for_binary_classifier, 
        dropout=config.dropout_rate 
        ).to(config.device)

    # Load fulltrained checkpoint
    if os.path.exists(fulltrained_ckpt_path):
        state = torch.load(fulltrained_ckpt_path, map_location=config.device)
        sd = state.get('model_state_dict', state) if isinstance(state, dict) else state
        try:
            model.load_state_dict(sd, strict=True)
            print("Loaded fulltrained checkpoint (strict=True).")
        except Exception as e:
            model.load_state_dict(sd, strict=False)
            print("Loaded fulltrained checkpoint with strict=False.\n")
            # sys.exit(1)
    else:
        raise FileNotFoundError(f"Fulltrained checkpoint not found: {fulltrained_ckpt_path}")

    # Base output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_folder = f"{config.label}_test_save_{timestamp}"
    os.makedirs(out_folder, exist_ok=True)

    # Run testing
    history = test_model(
        model,
        test_loader,
        config,
        class_names=list(test_dataset.label_encoder.classes_),
        output_folder=out_folder
    )
    print("Testing on held-out set finished. Outputs saved to:", out_folder)
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
    fulltrained_ckpt_path = getattr(config, 'fulltrained_ckpt_path', None)
    fulltrained_kmer_json = getattr(config, 'fulltrained_kmer_json', None)
    fulltrained_le_classes_path = getattr(config, 'fulltrained_le_classes_path', None)
    finetune_test_fasta = getattr(config, 'finetune_test_fasta', None)
    reference_fasta = getattr(config, 'reference_fasta', None)

    # Check paths 
    required_paths = {
        'fulltrained_ckpt_path': fulltrained_ckpt_path,
        'fulltrained_kmer_json': fulltrained_kmer_json,
        'fulltrained_le_classes_path': fulltrained_le_classes_path,
        'finetune_test_fasta': finetune_test_fasta,
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

    main_test(config)

 
 