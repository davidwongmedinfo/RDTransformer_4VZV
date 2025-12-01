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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from Bio import SeqIO
from models import SinusoidalPositionalEncoding, TransformerEncoderModel_Pretrain
from utils import save_config
from dataclasses import dataclass, is_dataclass, asdict
import warnings
warnings.filterwarnings('ignore')


# -----------------------------
# Reproducibility  
# -----------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Matplotlib style (global)
# -----------------------------
plt.rcParams.update({
    'figure.dpi': 300,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'font.size': 16,
})

# -----------------------------
# Configuration 
# -----------------------------
@dataclass
class Config:
    # Data / preprocessing
    kmer_size: int = 3
    num_classes: int = 6
    max_seq_length: int = 2048
    # Model architecture
    embedding_dim: int = 64  
    num_heads: int = 8
    num_layers: int = 2
    hidden_dim: int = 192  
    dropout_rate: float = 0.3
    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 8e-5
    num_epochs: int = 500
    use_class_weights: bool = True
    weight_decay: float = 1e-3
    max_grad_norm: float = 1.0
    # LR scheduling
    start_factor: float = 0.1
    end_factor: float = 1.0
    warmup_epochs: int = 7
    lr_min: float = 3e-6
    cosine_cycle_epochs: int = 50
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-3
    # Device  
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # DataLoader workers
    num_workers: int = 0
    # Seed  
    seed: int = SEED

config = Config()

# -----------------------------
# Dataset: Sequence -> k-mer indices
# -----------------------------
class SequenceDataset(Dataset):
    def __init__(self, fasta_file, kmer_size=3, max_length=2048, is_train=True, kmer_to_idx=None, label_encoder=None, compute_class_weights=True):
        """
        A Dataset that reads sequences from a FASTA file and converts them to k-mer indices.
        - Expects the label to be the last token in the FASTA description line by default.
        - Stores k-mer vocabulary when used as a training dataset (is_train=True).
        """
        self.kmer_size = kmer_size
        self.max_length = max_length
        self.sequences = []
        self.labels = []
        self.compute_class_weights = compute_class_weights
        self.class_weights = None
        
        # Read FASTA and collect sequences and labels
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq)
            # Description format: ">sequence_id label"
            label = record.description.split(' ')[-1]
            self.sequences.append(seq)
            self.labels.append(label)
        
        # Fit (on training set) or use provided LabelEncoder
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.labels)
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(self.labels)

        # Compute class weights for training set if requested (optional)
        if is_train and compute_class_weights:
            self.class_weights = self._compute_class_weights()
        
        # Build k-mer vocabulary on training set if training; otherwise use provided mapping
        if is_train:
            self.build_vocab()
        else:
            if kmer_to_idx is None:
                raise ValueError("kmer_to_idx must be provided for validation/test sets")
            self.kmer_to_idx = kmer_to_idx
            self.vocab_size = len(kmer_to_idx) + 1  # +1 for padding index 0
        
        # Precompute k-mer index sequences
        self.kmer_indices = [self.sequence_to_kmers(seq) for seq in self.sequences]
    
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
            kmer = sequence[i:i + self.kmer_size]
            kmers.append(kmer)
        return kmers
    
    def sequence_to_kmers(self, sequence):
        """Convert sequence to fixed-length k-mer index list (padding or truncation)."""
        kmers = self.extract_kmers(sequence)
        indices = [self.kmer_to_idx.get(kmer, 0) for kmer in kmers]  # unknown -> 0 (padding)
        # Padding or truncation to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [0] * (self.max_length - len(indices))
        
        return indices
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Return (kmer_index_tensor, label_tensor)."""
        return torch.tensor(self.kmer_indices[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# -----------------------------
# Training loop  
# -----------------------------
def train_model(model, train_loader, val_loader, criterion_train, criterion_val, optimizer, config, class_names=None):
    """
    Train the model and record metrics. Saves best checkpoint (by validation loss), confusion matrix, and training curves.
    """
    # Create output folder with timestamp
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    output_folder = f"pretrain_save_{current_time}"
    os.makedirs(output_folder, exist_ok=True)

    # Save config and vocabulary (if available)
    save_config(config, output_folder, 'pretrain_config.json')   

    try:
        kmer_file = os.path.join(output_folder, 'pretrained_kmer_to_idx.json')
        with open(kmer_file, 'w') as f:
            json.dump(train_loader.dataset.kmer_to_idx, f)
        print(f"Saved pretrained kmer_to_idx -> {kmer_file}")
    except Exception as e:
        print(f"Warning: failed to save kmer_to_idx: {e}")

    # Bookkeeping
    best_val_loss = float('inf') 
    train_losses, val_losses, lrs = [], [], []
    train_weighted_f1_scores, train_macro_f1_scores, train_auc_scores = [], [], []
    val_weighted_f1_scores, val_macro_f1_scores, val_auc_scores = [], [], []
    best_epoch_preds = best_epoch_labels = best_epoch_probs = None
    best_epoch = -1
    patience_counter = 0

    # Scheduler: convert epoch-based warmup to step-based iter count
    num_batches_per_epoch = len(train_loader)
    total_steps = int(config.cosine_cycle_epochs * num_batches_per_epoch)
    total_warmup_steps = max(1, int(config.warmup_epochs * num_batches_per_epoch))
    if total_steps <= total_warmup_steps:
        total_steps = total_warmup_steps + 1
    remaining_steps = total_steps - total_warmup_steps  

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
    print(f"\nStarting training for {config.num_epochs} epochs, running on {config.device}...\n")
    
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_labels, train_probs = [], [], []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion_train(outputs, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)
            train_probs.extend(probs.detach().cpu().numpy())

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
        train_weighted_f1 = f1_score(train_labels_arr, train_preds_arr, average='weighted')
        train_macro_f1 = f1_score(train_labels_arr, train_preds_arr, average='macro')
        try:
            train_auc = roc_auc_score(train_labels_arr, train_probs_arr, average='weighted', multi_class='ovr')
        except Exception as e:
            print(f"Error calculating training AUC: {e}")
            train_auc = 0.0

        train_weighted_f1_scores.append(train_weighted_f1)
        train_macro_f1_scores.append(train_macro_f1)
        train_auc_scores.append(train_auc)
        
        # ==================== Validation stage ====================
        model.eval()
        val_loss = 0.0
        val_preds, val_labels, val_probs = [], [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                outputs = model(inputs)
                loss = criterion_val(outputs, labels)
                val_loss += loss.item()             
                probs = torch.softmax(outputs, dim=1)  
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
        
        val_loss /= max(1, len(val_loader))
        val_losses.append(val_loss)
        
        # Compute validation metrics
        val_labels_arr = np.array(val_labels)
        val_preds_arr = np.array(val_preds)
        val_probs_arr = np.array(val_probs)
        val_weighted_f1 = f1_score(val_labels_arr, val_preds_arr, average='weighted')
        val_macro_f1 = f1_score(val_labels_arr, val_preds_arr, average='macro')
        
        try:
            val_auc = roc_auc_score(val_labels_arr, val_probs_arr, average='weighted', multi_class='ovr')
        except Exception as e:
            print(f"Error calculating validation AUC: {e}")
            val_auc = 0.0
        
        val_weighted_f1_scores.append(val_weighted_f1)
        val_macro_f1_scores.append(val_macro_f1)
        val_auc_scores.append(val_auc)
        
        # Print epoch summary
        print(f"Epoch {epoch+1:2d}/{config.num_epochs} | "
            f"Trn Loss: {epoch_loss:7.4f} | "
            f"Trn F1(w): {train_weighted_f1:7.4f} | "
            f"Trn F1(m): {train_macro_f1:7.4f} | "
            f"Trn AUC: {train_auc:7.4f}")

        print(f"Epoch {epoch+1:2d}/{config.num_epochs} | "
            f"Val Loss: {val_loss:7.4f} | "
            f"Val F1(w): {val_weighted_f1:7.4f} | "
            f"Val F1(m): {val_macro_f1:7.4f} | "
            f"Val AUC: {val_auc:7.4f}")
        
        # Early stopping logic 
        if math.isnan(val_loss):
            print(f"Epoch {epoch+1}: val_loss is NaN — skipping improvement check, incrementing patience")
            patience_counter += 1
        else:
            previous_best = best_val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_path = os.path.join(output_folder, f'pretrained_best_model_epoch{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss
                    }, best_model_path)
                print(f"Saved checkpoint to {best_model_path} (epoch {epoch+1}, val_loss={best_val_loss:.4f})")

                # Save best epoch predictions/probs for confusion matrix & analysis
                best_epoch_preds = val_preds_arr
                best_epoch_labels = val_labels_arr
                best_epoch_probs = val_probs_arr

                print(f"Epoch {epoch+1}/{config.num_epochs}: [BEST] Val Loss: {best_val_loss:.4f}")

            improve = previous_best - val_loss
            if improve > config.early_stopping_delta:
                patience_counter = 0
                print(f"Epoch {epoch+1}: Significant improvement ({improve:.6f} > {config.early_stopping_delta:.6f}), resetting patience")
            else:
                patience_counter += 1
                print(f"Epoch {epoch+1}: No significant improvement ({improve:.6f} <= {config.early_stopping_delta:.6f}), patience: {patience_counter}/{config.early_stopping_patience}")

        # Check if early stopping is triggered
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs. No improvement for {config.early_stopping_patience} consecutive epochs.")
            break
        print("\n")
    
    # Training finished  
    print(f"\nTraining completed. Best epoch: {best_epoch+1}/{config.num_epochs}, "
          f"Best Val Loss: {best_val_loss:.4f}")
    
    # Confusion matrix for best epoch 
    if best_epoch_preds is not None and best_epoch_labels is not None:
        cm = confusion_matrix(best_epoch_labels, best_epoch_preds)
        plt.figure(figsize=(10, 8))
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(best_epoch_labels)))]  
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Epoch {best_epoch+1})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'confusion_matrix.pdf'))
        plt.close()
        print(f"Confusion matrix saved: 'confusion_matrix.pdf'")

    # Plot training curves (loss, f1, auc, lr)
    epoch_numbers = list(range(1, len(train_losses) + 1))

    # 1. Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_losses, label='Train Loss', linewidth=2.5)
    plt.plot(epoch_numbers, val_losses, label='Val Loss', linewidth=2.5) 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'loss_curve.pdf'))
    plt.close()

    # 2. Weighted F1
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_weighted_f1_scores, label='Train Weighted F1', linewidth=2.5)
    plt.plot(epoch_numbers, val_weighted_f1_scores, label='Val Weighted F1', linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Weighted F1')
    plt.yticks(np.arange(0, 1.1, 0.2))
    # plt.title('Training and Validation Weighted F1')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'f1_curve.pdf'))
    plt.close()

    # 3. Macro F1
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_macro_f1_scores, label='Train Macro F1', linewidth=2.5)
    plt.plot(epoch_numbers, val_macro_f1_scores, label='Val Macro F1', linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1')
    plt.yticks(np.arange(0, 1.1, 0.2))
    # plt.title('Training and Validation Macro F1')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'macro_f1_curve.pdf'))
    plt.close()

    # 4. AUROC
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_numbers, train_auc_scores, label='Train AUROC', linewidth=2.5)
    plt.plot(epoch_numbers, val_auc_scores, label='Val AUROC', linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.yticks(np.arange(0.5, 1.0, 0.1))
    # plt.title('Training and Validation AUROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'auroc_curve.pdf'))
    plt.close()
    print("\nTraining curves saved: 'loss_curve.pdf', 'f1_curve.pdf', 'auroc_curve.pdf'")

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

    # Save training history to CSV
    history_for_df = {
        'epoch': epoch_numbers,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_weighted_f1_scores': train_weighted_f1_scores,
        'val_weighted_f1_scores': val_weighted_f1_scores,
        'train_macro_f1_scores': train_macro_f1_scores,
        'val_macro_f1_scores': val_macro_f1_scores,
        'train_auc_scores': train_auc_scores,
        'val_auc_scores': val_auc_scores,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }

    history_df = pd.DataFrame(history_for_df)
    history_path = os.path.join(output_folder, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"\nTraining history saved to '{history_path}'")

    # Save confusion matrix numeric data
    if 'cm' in locals() and cm is not None:
        confusion_df = pd.DataFrame(cm)
        confusion_path = os.path.join(output_folder, 'confusion_matrix.csv')
        confusion_df.to_csv(confusion_path, index=False)
        print(f"Confusion matrix data saved to '{confusion_path}'")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_weighted_f1_scores': train_weighted_f1_scores,
        'val_weighted_f1_scores': val_weighted_f1_scores,
        'train_macro_f1_scores': train_macro_f1_scores,
        'val_macro_f1_scores': val_macro_f1_scores,
        'train_auc_scores': train_auc_scores,
        'val_auc_scores': val_auc_scores,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'confusion_matrix': cm if 'cm' in locals() else None,  
        'output_folder': output_folder
    }


# -----------------------------
# Main entrypoint
# -----------------------------
def main():
    # Build datasets
    print("\nLoading training data...")
    train_dataset = SequenceDataset(
        fasta_file="../data/preprocessed/pretrain/splits/pretrain_train_set.fasta", 
        kmer_size=config.kmer_size, 
        max_length=config.max_seq_length,
        is_train=True,
        compute_class_weights=config.use_class_weights
    )
    # Print training dataset details
    # print(f"Training set size: {len(train_dataset)}")
    # print(f"Training set vocabulary size: {train_dataset.vocab_size}")
    # train_sample_0, train_label_0 = train_dataset[0]
    # print(f"First training sample: Label: {train_label_0}, "
    #       f"K-mer indices shape: {train_sample_0.shape}, "
    #       f"first 10 k-mer indices: {train_sample_0[:10].tolist()}")
    
    print("Loading validation data...")
    val_dataset = SequenceDataset(
        fasta_file="../data/preprocessed/pretrain/splits/pretrain_val_set.fasta", 
        kmer_size=config.kmer_size, 
        max_length=config.max_seq_length,
        is_train=False,
        kmer_to_idx=train_dataset.kmer_to_idx, 
        label_encoder=train_dataset.label_encoder   
    )
    # Print validation dataset details
    # print(f"Validation set size: {len(val_dataset)}")
    # print(f"Using same vocabulary as training set: {val_dataset.vocab_size}")
    # val_sample_0, val_label_0 = val_dataset[0]
    # print(f"First validation sample: Label: {val_label_0}, "
    #       f"K-mer indices shape: {val_sample_0.shape}, "
    #       f"first 10 k-mer indices: {val_sample_0[:10].tolist()}")
    
    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Align config.num_classes with encoder if needed
    encoder_num_classes = len(train_dataset.label_encoder.classes_)
    if encoder_num_classes != config.num_classes:
        print(f"Warning: config.num_classes ({config.num_classes}) != label encoder classes ({encoder_num_classes}), using encoder value.")
        config.num_classes = encoder_num_classes
    
    # Initialize model
    model = TransformerEncoderModel_Pretrain(
        vocab_size=train_dataset.vocab_size,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
        max_length=config.max_seq_length
    ).to(config.device)
    print(model)

    # Print parameter summary
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    print("\n--- Model parameter summary ---")
    print(f"Total params      : {total:,}")
    print(f"Trainable params  : {trainable:,}")
    print(f"Non-trainable     : {non_trainable:,}")
    print("--------------------------------")

    # Loss with optional class weights
    if config.use_class_weights and train_dataset.class_weights is not None:
        weights = [train_dataset.class_weights.get(i, 1.0) for i in range(config.num_classes)]
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(config.device)
        criterion_train = nn.CrossEntropyLoss(weight=weights_tensor)   
        print("\nUsing weighted CrossEntropyLoss for class imbalance correction")
        print("Class weights applied:")
        for class_idx, weight in enumerate(weights):
            print(f"  Class {class_idx}: {weight:.4f}")
    else:
        criterion_train = nn.CrossEntropyLoss()
        print("\nUsing standard CrossEntropyLoss (no class weighting applied)")

    criterion_val = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Train
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion_train,
        criterion_val, 
        optimizer, 
        config,
        class_names=list(train_dataset.label_encoder.classes_) 
    )
    
    print(f"Training completed! All results saved in: {history['output_folder']}")


if __name__ == "__main__":
    main()