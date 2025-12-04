# Copyright (c) 2025 Beijing Youcare Kechuang Pharmaceutical Technology Co. Ltd., All rights reserved.
# Author: Dawei Wang
# This code may not be used, modified, or distributed without prior written consent from Beijing Youcare Kechuang Pharmaceutical Technology Co. Ltd.

import pandas as pd
import re
import time
import subprocess
from datetime import timedelta
import logging
import os
import sys
from collections import defaultdict
import random
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from mafft_aligner import run_mafft_and_write, validate_alignment


# Random seed  
SEED = 42

def check_sequence_characters(df):
    """
    Check if sequences in the 'Sequence' column contain only AUGC characters.
    """
    # Validate input columns
    if 'Sequence' not in df.columns or 'ID' not in df.columns:
        raise ValueError("DataFrame must contain both 'Sequence' and 'ID' columns")
    
    # Define the set of valid characters
    valid_chars = {'A', 'U', 'G', 'C'}
    invalid_entries = []
    
    # Process each row
    for index, row in df.iterrows():
        seq = row['Sequence']
        seq_id = row['ID']
        
        # Remove whitespace (including spaces, tabs, etc.)
        if isinstance(seq, str):
            cleaned_seq = re.sub(r'\s+', '', seq)
            df.at[index, 'Sequence'] = cleaned_seq
        else:
            cleaned_seq = seq
        
        # Skip non-string types
        if not isinstance(cleaned_seq, str):
            invalid_chars = {f"Non-string type ({type(cleaned_seq).__name__})"}
            invalid_entries.append((seq_id, invalid_chars))
            continue
        
        # Check each character
        invalid_chars = set()
        for char in cleaned_seq.upper():  # Convert to uppercase for checking
            if char not in valid_chars:
                invalid_chars.add(char)
        
        if invalid_chars:
            invalid_entries.append((seq_id, invalid_chars))
    
    # Output results
    if not invalid_entries:
        print("\nAll sequences contain only AUGC characters.")
    else:
        print("\nThe following sequences contain invalid characters:")
        for seq_id, invalid_chars in invalid_entries:
            sorted_chars = sorted(invalid_chars)  # Sort invalid characters
            chars_str = ", ".join(sorted_chars)
            print(f"ID: {seq_id}, Invalid characters: {chars_str}")
    
    return df


def process_dataframe(df, wb_thrd, elisa_thrd):
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Calculate sequence length and store in new column
    result_df['Seq_len'] = result_df['Sequence'].str.len()
    
    # Generate binary labels based on wb_thrd threshold
    result_df['Label_wb'] = (result_df['Exp_value(WB)'] >= wb_thrd).astype(int)
    # result_df['Label_wb'] = (result_df['WB'] >= wb_thrd).astype(int)
    
    # Generate binary labels based on elisa_thrd threshold
    result_df['Label_elisa'] = (result_df['Exp_value(ELISA)'] >= elisa_thrd).astype(int)
    # result_df['Label_elisa'] = (result_df['IgG'] >= elisa_thrd).astype(int)
    
    # Create combined ID with both wb and elisa labels
    result_df['ID_with_Label'] = (result_df['ID'].astype(str) + '|' + 
                                  result_df['Label_wb'].astype(str) + '|' + 
                                  result_df['Label_elisa'].astype(str))

    # Calculate and print label distribution statistics
    wb_counts = result_df['Label_wb'].value_counts()
    elisa_counts = result_df['Label_elisa'].value_counts()
    print("\nLabel_wb positive/negative sample count ratio:")
    print(f"Positive(1):Negative(0) = {wb_counts.get(1, 0)} : {wb_counts.get(0, 0)}")
    print("Label_elisa positive/negative sample count ratio:")
    print(f"Positive(1):Negative(0) = {elisa_counts.get(1, 0)} : {elisa_counts.get(0, 0)}")

    return result_df

def write_fasta(df, output_file):
    # Write FASTA format: sequence written as single line without extra newline
    with open(output_file, 'w') as f:
        for _, row in df.iterrows():
            seq_id = str(row['ID_with_Label'])
            sequence = str(row['Sequence'])
            f.write(f">{seq_id}\n")
            f.write(f"{sequence}\n")   
    print(f'\nFasta file successfully written to {output_file}')

def run_cdhit_est(input_fasta, output_prefix, c, n, M=16000, T=16, G=1):
    """
    Remove sequence redundancy using CD-HIT-EST
    
    Parameters:
    input_fasta: Path to input FASTA file
    output_prefix: Output file prefix (without extension)
    c: Sequence similarity threshold (0-1, default 0.9)
    n: Word length (default 5)
    M: Maximum memory in MB (default 16000=16GB)
    T: Number of CPU cores (default 4)
    G: Use global alignment (default 1=yes)
    
    Returns:
    Path to output FASTA file
    """
    
    # Build command
    cmd = [
        "cd-hit-est",
        "-i", input_fasta,
        "-o", f"{output_prefix}.fasta",
        "-c", str(c),
        "-n", str(n),
        "-M", str(M),
        "-T", str(T),
        "-G", str(G),
        "-d", "0", 
    ]

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"\nExecuting CD-HIT-EST command: {' '.join(cmd)}")
    
    # Add extra parameter
    cmd.extend(["-g", "1"])
    
    # Run CD-HIT
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check output file
        output_file = f"{output_prefix}.fasta"
        if not os.path.exists(output_file):
            logger.error(f"Output file not created: {output_file}")
            raise RuntimeError("CD-HIT failed to generate output file")
        
        # Count results
        input_count = sum(1 for _ in open(input_fasta) if _.startswith(">"))
        output_count = sum(1 for _ in open(output_file) if _.startswith(">"))
        reduction = (1 - output_count/input_count) * 100
        logger.info(f"Redundancy removal completed! Input sequences: {input_count}, Output sequences: {output_count}, Redundancy: {reduction:.1f}%")
        
        return output_file
        
    except subprocess.CalledProcessError as e:
        logger.error(f"CD-HIT execution failed, exit code: {e.returncode}")
        logger.error(f"Error message:\n{e.stderr}")
        raise RuntimeError(f"CD-HIT execution failed: {e.stderr}")

def stratified_split_fasta(input_fasta, trainval_output, test_output, random_seed=SEED, test_ratio=0.2, label_type=None):
    """
    Performs stratified sampling by label to split data into training/validation and test sets.
    label_type: Type of label to extract from the description, either 'wb' or 'elisa'
    """
    rng = random.Random(random_seed)
    np.random.seed(random_seed)

    # Read FASTA file using Biopython and group sequences by label
    mrna_sequences = defaultdict(list)
    for record in SeqIO.parse(input_fasta, "fasta"):
        label = 'unknown'
        # Extract label (e.g., mRNA-033|0|1) - first part is wb label, second is elisa label
        if '|' in record.description and label_type is not None:
            parts = record.description.split('|')
            if label_type == 'wb':
                label = parts[1].strip() if len(parts) >= 2 else 'unknown'
                # print(record.description, label)
            elif label_type == 'elisa':
                label = parts[2].strip() if len(parts) >= 3 else 'unknown'
                # print(record.description, label)
            
        # Store sequence ID and content
        mrna_sequences[label].append((record.description, str(record.seq)))

    # Print counts per label
    print("Count statistics per label:")
    for rna_label, sequences in mrna_sequences.items():
        print(f"{rna_label}: {len(sequences)}")

   # Perform stratified sampling by label to split into train/val and test sets
    trainval_seqs = []
    test_seqs = []
    trainval_counts = defaultdict(int)
    test_counts = defaultdict(int)
    
    for rna_label, sequences in mrna_sequences.items():
        # Shuffle sequences for the current label
        rng.shuffle(sequences)
        # Calculate test set size
        test_size = max(1, int(len(sequences) * test_ratio))  # Ensure at least one sample per class in test set
        # Split into test set and train/val set
        test_set = sequences[:test_size]
        trainval_set = sequences[test_size:]
        # Add to global sets
        test_seqs.extend(test_set)
        trainval_seqs.extend(trainval_set)
        # Record counts
        test_counts[rna_label] = len(test_set)
        trainval_counts[rna_label] = len(trainval_set)

    # Shuffle the global sets
    rng.shuffle(trainval_seqs)
    rng.shuffle(test_seqs)
    
    # Write train/val set file (no extra newlines)
    write_fasta_file(trainval_seqs, trainval_output)
    write_fasta_file(test_seqs, test_output)  
    
    # Report label counts in train/val and test sets
    print("Label counts in training/validation set:")
    for rna_label, count in trainval_counts.items():
        print(f"{rna_label}: {count}")

    print("Label counts in test set:")
    for rna_label, count in test_counts.items():
        print(f"{rna_label}: {count}")

def write_fasta_file(records, filename):
    with open(filename, 'w') as f:
        for desc, seq in records:
            f.write(f">{desc}\n{seq}\n")


if __name__ == "__main__":

    # Path to the original finetune data
    original_excel_path = "../../data/raw/finetune/515条微调序列20251030.xlsx"
    # Output path for the preprocessed DataFrame  
    preprocessed_dataframe_output_path = "../../data/preprocessed/finetune/finetune_data.csv"
    # Output FASTA file path for CD-HIT redundancy removal
    fasta_file_path_for_CDHIT = "../../data/preprocessed/finetune/finetune_seqs_4cdhit.fasta"
    # Output path for CD-HIT redundancy removal results
    cdhit_output_path = "../../data/preprocessed/finetune/remove_redundancy"
    # Path to the reference sequence (wild-type) for alignment
    reference_sequence_path = "../../data/raw/finetune/VZV-P01.fasta"
    # Output path for sequence alignment
    alignment_output_path = "../../data/preprocessed/finetune/aligned_sequences.fasta"
    # Directory for storing WB label-based dataset splits
    wb_splits_dir = "../../data/preprocessed/finetune/wb_splits/"
    os.makedirs(wb_splits_dir, exist_ok=True)
    # Directory for storing ELISA label-based dataset splits
    elisa_splits_dir = "../../data/preprocessed/finetune/elisa_splits/"
    os.makedirs(elisa_splits_dir, exist_ok=True)
 

    # 1. Read original excel file
    df = pd.read_excel(original_excel_path)
    # print(df.head())

    # 2. Remove wild-type sequence rows
    # print(len(df))
    df = df[df["ID"] != "VZV-P01"]
    # print(len(df))

    # 3. Check for invalid characters
    check_sequence_characters(df)
    # All sequences contain only AUGC characters.

    # 4. Generate columns of sequence length, wb label, elisa label, and new IDs with labels
    result_df = process_dataframe(df, wb_thrd=1, elisa_thrd=500)
    # print(result_df.head())
    # Label_wb positive/negative sample count ratio:
    # Positive(1):Negative(0) = 266 : 248
    # Label_elisa positive/negative sample count ratio:
    # Positive(1):Negative(0) = 267 : 247
    result_df.to_csv(preprocessed_dataframe_output_path, index=False)

    # Write to FASTA file
    write_fasta(result_df, fasta_file_path_for_CDHIT)

    # 5. Remove global sequence redundancy
    start_time = time.time() 
    cdhit_output = run_cdhit_est(
        input_fasta = fasta_file_path_for_CDHIT,   
        output_prefix = cdhit_output_path,
        c = 1.0,
        n = 5, 
        M = 64000, 
        T = 32, 
        G = 1
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_delta = timedelta(seconds=elapsed_time)
    print(f"CD-HIT-EST redundancy removal time: {time_delta}")
    # Fasta file successfully written to ../../data/preprocessed/finetune/finetune_seqs_4cdhit.fasta
    # INFO:
    # Executing CD-HIT-EST command: cd-hit-est -i ../../data/preprocessed/finetune/finetune_seqs_4cdhit.fasta -o ../../data/preprocessed/finetune/remove_redundancy.fasta -c 1.0 -n 5 -M 64000 -T 32 -G 1 -d 0:Redundancy removal completed! Input sequences: 514, Output sequences: 514, Redundancy: 0.0%
    # CD-HIT-EST redundancy removal time: 0:00:00.340283

    # 6. Align wild-type sequences to wild-type sequence
    run_mafft_and_write(
        wt_seq = reference_sequence_path,
        mutant_seqs = cdhit_output_path + '.fasta',
        out_fasta = alignment_output_path,
        mafft_exe = "mafft",
        mafft_args = ["--auto", "--thread", "-1"],
        # mafft_args=["--localpair", "--maxiterate", "1000", "--thread", "-1"]
        include_wt_in_output=False
    )
    # Verify alignment results by checking sequence lengths
    validate_alignment(alignment_output_path)
    # Verify alignment results: All sequences have the same length of 1872

    # 7. Split into training and validation sets​​ by WB label
    print('\n==============')
    stratified_split_fasta(
        input_fasta = alignment_output_path,
        trainval_output = os.path.join(wb_splits_dir, 'wb_finetune_trainval_set.fasta'),
        test_output = os.path.join(wb_splits_dir, 'wb_finetune_test_set.fasta'),
        random_seed = SEED,
        test_ratio = 0.2,  
        label_type = 'wb'
    )
    # ==============
    # Count statistics per label:
    # 1: 266
    # 0: 248
    # Label counts in training/validation set:
    # 1: 213
    # 0: 199
    # Label counts in test set:
    # 1: 53
    # 0: 49

    # Split into training and validation sets​​ by ELISA label
    print('\n==============')
    stratified_split_fasta(
        input_fasta = alignment_output_path,
        trainval_output = os.path.join(elisa_splits_dir, 'elisa_finetune_trainval_set.fasta'),
        test_output = os.path.join(elisa_splits_dir, 'elisa_finetune_test_set.fasta'),
        random_seed=SEED,
        test_ratio=0.2,   
        label_type='elisa'
    )
    # ==============
    # Count statistics per label:
    # 1: 267
    # 0: 247
    # Label counts in training/validation set:
    # 1: 214
    # 0: 198
    # Label counts in test set:
    # 1: 53
    # 0: 49
    
