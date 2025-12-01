from Bio import SeqIO
import hashlib
import os
import subprocess
import logging
import time
from datetime import timedelta
from collections import defaultdict
import random


# Random seed  
SEED = 42

def check_fasta_duplicates_simple(fasta_file):
    """
    Check duplicate IDs and sequences in FASTA files.
    """
    seen_ids = set()
    seen_sequences = set()
    duplicate_ids = False
    duplicate_seqs = False
    
    print(f"Checking FASTA file: {fasta_file}")
    
    # Iterate through the file
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_id = record.id
        seq_str = str(record.seq)
        
        # Check for duplicate IDs
        if seq_id in seen_ids:
            if not duplicate_ids:  
                print(f"Duplicate ID found: {seq_id}")
            duplicate_ids = True
        else:
            seen_ids.add(seq_id)
        
        # Check for duplicate sequences (using MD5 hash to save memory)
        seq_hash = hashlib.md5(seq_str.encode()).hexdigest()
        if seq_hash in seen_sequences:
            if not duplicate_seqs:   
                print(f"Duplicate sequence found (length: {len(seq_str)}): {seq_str[:30]}...")
            duplicate_seqs = True
        else:
            seen_sequences.add(seq_hash)
    
    # Print checking results
    if duplicate_ids:
        print("WARNING: Duplicate IDs detected in the FASTA file")
    else:
        print("\nNo duplicate IDs found")
    
    if duplicate_seqs:
        print("WARNING: Duplicate sequences detected in the FASTA file")
    else:
        print("No duplicate sequences found")

def validate_fasta_sequences(file_path):
    """
    Validate if sequences in the FASTA file contain only 'A', 'U', 'G', 'C'.
    """
    legal_chars = {'A', 'U', 'G', 'C'}
    invalid_seq_idx = []
    
    # Check each character in the sequence
    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq)
        for char in sequence:
            if char.upper() not in legal_chars:
                invalid_seq_idx.append(record.id)
                break  # Break inner loop after finding first invalid character

    # Print checking results
    if invalid_seq_idx:
        print(f"\nThe following sequences contain invalid characters: {', '.join(invalid_seq_idx)}")
        print(f"{len(invalid_seq_idx)} sequences contain invalid characters")
    else:
        print("All sequences are valid")
    
    return invalid_seq_idx
    
def filter_fasta_sequences(input_file, output_file_path):
    """
    Filter out sequences containing non-AUGC characters and save the cleaned sequences.
    """
    # Initialize counters
    total_sequences = 0
    filtered_sequences = 0
    
    # Read and filter sequences
    valid_sequences = []
    for record in SeqIO.parse(input_file, "fasta"):
        total_sequences += 1
        sequence_str = str(record.seq).upper()   
        # sequence_str = str(record.seq)
        if all(char in 'AUGC' for char in sequence_str):
            record.seq = record.seq.upper()
            valid_sequences.append(record)
        else:
            filtered_sequences += 1
    
    # Save valid sequences
    with open(output_file_path, "w") as output_handle:
        SeqIO.write(valid_sequences, output_handle, "fasta")
    
    # Print results
    print(f"\nOriginal sequence count: {total_sequences}")
    print(f"Filtered sequence count: {len(valid_sequences)}")
    print(f"Removed sequences: {filtered_sequences}")
    print(f"Validated sequences saved to: {os.path.abspath(output_file_path)}")

    return valid_sequences

def analyze_seq_length_distribution(input_file_path, len_thresh):
    """
    Analyze length distribution in a set of FASTA sequences.
    """
    # Initialize analysis variables
    min_length = float('inf')
    max_length = 0
    total_sequences = 0
    valid_len_sequences = 0
    length_distribution = {}
    
    # Read and analyze FASTA file
    with open(input_file_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_length = len(record.seq)
            total_sequences += 1        
            # Update min and max lengths
            if seq_length < min_length:
                min_length = seq_length
            if seq_length > max_length:
                max_length = seq_length
            if seq_length <= len_thresh:
                valid_len_sequences+=1
            
            # Calculate length interval (0-100, 100-200, etc.) using floor division to determine interval
            interval_index = seq_length // 100
            interval_start = interval_index * 100
            interval_end = interval_start + 100
            interval_key = f"{interval_start}-{interval_end}"
            
            # Update interval count
            if interval_key in length_distribution:
                length_distribution[interval_key] += 1
            else:
                length_distribution[interval_key] = 1
    
    # Sort distribution by interval start
    sorted_distribution = {}
    for key in sorted(length_distribution.keys(), key=lambda x: int(x.split('-')[0])):
        sorted_distribution[key] = length_distribution[key]
    
    # Print analysis results
    print(f"\nTotal sequences: {total_sequences}")
    print(f"Sequences within {len_thresh} bp: {valid_len_sequences}")
    print(f"Minimum sequence length: {min_length}")
    print(f"Maximum sequence length: {max_length}")
    print("Length distribution:")
    for interval, count in sorted_distribution.items():
        print(f"{interval} bp: {count} sequences") 

def filter_seq_by_length(input_file_path, output_file_path, min_length, max_length):
    """
    Filter sequences in a FASTA file by length thresholds.
    """
    filtered_sequences = [
        record for record in SeqIO.parse(input_file_path, "fasta")
        if min_length <= len(record.seq) <= max_length
    ]
    
    SeqIO.write(filtered_sequences, output_file_path, "fasta")

    print(f"\nSuccessfully retained {len(filtered_sequences)} sequences within length range [{min_length}, {max_length}] bp")

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

def stratified_split_fasta(input_fasta, train_output, test_output, random_seed=42, test_ratio=0.1):
    """
    Performs stratified sampling by RNA type to split into train and test sets.
    """
    random.seed(random_seed)
    
    # Read FASTA file and classify by RNA type
    rna_sequences = defaultdict(list)
    
    for record in SeqIO.parse(input_fasta, "fasta"):
        # Extract RNA type (the part after the space in the ID)
        if ' ' in record.description:
            rna_type = record.description.split(' ', 1)[1].strip()
        else:
            rna_type = 'unknown'
        
        # Store ID and sequence 
        rna_sequences[rna_type].append((record.description, str(record.seq)))
    
    # Report initial counts
    print("\nCounts of each RNA type:")
    for rna_type, sequences in rna_sequences.items():
        print(f"{rna_type}: {len(sequences)}")
    
    # Stratified sampling by category to split into train and test sets
    train_seqs = []
    test_seqs = []
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    
    for rna_type, sequences in rna_sequences.items():
        # Randomly shuffle sequences of the current category
        random.shuffle(sequences)
        
        # Calculate test set size (ensure minimum representation)
        test_size = max(1, int(len(sequences) * test_ratio))  
        
        # Split into train set and test set
        test_set = sequences[:test_size]
        train_set = sequences[test_size:]
        
        # Add to the overall set
        test_seqs.extend(test_set)
        train_seqs.extend(train_set)
        
        # Record counts
        train_counts[rna_type] = len(train_set)
        test_counts[rna_type] = len(test_set)
    
    # Write train set file 
    with open(train_output, 'w') as f:
        for seq_id, sequence in train_seqs:
            f.write(f">{seq_id}\n")
            f.write(f"{sequence}\n")   
    
    # Write test set file 
    with open(test_output, 'w') as f:
        for seq_id, sequence in test_seqs:
            f.write(f">{seq_id}\n")
            f.write(f"{sequence}\n")   
    
    # Report stratified sampling results
    print(f"\nTrain and test set split ratio: {1-test_ratio}:{test_ratio}")
    print("\nCounts of each RNA type in train set:")
    for rna_type, count in train_counts.items():
        print(f"{rna_type}: {count}")
    
    print("\nCounts of each RNA type in test set:")
    for rna_type, count in test_counts.items():
        print(f"{rna_type}: {count}")


if __name__ == "__main__":

    # Path to the raw RNA sequence data file 
    source_fasta_file = "../../data/raw/pretrain/RNA_central_sequences.fasta"
    # Path for validated sequences after filtering non-AUGC characters
    validated_fasta_path = "../../data/preprocessed/pretrain/validated_seqs.fasta"  
    # Length filtering parameters
    len_thresh = 2048  # length distribution analysis
    min_len = 50
    max_len = 2048
    # Path for length-filtered sequences
    length_filtered_fasta_path  = "../../data/preprocessed/pretrain/length_filtered_seqs.fasta"
    # Output path for CD-HIT redundancy removal  
    cdhit_output_path = "../../data/preprocessed/pretrain/remove_redundancy/non_redundancy"
    # Directory for dataset splits (train & val)
    dataset_split_output_dir = "../../data/preprocessed/pretrain/splits/"
    os.makedirs(dataset_split_output_dir, exist_ok=True)
    # Output paths for dataset splits 
    train_output_path = "../../data/preprocessed/pretrain/splits/pretrain_train_set.fasta"
    val_output_path = "../../data/preprocessed/pretrain/splits/pretrain_val_set.fasta"


    # 1. Check duplicate IDs and sequences
    check_fasta_duplicates_simple(source_fasta_file)
    # No duplicate IDs found
    # No duplicate sequences found
 
    # 2. Validate for invalid characters (non-AUGC) in sequences
    invalid_seq_idx = validate_fasta_sequences(source_fasta_file)
    # 1356 sequences contain invalid characters
 
    # 3. Filter out sequences containing non-AUGC characters
    if invalid_seq_idx:
        valid_sequences = filter_fasta_sequences(source_fasta_file, validated_fasta_path)
        # Original sequence count: 60005
        # Filtered sequence count: 58649
        # Removed sequences: 1356

    # 4. Analyze length distribution in filtered sequences
    analyze_seq_length_distribution(validated_fasta_path, len_thresh)
    
    # 5. Filter sequences by length 
    filter_seq_by_length(validated_fasta_path, length_filtered_fasta_path, min_len, max_len)
    # Successfully retained 49048 sequences within length range [50, 2048] bp

    # 6. Remove global sequence redundancy
    start_time = time.time() 
    cdhit_output = run_cdhit_est(
        input_fasta = length_filtered_fasta_path,   
        output_prefix = cdhit_output_path,
        c = 0.85,
        n = 5, 
        M = 64000, 
        T = 32, 
        G = 1
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_delta = timedelta(seconds=elapsed_time)
    print(f"CD-HIT-EST redundancy removal time: {time_delta}")
    # INFO:CD-HIT-EST:Executing CD-HIT command: cd-hit-est -i ../data/preprocessed/pretrain/length_filtered_seqs.fasta -o ../data/preprocessed/pretrain/remove_redundancy/.fasta -c 0.85 -n 5 -M 64000 -T 32 -G 1 -d 0 -g 1
    # INFO:CD-HIT-EST:Redundancy removal completed! Input sequences: 49048, Output sequences: 38712, Redundancy: 21.1%
    # CD-HIT-EST redundancy removal time: 0:05:08.559344

    # 7. Split into train and validation sets​​  
    stats = stratified_split_fasta(
        input_fasta = cdhit_output_path + '.fasta',
        train_output = train_output_path,
        test_output = val_output_path,
        random_seed = SEED,
        test_ratio = 0.1   
    )
    # Counts of each RNA type:
    # lncRNA: 7910
    # snRNA: 7944
    # snoRNA: 7727
    # tRNA: 7120
    # rRNA: 4478
    # miRNA: 3533

    # Train and test set split ratio: 0.9:0.1

    # Counts of each RNA type in train set:
    # lncRNA: 7119
    # snRNA: 7150
    # snoRNA: 6955
    # tRNA: 6408
    # rRNA: 4031
    # miRNA: 3180

    # Counts of each RNA type in test set:
    # lncRNA: 791
    # snRNA: 794
    # snoRNA: 772
    # tRNA: 712
    # rRNA: 447
    # miRNA: 353

