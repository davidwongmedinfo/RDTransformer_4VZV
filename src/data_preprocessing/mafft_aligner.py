import tempfile
import subprocess
import shutil
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def run_mafft_and_write(wt_seq, mutant_seqs, out_fasta="mafft_aligned.fasta", mafft_exe="mafft", mafft_args=None, include_wt_in_output=False):
    """
    RNA multiple sequence alignment tool: Aligns wild-type and mutant sequences using MAFFT
    
    Core functionality:
    - Integrates WT and mutant sequence inputs (supports files/lists/strings)
    - Preprocesses sequences with U→T conversion to prevent MAFFT errors
    - Executes MAFFT for multiple sequence alignment
    - Restores T→U and outputs single-line FASTA
    
    Parameters:
    wt_seq: str                   Path to WT sequence FASTA file
    mutant_seqs: str/list         Mutant sequences (file path/list of strings/single string)
    out_fasta: str                Output file path (default: mafft_aligned.fasta)
    mafft_exe: str                MAFFT executable path (default: mafft)
    mafft_args: list              MAFFT command arguments (default: ["--auto", "--thread", "-1"])
    include_wt_in_output: bool    Whether to include wild-type sequences in output (default: False)
    
    Returns:
    str: Path to aligned FASTA file
    """
    print("\nStarting MAFFT alignment process...")

    # Set default MAFFT arguments if none provided
    if mafft_args is None:
        mafft_args = ["--auto", "--thread", "-1"]
        print("Using default MAFFT arguments: --auto --thread -1")
    else:
        print(f"Using custom MAFFT arguments: {' '.join(mafft_args)}")

    # Validate WT sequence file exists
    if not (isinstance(wt_seq, str) and os.path.isfile(wt_seq)):
        raise ValueError("wt_seq must be a valid FASTA file path")

    # Create temporary working directory
    workdir = tempfile.mkdtemp(prefix="mafft_tmp_")
    try:
        input_path = os.path.join(workdir, "input.fasta")
        aligned_path = os.path.join(workdir, "aligned.fasta")

        # --------------------
        # 1) Read WT FASTA records
        # --------------------
        print(f"Reading WT sequences from: {wt_seq}")
        records = list(SeqIO.parse(wt_seq, "fasta"))
        if len(records) == 0:
            raise ValueError(f"WT file {wt_seq} contains no records")
        print(f"Found {len(records)} WT sequence(s)")
        wt_ids = {rec.id for rec in records}

        # --------------------
        # 2) Append mutant sequences
        # --------------------
        if isinstance(mutant_seqs, str) and os.path.isfile(mutant_seqs):
            print(f"Reading mutant sequences from file: {mutant_seqs}")
            # Mutant sequences provided as FASTA file
            records.extend(list(SeqIO.parse(mutant_seqs, "fasta")))
        elif isinstance(mutant_seqs, (list, tuple)):
            # Mutant sequences provided as list of strings
            print(f"Processing {len(mutant_seqs)} mutant sequence(s) from list input")
            for i, s in enumerate(mutant_seqs, start=1):
                records.append(SeqRecord(Seq(s), id=f"mutant_{i}", description=""))
        elif isinstance(mutant_seqs, str):
            # Single mutant sequence string
            print("Processing single mutant sequence from string input")
            records.append(SeqRecord(Seq(mutant_seqs), id="mutant_1", description=""))
        else:
            # No mutants provided - proceed with WT only
            print("No mutant sequences provided - aligning WT only")
    
        total_seqs = len(records)
        if total_seqs == 0:
            raise ValueError("No sequence records available (WT + mutants)")
        print(f"Total sequences to align: {total_seqs}")

        # --------------------
        # 3) Write MAFFT input: Temporary U→T conversion
        # --------------------
        records_to_write = []
        for r in records:
            seq_s = str(r.seq).upper().replace("U", "T")   # Temporary conversion
            records_to_write.append(SeqRecord(Seq(seq_s), id=r.id, description=r.description))

        with open(input_path, "w") as fh:
            SeqIO.write(records_to_write, fh, "fasta")

        # --------------------
        # 4) Execute MAFFT alignment
        # --------------------
        cmd = [mafft_exe] + mafft_args + [input_path]
        print(f"Executing MAFFT command:\n{' '.join(cmd)}")

        with open(aligned_path, "w") as outfh:
            proc = subprocess.run(cmd, stdout=outfh, stderr=subprocess.PIPE, text=True)

        if proc.returncode != 0:
            raise RuntimeError(f"MAFFT failed (rc={proc.returncode}). STDERR:\n{proc.stderr}")
        print("MAFFT alignment completed successfully")

        # --------------------
        # 5) Process MAFFT output: Restore T→U and write to FASTA
        # --------------------
        aligned_recs = list(SeqIO.parse(aligned_path, "fasta"))

        # Filter out WT sequences if requested
        if not include_wt_in_output:
            filtered_recs = [rec for rec in aligned_recs if rec.id not in wt_ids]
            print(f"Filtered out {len(aligned_recs) - len(filtered_recs)} WT sequences")
            aligned_recs = filtered_recs

        # Restore T→U conversion
        for r in aligned_recs:
            r.seq = Seq(str(r.seq).upper().replace("T", "U"))

        # Write non-wrapped FASTA  
        if include_wt_in_output:
            print(f"Writing final alignment to: {out_fasta}, with WT sequence")
        else:
            print(f"Writing final alignment to: {out_fasta}, without WT sequence")
        with open(out_fasta, "w") as fh:
            for r in aligned_recs:
                seq_str = str(r.seq)
                fh.write(f">{r.id}\n")
                fh.write(seq_str + "\n")

        print(f"Alignment complete!")
        return out_fasta

    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass

def validate_alignment(fasta_path):
    """
    Check if all aligned sequences have the same length.
    """
    if not os.path.isfile(fasta_path):
        raise ValueError("The specified FASTA file does not exist: " + fasta_path)
    
    lengths = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq_len = len(str(rec.seq))
        # print(f"{rec.id}\t{seq_len}")
        lengths.append(seq_len)
    
    if all(len == lengths[0] for len in lengths):
        print(f"\nVerify alignment results: All sequences have the same length of {lengths[0]}")
    else:
        print("\nVerify alignment results: Sequences have varying lengths")

