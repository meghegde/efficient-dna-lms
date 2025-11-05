from datasets import load_dataset
import random

# ----------------------------
# Step 1: Load the genome dataset
# ----------------------------
dataset = load_dataset("InstaDeepAI/human_reference_genome", split="train")

# ----------------------------
# Step 2: Parameters
# ----------------------------
WINDOW_SIZE = 1024
OVERLAP = 0
SHUFFLE = True
OUTPUT_FILE = f"./data/processed/hg38_sequences_len_{WINDOW_SIZE}.txt"

# ----------------------------
# Step 3: Function to break chromosome into windows
# ----------------------------
def split_into_windows(example):
    seq = example["sequence"]
    subsequences = []
    for start in range(0, len(seq) - WINDOW_SIZE + 1, WINDOW_SIZE - OVERLAP):
        subseq = seq[start:start + WINDOW_SIZE]
        subsequences.append(subseq)
    return subsequences

# ----------------------------
# Step 4: Extract sequences
# ----------------------------
all_sequences = []
for chrom in dataset:
    chrom_sequences = split_into_windows(chrom)
    all_sequences.extend(chrom_sequences)

# ----------------------------
# Step 5: Optional shuffle
# ----------------------------
if SHUFFLE:
    random.shuffle(all_sequences)

# ----------------------------
# Step 6: Save to text file
# ----------------------------
with open(OUTPUT_FILE, "w") as f:
    for seq in all_sequences:
        f.write(seq + "\n")

print(f"Saved {len(all_sequences)} sequences to {OUTPUT_FILE}")

