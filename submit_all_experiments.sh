#!/bin/bash
#SBATCH --job-name=bo_experiments
#SBATCH --output=logs/bo_experiments_%A_%a.out
#SBATCH --error=logs/bo_experiments_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-59

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .bo_project_env/bin/activate

# Define experiment configurations
DATASETS=("dft_descriptors.csv" "dft_chemberta2.csv" "dft_mordred.csv")
METHODS=("vanilla" "fabo" "pca" "pls" "opls")
KERNELS=("matern" "rbf")
ACQUISITIONS=("ei" "ucb")

# Calculate total number of experiments
# 3 datasets x 5 methods x 2 kernels x 2 acquisitions = 60 experiments

# Parse SLURM_ARRAY_TASK_ID to get dataset, method, kernel, and acquisition
TASK_ID=$SLURM_ARRAY_TASK_ID

# Calculate indices
NUM_DATASETS=${#DATASETS[@]}
NUM_METHODS=${#METHODS[@]}
NUM_KERNELS=${#KERNELS[@]}
NUM_ACQUISITIONS=${#ACQUISITIONS[@]}

# Decompose task ID
ACQ_IDX=$((TASK_ID % NUM_ACQUISITIONS))
TASK_ID=$((TASK_ID / NUM_ACQUISITIONS))

KERNEL_IDX=$((TASK_ID % NUM_KERNELS))
TASK_ID=$((TASK_ID / NUM_KERNELS))

METHOD_IDX=$((TASK_ID % NUM_METHODS))
DATASET_IDX=$((TASK_ID / NUM_METHODS))

# Get actual values
DATASET=${DATASETS[$DATASET_IDX]}
METHOD=${METHODS[$METHOD_IDX]}
KERNEL=${KERNELS[$KERNEL_IDX]}
ACQUISITION=${ACQUISITIONS[$ACQ_IDX]}

# Extract dataset name without extension
DATASET_NAME=$(basename "$DATASET" .csv)

# Set output directory
OUTPUT_DIR="results/${DATASET_NAME}/${METHOD}_${KERNEL}_${ACQUISITION}"

echo "=========================================="
echo "Running Experiment:"
echo "  Dataset: $DATASET"
echo "  Method: $METHOD"
echo "  Kernel: $KERNEL"
echo "  Acquisition: $ACQUISITION"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

# Run the experiment
python -m src.cli \
    --mode "$METHOD" \
    --input "data/$DATASET" \
    --cache "data/dft_G.json" \
    --output-dir "$OUTPUT_DIR" \
    --n-initial 10 \
    --n-iter 100 \
    --seed 42 \
    --kernels "$KERNEL" \
    --acquisitions "$ACQUISITION"

echo "Experiment completed!"

