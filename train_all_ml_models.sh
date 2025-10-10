#!/bin/bash
# Train ML models on all three datasets and create comparison plots

echo "=============================================="
echo "Training ML Models on All Datasets"
echo "=============================================="

# Activate virtual environment
source .bo_project_env/bin/activate

# Create output directory
mkdir -p ml_results

# Train on Mordred dataset
echo ""
echo "🔬 Training on Mordred dataset..."
python -m ml_models.train_models \
    --input-csv data/dft_mordred.csv \
    --cache-path data/dft_G.json \
    --output-dir ml_results/mordred \
    --test-size 0.2 \
    --random-state 42

# Train on ChemBERTa2 dataset
echo ""
echo "🔬 Training on ChemBERTa2 dataset..."
python -m ml_models.train_models \
    --input-csv data/dft_chemberta2.csv \
    --cache-path data/dft_G.json \
    --output-dir ml_results/chemberta2 \
    --test-size 0.2 \
    --random-state 42

# Train on DFT descriptors dataset
echo ""
echo "🔬 Training on DFT descriptors dataset..."
python -m ml_models.train_models \
    --input-csv data/dft_descriptors.csv \
    --cache-path data/dft_G.json \
    --output-dir ml_results/dft \
    --test-size 0.2 \
    --random-state 42

# Create comparison plots
echo ""
echo "📊 Creating comparison plots..."
python ml_models/plot_model_results.py \
    --results-dir ml_results \
    --output-dir ml_plots

echo ""
echo "=============================================="
echo "✅ All models trained and plots created!"
echo "Results: ml_results/"
echo "Plots: ml_plots/"
echo "=============================================="

