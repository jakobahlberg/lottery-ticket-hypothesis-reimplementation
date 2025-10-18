# Lottery Ticket Hypothesis - Implementation

This repository contains a complete implementation of the Lottery Ticket Hypothesis experiment from Frankle and Carbin (2018) using TensorFlow 2.x, designed to run in Google Colab.

## Authors
- Johan Nielsen
- Lasse Abildhauge Christensen  
- Jakob Ahlberg

## Overview

The Lottery Ticket Hypothesis proposes that dense neural networks contain smaller sub-networks (winning tickets) that, when trained in isolation, can achieve comparable performance to the original network. This implementation tests this hypothesis through iterative pruning experiments on MNIST classification.

## Key Features

- **Complete Implementation**: Full recreation of the original lottery ticket algorithm
- **TensorFlow 2.x Compatible**: Updated for modern TensorFlow while maintaining original algorithm fidelity
- **Comprehensive Experiments**: Main experiment plus extensive ablation studies
- **Visualization**: Multiple plotting functions for analyzing results
- **Carbon Tracking**: Environmental impact monitoring during training

## Quick Start

### Prerequisites
- Google Colab environment (recommended)
- Python 3.x
- Internet connection for dependency installation

### Installation & Setup

1. **Open in Colab**: Upload the `lottery_ticket_hypothesis_colab.ipynb` notebook to Google Colab

2. **Install Dependencies**: Run the first cell to install required packages:
   ```python
   # This installs TensorFlow 2.19.0, Keras, NumPy, scikit-learn, matplotlib, and carbontracker
   ```

3. **Data Preparation**: The notebook automatically downloads and prepares MNIST data (55k train / 5k validation / 10k test)

### Running Experiments

#### Main Experiment
Execute the main lottery ticket experiment:
```python
# This runs 3 complete experiments with 25 pruning iterations each
# Each experiment takes approximately 15-20 minutes on Colab GPU
```

#### Ablation Studies
Run comprehensive ablation studies to test different configurations:
- **Pruning Types**: Unstructured vs Structured pruning
- **Optimizers**: Adam vs SGD comparison  
- **Pruning Aggressiveness**: Different pruning rates
- **Early Stopping**: Validation accuracy vs loss
- **Pruning Scope**: Local vs Global pruning


### Network Architecture

- **Input**: 784 features (28×28 MNIST pixels)
- **Hidden Layers**: 300 → 100 neurons
- **Output**: 10 classes (digits 0-9)
- **Total Parameters**: ~266,610 weights

### Default Configuration

- **Training**: 50,000 iterations per pruning round
- **Pruning Rates**: Layer 0 (20%), Layer 1 (20%), Layer 2 (10%)
- **Optimizer**: Adam with learning rate 1.2e-3
- **Batch Size**: 60
- **Early Stopping**: Validation loss-based

## Output Files

- `experiment_main_results.pkl`: Complete results from main experiments
- `ablation_results.pkl`: Results from all ablation studies


## Citation

If you use this implementation, please cite the original paper:
```
Frankle, J., & Carbin, M. (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635.
```
