# Federated Learning with Parameter-Efficient Fine-Tuning

This repository contains the implementation for the research project on federated fine-tuning of foundation AI models using lightweight computing approaches. The code provides implementations of various parameter-efficient fine-tuning (PEFT) methods combined with different federated learning algorithms.

## Overview

The project investigates the intersection of parameter-efficient fine-tuning methods and federated learning algorithms to address the computational, privacy, and performance challenges in deploying foundation AI models in resource-constrained environments. The implementation demonstrates how different combinations of PEFT methods and federated learning algorithms perform on NLP (IMDB sentiment analysis) and computer vision (CIFAR10 binary classification) tasks.

## PEFT Methods Implemented

- **LoRA (Low-Rank Adaptation)**: Implements efficient fine-tuning by adding pairs of rank decomposition matrices to existing weights
- **P-Tuning**: Adds continuous prompts that are optimized during training
- **Prefix Tuning**: Prepends trainable continuous vectors to each transformer layer
- **Discrete Prompt Tuning**: Uses fixed natural language prompts prepended to inputs

## Federated Learning Algorithms

- **FedAvg**: Standard federated averaging algorithm
- **FedProx**: Adds regularization to limit local updates from diverging
- **FedOpt**: Implements adaptive optimization at the server level
- **FedDyn**: Uses dynamic regularization based on parameter trajectory
- **SCAFFOLD**: Employs control variates to correct client drift
- **MOON**: Implements model-contrastive federated learning
- **FedNova**: Normalizes averaging based on client training iterations

## Project Structure

- `fedVer5_hardprompt.py`: Implementation of discrete prompt tuning for IMDB
- `fedVer5_lora.py`: LoRA implementation for IMDB
- `fedVer5_prefix.py`: Prefix tuning implementation for IMDB
- `fedVer5_ptuning.py`: P-tuning implementation for IMDB
- `fedVer5_catdog_hardprompt.py`: Discrete prompt tuning for CIFAR10 cat-dog classification
- `fedVer5_catdog_lora.py`: LoRA for CIFAR10 cat-dog classification
- `fedVer5_catdog_prefix.py`: Prefix tuning for CIFAR10 cat-dog classification
- `fedVer5_catdog_ptuning.py`: P-tuning for CIFAR10 cat-dog classification

## Usage

1. Install required dependencies:
   ```bash
   pip install torch torchvision transformers datasets numpy pandas
   ```

2. To run an experiment with one of the implementations:
   ```bash
   python fedVer5_lora.py  # Example for LoRA on IMDB
   ```

3. Each script includes an experiment repeater that runs the experiment multiple times with different seeds for statistical validation.

## Results

The experiments show:
- LoRA achieves superior performance on NLP tasks (84.87% accuracy with SCAFFOLD)
- P-Tuning excels in computer vision tasks (65.39% accuracy with FedAvg)
- Selective combinations reduce communication overhead by up to three orders of magnitude while maintaining competitive accuracy

## Contact

For questions or feedback, please contact the repository owner.
