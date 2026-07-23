# Neural-Network-Architecture

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains the deep learning models designed to reconstruct complete, high-fidelity spatio-temporal spectrum maps from noisy, sparse RF measurements. Developed for UCF Senior Design, the architecture acts as the intelligence layer for an autonomous UAS/UGV navigating environments where exhaustive RF sampling is strategically or physically impossible.

## Dataset Generation
Due to the scarcity of classified adversarial RF data, the models are trained from scratch using a highly realistic synthetic dataset generated via **Nvidia Sionna RT**. 
* **Environment:** OpenStreetMap data processed into Mitsuba XML via Blender.
* **Propagation Modeling:** Differentiable ray tracing simulating signal strength, path loss, shadowing, diffraction, and multi-path fading. 
* **Data Format:** $3D$ spatial tensors $(X, Y, \text{Frequency})$ stored in Parquet format.

*Average Path Gain Calculation:*
$$g_{i}=\frac{1}{|C_{i}|}\int_{C_{i}}|h(s)|^{2}ds$$

## Model Architectures & Ablation Study
This repository includes four distinct architectures benchmarked against one another to determine the optimal balance of reconstruction accuracy (MSE/RMSE) and edge-deployment latency on an Nvidia Jetson Orin Nano.

1. **Baseline CNN:** A standard convolutional neural network. Fast convergence, but struggles with high sparsity and fails to extrapolate global context from highly localized signal pockets.
2. **Graph Neural Network (GNN):** Treats pixels as nodes and physical distance as edges.
3. **Reconstructive Masked Autoencoder (RecMAE / ConvMAE):** Utilizes standard transformers with tokenized patches to reconstruct missing data. 
4. **PartialConvMAE (Proposed & Final):** Our novel architecture. It integrates *partial convolutional layers* into the MAE framework. Standard convolutions treat masked/unexplored regions (padded with zeros) as valid data, skewing results. Partial convolutions ensure that *only* physically captured HackRF measurements inform the feature extraction and spatial reconstruction.

## Project Structure
```text
├── data/                  # Scripts for Nvidia Sionna RT pipeline
├── models/
│   ├── CNN/               # Standard PyTorch CNN
│   ├── ConvMAE/           # Standard Masked Autoencoder
│   ├── GNN/               # Node/Edge based spectral network
│   ├── U-Net/             # U-Net implementation (CNN-based)
│   └── PartialConvMAE/    # Final architecture w/ partial convolutions
├── train.py               # Main training loop with hyperparameter tuning
├── evaluate.py            # Ablation study benchmarking (MSE, RMSE, Latency)
└── export_tensorrt.py     # Script to export the model to ONNX/TensorRT for Jetson
