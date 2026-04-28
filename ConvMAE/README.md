# ConvMAE Pre-Training for Radio-Map Data

This repository contains two complete, self-contained Google Colab Jupyter Notebooks for pre-training Convolutional Masked Autoencoders (ConvMAE) on spatial radio-map datasets. 

The pipelines are built to handle two distinct data formats—**Parquet** files containing raw scalar data and **PNG** images containing pre-rendered heatmaps. Both notebooks manage environment setup, dataset partitioning, model training, and performance visualization.

## 📁 Repository Contents

* [`ConvMAE_Parquet_Pretrain.ipynb`](./ConvMAE_Parquet_Pretrain.ipynb)
    * **Data Input:** Parquet files, where each file contains a single sample with a flat `map` column of 810,000 `float64` dBm values (representing a 900×900 grid).
    * **Processing:** Dynamically reshapes the array, normalizes the dBm values to a `[0, 1]` range, and renders them on-the-fly into RGB images using a Jet colormap.
    * **Use Case:** Best when working directly with raw simulation output or unprocessed sensor data.
* [`ConvMAE_PNG_Pretrain.ipynb`](./ConvMAE_PNG_Pretrain.ipynb)
    * **Data Input:** Pre-rendered 900×900 px PNG images.
    * **Processing:** Leverages a standard PyTorch `ImageFolder` structure with symbolic links for train/val/test splits to efficiently load image data.
    * **Use Case:** Best when the spatial data has already been rendered into heatmap visuals.

## ✨ Key Features

* **Automated Environment & Patching:** Both notebooks automatically clone the [Alpha-VL/ConvMAE](https://github.com/Alpha-VL/ConvMAE) repository and apply critical patches to ensure compatibility with modern deep learning environments:
    * Replaces deprecated `np.float` with `np.float32` (NumPy 1.24+).
    * Monkey-patches `torch._six` for PyTorch 2.x compatibility.
    * Updates `misc.add_weight_decay` to `optim_factory.add_weight_decay` (`timm==0.3.2`).
    * Updates deprecated AMP calls (`torch.cuda.amp.GradScaler` and `autocast` -> `torch.amp` equivalents).
* **Custom Data Loaders:** Handles distinct data modalities, managing an 80/10/10 split natively.
* **Performance Visualization:** Generates a comprehensive 6-panel performance dashboard upon training completion, which includes:
    * Learning Curve (MSE and Smoothed MSE)
    * RMSE Tracking
    * Learning Rate Schedule
    * MSE Distribution (Histogram)
    * Epoch-over-Epoch $\Delta$ MSE
    * Log-Scale Convergence
* **Automated Logging:** Exports summary statistics to a CSV log and manages checkpoint saving dynamically.

## 🚀 Usage

1.  **Open in Colab:** Upload the desired notebook to Google Colab.
2.  **Mount Drive:** The notebooks expect your datasets to be stored in Google Drive under:
    * Parquet: `dataset/labels_parquet_v2/{training,validation,testing}`
    * PNG: `dataset/labels_png_v2/{training,validation,testing}`
3.  **Configure Hyperparameters:** Modify Section 9 (Parquet) or Section 8 (PNG) to adjust specific hyperparameters. Default parameters include:
    * `MODEL = 'convmae_convvit_base_patch16'`
    * `INPUT_SIZE = 224`
    * `MASK_RATIO = 0.75`
    * `EPOCHS = 50`
4.  **Run All:** Execute the cells. The notebook will provision the GPU, install dependencies, prepare the data, and launch the single-GPU pre-training script.

## 🛠 Dependencies

The notebooks handle the installation of required packages natively via `pip`. Core dependencies include:
* `torch` & `torchvision` (PyTorch 2.x compatible)
* `timm==0.3.2`
* `einops`
* `pandas` & `pyarrow` (for Parquet reading)
* `matplotlib` & `seaborn` (for visualization)
