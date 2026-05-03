# U-net training for Radio-Map data

This repository contains a google colab jupyter notebook for training a U-net model for radio-map reconstruction from sparse measurements adapted from the Radio-U-Net architecture.

The pipeline is built to process 3-channel parquet inputs of radio maps consisting of a binary mask of the physical environment, location of the emitter, and simulated radiomap of the environment.

## Usage

1.  **Open in Colab:** Upload the desired notebook to Google Colab.
2.  **Mount Drive:** The notebooks expect your dataset (link provided in data README) to be stored in Google Drive under:
    * `/content/drive/MyDrive/SCv3_compressed/{train-(001 to 004),val,test}`
4.  **Run All:** Execute the cells. The notebook will provision the GPU, install dependencies, prepare the data, and launch the single-GPU pre-training script.

## Dependencies

The notebooks handle the installation of required packages.
