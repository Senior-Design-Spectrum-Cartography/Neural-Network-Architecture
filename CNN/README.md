## Basic CNN Implementation for Spectrum Map Reconstruct

There are two CNN implementations, one in Tensorflow and Keras, and another in PyTorch.

Set up a venv in VSCode (or your preferred IDE) and install requirements.txt using pip
```
python3 -m venv [venv_name]
source [venv_name}/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

To use, partition your image and CSV datasets into training, validation, and test sets. Next, update the file paths to reflect their directory location. For our dataset, we generated synthetic data using Nvidia Sionna RT.
