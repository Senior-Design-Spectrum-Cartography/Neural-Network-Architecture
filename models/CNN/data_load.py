import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from PIL import Image

# --- Configuration ---
DATA_DIR = 'path/to/your/dataset' # Update this!
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
# You will need to check the exact number of path gain values in your CSV
OUTPUT_DIM = 1000 # Example: Update this to match the number of columns/values in your CSV

class SpectrumDataGenerator(Sequence):
    def __init__(self, file_ids, data_dir, batch_size, img_shape, output_dim):
        self.file_ids = file_ids
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.output_dim = output_dim

    def __len__(self):
        return int(np.ceil(len(self.file_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_ids = self.file_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        batch_x = []
        batch_y = []
        
        for file_id in batch_ids:
            # Load Image (Input)
            img_path = os.path.join(self.data_dir, f"{file_id}.jpg") # Or .png
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_shape[1], self.img_shape[0]))
            img_array = np.array(img) / 255.0 # Normalize to [0, 1]
            batch_x.append(img_array)
            
            # Load CSV (Target)
            csv_path = os.path.join(self.data_dir, f"{file_id}.csv")
            # Assuming the CSV has a header and one row of values
            df = pd.read_csv(csv_path)
            # Flatten to a 1D array of floats
            target_array = df.values.flatten().astype(np.float32) 
            
            # Ensure it matches expected dimension, pad or truncate if necessary
            # (Ideally, all your CSVs have the exact same number of values)
            batch_y.append(target_array[:self.output_dim]) 
            
        return np.array(batch_x), np.array(batch_y)
