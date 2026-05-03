import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import Sequence
from PIL import Image

# --- Configuration ---
# Update these to match your local paths
TRAIN_IMG_DIR = '/Users/intoward/Documents/Senior Design/datasets/samples_csv/training'
VAL_IMG_DIR = '/Users/intoward/Documents/Senior Design/datasets/samples_csv/validation'
TEST_IMG_DIR = '/Users/intoward/Documents/Senior Design/datasets/samples_csv/testing'

# Assuming CSVs are in the same directories. If they are in a separate 'samples_csv' 
# directory with the same train/val/test split, update the paths accordingly.
TRAIN_CSV_DIR = '/Users/intoward/Documents/Senior Design/datasets/samples_jpg/training' 
VAL_CSV_DIR = '/Users/intoward/Documents/Senior Design/datasets/samples_jpg/validation'
TEST_CSV_DIR = '/Users/intoward/Documents/Senior Design/datasets/samples_jpg/testing'

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
OUTPUT_DIM = 40000 # *You will need to update this to the exact number of elements in your CSVs*

class SpectrumDataGenerator(Sequence):
    def __init__(self, img_dir, csv_dir, batch_size, img_shape, output_dim):
        self.img_dir = img_dir
        self.csv_dir = csv_dir
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.output_dim = output_dim
        
        # Grab all .jpg/.png files in the target directory
        self.valid_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.valid_files.sort() # Ensure consistent order

    def __len__(self):
        return int(np.ceil(len(self.valid_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_files = self.valid_files[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        batch_x = []
        batch_y = []
        
        for file_name in batch_files:
            file_id = file_name.split('.')[0]
            
            # Load Image
            img_path = os.path.join(self.img_dir, file_name)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_shape[1], self.img_shape[0]))
            img_array = np.array(img) / 255.0 # Normalize
            batch_x.append(img_array)
            
            # Load CSV
            csv_path = os.path.join(self.csv_dir, f"{file_id}.csv")
            df = pd.read_csv(csv_path)
            target_array = df.values.flatten().astype(np.float32) 
            
            # Pad or truncate to ensure uniform shape (optional based on your data)
            batch_y.append(target_array[:self.output_dim]) 
            
        return np.array(batch_x), np.array(batch_y)

# Initialize Generators based on explicit directories
train_gen = SpectrumDataGenerator(TRAIN_IMG_DIR, TRAIN_CSV_DIR, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH), OUTPUT_DIM)
val_gen = SpectrumDataGenerator(VAL_IMG_DIR, VAL_CSV_DIR, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH), OUTPUT_DIM)
test_gen = SpectrumDataGenerator(TEST_IMG_DIR, TEST_CSV_DIR, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH), OUTPUT_DIM)

print(f"Generators Loaded -> Train: {len(train_gen.valid_files)} | Val: {len(val_gen.valid_files)} | Test: {len(test_gen.valid_files)}")

# --- Model Architecture with L2 Regularization ---
def build_regularized_cnn(input_shape, output_dim):
    # L2 Regularizer
    l2_reg = regularizers.l2(1e-4) # 0.0001 is a standard starting point
    
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        
        # Dense mapping with L2 and Dropout
        layers.Dense(512, activation='relu', kernel_regularizer=l2_reg),
        layers.Dropout(0.4), # Increased slightly to combat small dataset size
        
        layers.Dense(256, activation='relu', kernel_regularizer=l2_reg),
        layers.Dropout(0.3),
        
        # Output Layer: Linear activation
        layers.Dense(output_dim, activation='linear') 
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', 
                  metrics=['mae'])
    return model

# Build the model
model = build_regularized_cnn((IMG_HEIGHT, IMG_WIDTH, 3), OUTPUT_DIM)

# --- Training ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30, # Slightly higher since regularization will slow down overfitting
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3) # Drops learning rate if val_loss plateaus
    ]
)

# --- Evaluation ---
test_loss, test_mae = model.evaluate(test_gen)
print(f"\nFinal Test Loss (MSE): {test_loss:.4f}")
print(f"Final Test MAE (dB): {test_mae:.4f}")
