import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- 1. Configuration ---
TRAIN_IMG_DIR = '/Users/intoward/Documents/Senior Design/datasets/samples_jpg/training'
VAL_IMG_DIR = '/Users/intoward/Documents/Senior Design/datasets/samples_jpg/validation'
TEST_IMG_DIR = '/Users/intoward/Documents/Senior Design/datasets/samples_jpg/testing'

TRAIN_CSV_DIR = '/Users/intoward/Documents/Senior Design/datasets/samples_csv/training' 
VAL_CSV_DIR = '/Users/intoward/Documents/Senior Design/datasets/samples_csv/validation'
TEST_CSV_DIR = '/Users/intoward/Documents/Senior Design/datasets/samples_csv/testing'

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
OUTPUT_DIM = 40000 # *Update this to the exact number of elements in your CSVs*
EPOCHS = 30

# --- 2. Dataset & DataLoader ---
class SpectrumDataset(Dataset):
    def __init__(self, img_dir, csv_dir, img_shape, output_dim):
        self.img_dir = img_dir
        self.csv_dir = csv_dir
        self.output_dim = output_dim
        
        self.valid_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.valid_files.sort()
        
        self.transform = transforms.Compose([
            transforms.Resize((img_shape[0], img_shape[1])),
            transforms.ToTensor() 
        ])

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        file_name = self.valid_files[idx]
        file_id = file_name.split('.')[0]
        
        # Load Image
        img_path = os.path.join(self.img_dir, file_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        # Load CSV
        csv_path = os.path.join(self.csv_dir, f"{file_id}.csv")
        df = pd.read_csv(csv_path)
        target_array = df.values.flatten().astype(np.float32)
        
        if len(target_array) < self.output_dim:
            target_array = np.pad(target_array, (0, self.output_dim - len(target_array)))
        else:
            target_array = target_array[:self.output_dim]
            
        target_tensor = torch.tensor(target_array)
        return img_tensor, target_tensor

train_dataset = SpectrumDataset(TRAIN_IMG_DIR, TRAIN_CSV_DIR, (IMG_HEIGHT, IMG_WIDTH), OUTPUT_DIM)
val_dataset = SpectrumDataset(VAL_IMG_DIR, VAL_CSV_DIR, (IMG_HEIGHT, IMG_WIDTH), OUTPUT_DIM)
test_dataset = SpectrumDataset(TEST_IMG_DIR, TEST_CSV_DIR, (IMG_HEIGHT, IMG_WIDTH), OUTPUT_DIM)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Datasets Loaded -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# --- 3. Model Architecture ---
class SpectrumCNN(nn.Module):
    def __init__(self, output_dim):
        super(SpectrumCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        flattened_size = 128 * 32 * 32
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# Setup device and INITIALIZE THE MODEL (This is what was missing!)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = SpectrumCNN(OUTPUT_DIM).to(device)

# --- 4. Training Configuration ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# --- 5. Training Loop ---
best_val_loss = float('inf')
patience = 7
patience_counter = 0

history_train_loss = []
history_val_loss = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad() 
        outputs = model(images) 
        loss = criterion(outputs, targets) 
        loss.backward() 
        optimizer.step() 
        
        train_loss += loss.item() * images.size(0)
        
    train_loss /= len(train_loader.dataset)
    history_train_loss.append(train_loss)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)
            
    val_loss /= len(val_loader.dataset)
    history_val_loss.append(val_loss)
    
    scheduler.step(val_loss) 
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth') 
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

# --- 6. Visualization Functions ---
def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss (MSE)', linewidth=2)
    plt.plot(val_losses, label='Validation Loss (MSE)', linewidth=2, linestyle='--')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss_curve.png')
    plt.show()

def plot_heatmap_and_scatter(model, test_loader, device, output_dim):
    model.eval()
    dataiter = iter(test_loader)
    images, targets = next(dataiter)
    images, targets = images.to(device), targets.to(device)
    
    with torch.no_grad():
        preds = model(images)
    
    true_1d = targets[0].cpu().numpy()
    pred_1d = preds[0].cpu().numpy()
    grid_size = int(np.sqrt(output_dim))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(true_1d.reshape((grid_size, grid_size)), ax=axes[0], cmap='viridis', cbar_kws={'label': 'Path Gain (dB)'})
    axes[0].set_title('Ground Truth: Path Gain')
    axes[0].axis('off')
    
    sns.heatmap(pred_1d.reshape((grid_size, grid_size)), ax=axes[1], cmap='viridis', cbar_kws={'label': 'Path Gain (dB)'})
    axes[1].set_title('Model Prediction: Path Gain')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('heatmap_comparison.png')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.scatter(true_1d, pred_1d, alpha=0.1, color='blue', s=1)
    min_val = min(true_1d.min(), pred_1d.min())
    max_val = max(true_1d.max(), pred_1d.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    plt.title('Predicted vs. Actual Path Gain (Single Sample)')
    plt.xlabel('True Path Gain (dB)')
    plt.ylabel('Predicted Path Gain (dB)')
    plt.legend()
    plt.grid(True)
    plt.savefig('scatter_metrics.png')
    plt.show()

# --- 7. Final Evaluation & Plot Execution ---
print("\nEvaluating and Plotting Metrics...")
model.load_state_dict(torch.load('best_model.pth')) 

plot_training_curves(history_train_loss, history_val_loss)
plot_heatmap_and_scatter(model, test_loader, device, OUTPUT_DIM)

model.eval()
test_loss = 0.0
test_mae = 0.0

with torch.no_grad():
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * images.size(0)
        mae = torch.abs(outputs - targets).mean()
        test_mae += mae.item() * images.size(0)

test_loss /= len(test_loader.dataset)
test_mae /= len(test_loader.dataset)

print(f"Final Test Loss (MSE): {test_loss:.4f}")
print(f"Final Test MAE (dB): {test_mae:.4f}")
