“””
Training pipeline for SwipeLSTM model.
“””
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

from ..config import TRAINING_CONFIG, MODEL_CONFIG, CHECKPOINTS_DIR, LOGS_DIR
from .metrics import calculate_accuracy, calculate_top_k_accuracy

class GestureDataset(Dataset):
“”“PyTorch Dataset for swipe gestures.”””

```
def __init__(
    self, 
    data: List[Dict],
    word_to_idx: Dict[str, int],
    max_length: int = 50
):
    """
    Initialize dataset.
    
    Args:
        data: List of gesture dictionaries
        word_to_idx: Mapping from words to indices
        max_length: Maximum sequence length
    """
    self.data = data
    self.word_to_idx = word_to_idx
    self.max_length = max_length

def __len__(self) -> int:
    return len(self.data)

def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
    """
    Get a single sample.
    
    Returns:
        (gesture_tensor, label, actual_length)
    """
    sample = self.data[idx]
    
    # Get trajectory and timestamps
    trajectory = np.array(sample['trajectory'])  # (N, 2)
    timestamps = np.array(sample['timestamps']).reshape(-1, 1)  # (N, 1)
    
    # Normalize timestamps to 0-1
    if len(timestamps) > 1:
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
    
    # Combine into (N, 3) features
    features = np.concatenate([trajectory, timestamps], axis=1)
    
    # Pad or truncate to max_length
    actual_length = len(features)
    if len(features) < self.max_length:
        padding = np.zeros((self.max_length - len(features), 3))
        features = np.vstack([features, padding])
    else:
        features = features[:self.max_length]
        actual_length = self.max_length
    
    # Get label
    word = sample['word']
    label = self.word_to_idx.get(word, 0)  # 0 is <UNK>
    
    return (
        torch.FloatTensor(features),
        label,
        actual_length
    )
```

class Trainer:
“”“Training manager for SwipeLSTM.”””

```
def __init__(
    self,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cpu",
    learning_rate: float = 0.001,
    checkpoint_dir: Path = CHECKPOINTS_DIR
):
    """
    Initialize trainer.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        learning_rate: Initial learning rate
        checkpoint_dir: Directory to save checkpoints
    """
    self.model = model.to(device)
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.device = device
    self.checkpoint_dir = Path(checkpoint_dir)
    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss function
    self.criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    self.optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=MODEL_CONFIG.get("weight_decay", 1e-5)
    )
    
    # Learning rate scheduler
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer,
        mode='min',
        factor=TRAINING_CONFIG["reduce_lr_factor"],
        patience=TRAINING_CONFIG["reduce_lr_patience"],
        min_lr=TRAINING_CONFIG["min_lr"],
        verbose=True
    )
    
    # TensorBoard writer
    self.writer = SummaryWriter(log_dir=str(LOGS_DIR / "tensorboard"))
    
    # Training state
    self.epoch = 0
    self.best_val_loss = float('inf')
    self.patience_counter = 0
    self.history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top5_acc": []
    }

def train_epoch(self) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        (average_loss, average_accuracy)
    """
    self.model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
    
    for batch_idx, (inputs, labels, lengths) in enumerate(pbar):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs, lengths)
        
        # Calculate loss
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        
        # Update metrics
        total_loss += loss.item()
        total_correct += correct
        total_samples += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / labels.size(0):.2f}%'
        })
        
        # Log to TensorBoard
        global_step = self.epoch * len(self.train_loader) + batch_idx
        self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    avg_loss = total_loss / len(self.train_loader)
    avg_acc = 100 * total_correct / total_samples
    
    return avg_loss, avg_acc

def validate(self) -> Tuple[float, float, float]:
    """
    Validate model.
    
    Returns:
        (average_loss, top1_accuracy, top5_accuracy)
    """
    self.model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_top5_predictions = []
    
    with torch.no_grad():
        for inputs, labels, lengths in tqdm(self.val_loader, desc="Validation"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs, lengths)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Get top-5 predictions
            _, top5_pred = torch.topk(outputs, 5, dim=1)
            
            # Store for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_top5_predictions.extend(top5_pred.cpu().numpy())
    
    avg_loss = total_loss / len(self.val_loader)
    top1_acc = calculate_accuracy(all_predictions, all_labels)
    top5_acc = calculate_top_k_accuracy(all_top5_predictions, all_labels, k=5)
    
    return avg_loss, top1_acc, top5_acc

def train(
    self,
    num_epochs: int = 50,
    early_stopping_patience: int = 10
) -> Dict:
    """
    Full training loop.
    
    Args:
        num_epochs: Number of epochs to train
        early_stopping_patience: Patience for early stopping
        
    Returns:
        Training history dictionary
    """
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {self.device}")
    print(f"Training samples: {len(self.train_loader.dataset)}")
    print(f"Validation samples: {len(self.val_loader.dataset)}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        self.epoch = epoch + 1
        
        # Train
        train_loss, train_acc = self.train_epoch()
        
        # Validate
        val_loss, val_top1, val_top5 = self.validate()
        
        # Update learning rate
        self.scheduler.step(val_loss)
        
        # Log metrics
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_top1)
        self.history["val_top5_acc"].append(val_top5)
        
        # TensorBoard logging
        self.writer.add_scalar('Train/Loss', train_loss, epoch)
        self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        self.writer.add_scalar('Val/Top1Accuracy', val_top1, epoch)
        self.writer.add_scalar('Val/Top5Accuracy', val_top5, epoch)
        self.writer.add_scalar('LearningRate', 
                             self.optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%")
        print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint(is_best=True)
            print(f"  ✓ New best model saved!")
        else:
            self.patience_counter += 1
            print(f"  Patience: {self.patience_counter}/{early_stopping_patience}")
        
        # Early stopping
        if self.patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
        
        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            self.save_checkpoint(is_best=False)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes")
    print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    self.writer.close()
    return self.history

def save_checkpoint(self, is_best: bool = False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': self.epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
        'best_val_loss': self.best_val_loss,
        'history': self.history
    }
    
    if is_best:
        path = self.checkpoint_dir / "best_model.pt"
    else:
        path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
    
    torch.save(checkpoint, path)

def load_checkpoint(self, path: Path):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=self.device)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    self.epoch = checkpoint['epoch']
    self.best_val_loss = checkpoint['best_val_loss']
    self.history = checkpoint['history']
    print(f"Loaded checkpoint from epoch {self.epoch}")
```
