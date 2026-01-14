"""
Model trainer for GestureFlow.
"""
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import Optional
import json

from ..config import CHECKPOINT_DIR, TRAINING_CONFIG


class ModelTrainer:
    """Train GestureFlow models."""
    
    def __init__(
        self,
        model: keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        checkpoint_dir: Path = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        epochs: int = 50,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 5
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Update learning rate
        self.model.optimizer.learning_rate.assign(learning_rate)
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.checkpoint_dir / "best_model.h5"),
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(self.checkpoint_dir / "logs"),
                histogram_freq=1
            )
        ]
        
        # Train
        print(f"Starting training for {epochs} epochs...")
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = self.checkpoint_dir / "final_model.h5"
        self.model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Save training history
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=2)
        print(f"Training history saved to {history_path}")
        
        return history
    
    def evaluate(self) -> dict:
        """
        Evaluate model on validation set.
        
        Returns:
            Evaluation metrics
        """
        print("Evaluating model...")
        results = self.model.evaluate(self.val_dataset, verbose=1)
        
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = float(value)
            print(f"{name}: {value:.4f}")
        
        return metrics


def main():
    """Test trainer."""
    import argparse
    from ..data.loader import GestureDataLoader
    from ..models.lstm_model import create_model
    from ..config import PROCESSED_DATA_DIR
    
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "--data-file",
        type=str,
        default="processed_en.json",
        help="Data file name"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs"
    )
    
    args = parser.parse_args()
    
    data_path = PROCESSED_DATA_DIR / args.data_file
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    # Load data
    loader = GestureDataLoader(
        data_path=data_path,
        batch_size=TRAINING_CONFIG["batch_size"],
        validation_split=TRAINING_CONFIG["validation_split"]
    )
    
    train_ds, val_ds = loader.get_datasets()
    
    # Create model
    model = create_model(vocab_size=loader.vocab_size)
    
    # Train
    trainer = ModelTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds
    )
    
    history = trainer.train(
        epochs=args.epochs,
        learning_rate=TRAINING_CONFIG.get('learning_rate', 0.001),
        early_stopping_patience=TRAINING_CONFIG.get('early_stopping_patience', 5)
    )
    
    # Evaluate
    metrics = trainer.evaluate()
    print(f"\nFinal metrics: {metrics}")


if __name__ == "__main__":
    main()
