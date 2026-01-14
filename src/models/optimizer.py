"""Model optimization utilities."""

import tensorflow as tf
from typing import Optional


def create_optimizer(
    learning_rate: float = 0.001,
    optimizer_type: str = 'adam',
    **kwargs
) -> tf.keras.optimizers.Optimizer:
    """Create optimizer for model training.
    
    Args:
        learning_rate: Learning rate for optimizer
        optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
        **kwargs: Additional optimizer parameters
        
    Returns:
        Configured optimizer instance
    """
    optimizers = {
        'adam': tf.keras.optimizers.Adam,
        'sgd': tf.keras.optimizers.SGD,
        'rmsprop': tf.keras.optimizers.RMSprop,
        'adamw': tf.keras.optimizers.AdamW
    }
    
    optimizer_class = optimizers.get(optimizer_type.lower(), tf.keras.optimizers.Adam)
    return optimizer_class(learning_rate=learning_rate, **kwargs)


def create_lr_schedule(
    initial_lr: float = 0.001,
    schedule_type: str = 'exponential',
    **kwargs
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """Create learning rate schedule.
    
    Args:
        initial_lr: Initial learning rate
        schedule_type: Type of schedule ('exponential', 'cosine', 'polynomial')
        **kwargs: Additional schedule parameters
        
    Returns:
        Learning rate schedule
    """
    if schedule_type == 'exponential':
        decay_rate = kwargs.get('decay_rate', 0.96)
        decay_steps = kwargs.get('decay_steps', 10000)
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_lr, decay_steps, decay_rate, staircase=True
        )
    
    elif schedule_type == 'cosine':
        decay_steps = kwargs.get('decay_steps', 10000)
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_lr, decay_steps
        )
    
    elif schedule_type == 'polynomial':
        decay_steps = kwargs.get('decay_steps', 10000)
        end_learning_rate = kwargs.get('end_learning_rate', 0.0001)
        power = kwargs.get('power', 1.0)
        return tf.keras.optimizers.schedules.PolynomialDecay(
            initial_lr, decay_steps, end_learning_rate, power
        )
    
    else:
        return initial_lr


class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warm-up period."""
    
    def __init__(
        self,
        initial_lr: float,
        warmup_steps: int = 1000,
        target_lr: Optional[float] = None
    ):
        """Initialize warm-up schedule.
        
        Args:
            initial_lr: Starting learning rate
            warmup_steps: Number of warm-up steps
            target_lr: Target learning rate after warm-up
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr or initial_lr * 10
        
    def __call__(self, step):
        """Calculate learning rate for given step."""
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        
        # Linear warm-up
        warmup_lr = self.initial_lr + (
            (self.target_lr - self.initial_lr) * (step / warmup_steps)
        )
        
        # Use warm-up LR if still in warm-up period, otherwise use target LR
        return tf.cond(
            step < warmup_steps,
            lambda: warmup_lr,
            lambda: self.target_lr
        )
