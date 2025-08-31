from transformers import Trainer
from processing.data_processor import DataProcessor
from config.wandb_config import load_wandb_config


def save_model(trainer: Trainer, processor: DataProcessor, save_path: str = "./best_model"):
    trainer.save_model(save_path)
    processor.tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")
