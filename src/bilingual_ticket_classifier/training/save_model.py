import os
import json

from transformers import Trainer
from bilingual_ticket_classifier.processing.data_processor import DataProcessor
from bilingual_ticket_classifier.config.wandb_config import load_wandb_config


def save_model(trainer, processor, save_path):

    label_maps = processor.get_label_maps()

    with open(os.path.join(save_path, "queue_label_map.json"), "w") as f:
        json.dump(label_maps["queue"], f)

    with open(os.path.join(save_path, "type_label_map.json"), "w") as f:
        json.dump(label_maps["type"], f)


    trainer.save_model(save_path)

    encoder_path = os.path.join(save_path, "encoder")
    trainer.model.encoder.save_pretrained(encoder_path)

    tokenizer_path = os.path.join(save_path, "tokenizer")
    processor.tokenizer.save_pretrained(tokenizer_path)

    print(f"Full model saved to: {save_path}")
    print(f"Encoder saved to: {encoder_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")
