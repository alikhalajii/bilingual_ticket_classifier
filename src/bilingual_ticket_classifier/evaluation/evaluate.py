import wandb
from transformers import TrainingArguments, EarlyStoppingCallback
import torch

from bilingual_ticket_classifier.training.trainer import MultiTaskTrainer, MultiTaskCollator
from bilingual_ticket_classifier.processing.data_processor import DataProcessor
from bilingual_ticket_classifier.models.multi_head_classifier import MultiHeadTicketClassifier
from bilingual_ticket_classifier.config.wandb_config import load_wandb_config


# Load config
config = load_wandb_config()

# Initialize W&B
wandb.init(project=config["PROJECT_NAME"], name=config["RUN_NAME"] + "-eval")
wandb.config.update(config)

# Prepare data
processor = DataProcessor(config["MODEL_NAME"])
tokenized_datasets = processor.prepare_data(config["DATASET_NAME"])

num_labels_queue = len(processor.label_encoder_queue.classes_)
num_labels_type = len(processor.label_encoder_type.classes_)

# Load model
model = MultiHeadTicketClassifier(
    model_name=config["MODEL_NAME"],
    num_labels_queue=num_labels_queue,
    num_labels_type=num_labels_type
)

collator = MultiTaskCollator(tokenizer=processor.tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=config["BATCH_SIZE"],
    report_to="wandb",
    fp16=torch.cuda.is_available()
)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=processor.tokenizer,
    data_collator=collator,
    compute_metrics=None
)

# Run evaluation
metrics = trainer.evaluate()
print(metrics)
