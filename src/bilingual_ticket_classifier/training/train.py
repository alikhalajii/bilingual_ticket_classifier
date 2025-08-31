import torch
import wandb
from transformers import TrainingArguments, EarlyStoppingCallback

from bilingual_ticket_classifier.processing.data_processor import DataProcessor
from bilingual_ticket_classifier.models.multi_head_classifier import MultiHeadTicketClassifier
from bilingual_ticket_classifier.training.trainer import MultiTaskTrainer, MultiTaskCollator, compute_metrics
from bilingual_ticket_classifier.config.wandb_config import load_wandb_config


# Load W&B config from .env
config = load_wandb_config()

wandb.init(project=config["PROJECT_NAME"], name=config["RUN_NAME"])
wandb.config.update(config)

model_name = config["MODEL_NAME"]
dataset_name = config["DATASET_NAME"]

processor = DataProcessor(model_name)
tokenized_datasets = processor.prepare_data(dataset_name)

num_labels_queue = len(processor.label_encoder_queue.classes_)
num_labels_type = len(processor.label_encoder_type.classes_)

model = MultiHeadTicketClassifier(
    model_name=model_name,
    num_labels_queue=num_labels_queue,
    num_labels_type=num_labels_type
)

collator = MultiTaskCollator(tokenizer=processor.tokenizer)

training_args = TrainingArguments(
    seed=42,
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    report_to="wandb",
    learning_rate=config["LEARNING_RATE"],
    per_device_train_batch_size=config["BATCH_SIZE"],
    per_device_eval_batch_size=config["BATCH_SIZE"],
    num_train_epochs=config["EPOCHS"],
    weight_decay=config["WEIGHT_DECAY"],
    load_best_model_at_end=True,
    metric_for_best_model="f1_queue",
    save_total_limit=2,
    greater_is_better=True,
    fp16=torch.cuda.is_available()
)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=processor.tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
