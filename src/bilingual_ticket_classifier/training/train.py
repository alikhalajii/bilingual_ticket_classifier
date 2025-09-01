import torch
import wandb
from transformers import TrainingArguments, EarlyStoppingCallback

from bilingual_ticket_classifier.processing.data_processor import DataProcessor
from bilingual_ticket_classifier.models.multi_head_classifier import MultiHeadTicketClassifier
from bilingual_ticket_classifier.training.trainer import MultiTaskTrainer, MultiTaskCollator, compute_metrics
from bilingual_ticket_classifier.config.wandb_config import load_wandb_config
from save_model import save_model


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
    output_dir="./results",
    save_total_limit=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    report_to="wandb",
    run_name=config["RUN_NAME"],
    learning_rate=config["LEARNING_RATE"],
    per_device_train_batch_size=config["BATCH_SIZE"],
    per_device_eval_batch_size=config["BATCH_SIZE"],
    num_train_epochs=config["EPOCHS"],
    weight_decay=config["WEIGHT_DECAY"],
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    fp16=torch.cuda.is_available(),
    warmup_ratio=config.get("WARMUP_RATIO", 0.08),
    lr_scheduler_type="cosine_with_restarts",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

# Save model
save_path = config["FINETUNED_MODEL_PATH"]

save_model(trainer, processor, save_path=save_path)

wandb.config.update({"FINETUNED_MODEL_PATH": save_path})