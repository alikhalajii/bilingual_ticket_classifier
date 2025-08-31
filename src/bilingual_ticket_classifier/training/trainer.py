from transformers import Trainer, DataCollatorWithPadding
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_preds):
    logits_queue, logits_type = eval_preds.predictions
    labels_queue, labels_type = eval_preds.label_ids

    preds_queue = np.argmax(logits_queue, axis=1)
    preds_type = np.argmax(logits_type, axis=1)

    metrics = {
        "eval_accuracy_queue": accuracy_score(labels_queue, preds_queue),
        "eval_accuracy_type": accuracy_score(labels_type, preds_type),
        "eval_f1_queue": f1_score(labels_queue, preds_queue, average="weighted"),
        "eval_f1_type": f1_score(labels_type, preds_type, average="weighted"),
    }

    return metrics

    
class MultiTaskCollator(DataCollatorWithPadding):
    def __call__(self, features):
        label_queue = [f["label_queue"] for f in features]
        label_type = [f["label_type"] for f in features]

        for f in features:
            del f["label_queue"]
            del f["label_type"]

        batch = super().__call__(features)
        batch["label_queue"] = torch.tensor(label_queue, dtype=torch.long)
        batch["label_type"] = torch.tensor(label_type, dtype=torch.long)
        return batch


class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_queue = inputs.pop("label_queue")
        labels_type = inputs.pop("label_type")

        outputs = model(**inputs)
        logits_queue = outputs["logits_queue"]
        logits_type = outputs["logits_type"]

        loss_fn = torch.nn.CrossEntropyLoss()
        loss_queue = loss_fn(logits_queue, labels_queue)
        loss_type = loss_fn(logits_type, labels_type)
        loss = loss_queue + loss_type

        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is not None:
            self.optimizer = optimizer
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
        return self.lr_scheduler

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        val_loss = output.metrics.get("eval_loss")
        if val_loss is not None:
            self.lr_scheduler.step(val_loss)
        return output
