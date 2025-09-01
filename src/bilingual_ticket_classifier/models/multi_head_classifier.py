import torch.nn as nn
from transformers import AutoModel


class MultiHeadTicketClassifier(nn.Module):
    """ Multi-head classifier for ticket classification."""
    def __init__(self, model_name, num_labels_queue, num_labels_type, dropout_rate=0.1):
        super(MultiHeadTicketClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier_queue = nn.Linear(hidden_size, num_labels_queue)
        self.classifier_type = nn.Linear(hidden_size, num_labels_type)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_ids, attention_mask, label_queue=None, label_type=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        logits_queue = self.classifier_queue(cls_output)
        logits_type = self.classifier_type(cls_output)

        loss = None
        if label_queue is not None and label_type is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss_queue = loss_fn(logits_queue, label_queue)
            loss_type = loss_fn(logits_type, label_type)
            loss = loss_queue + loss_type

        return {
            "loss": loss,
            "logits_queue": logits_queue,
            "logits_type": logits_type
        }
