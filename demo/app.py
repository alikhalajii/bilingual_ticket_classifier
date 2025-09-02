import gradio as gr
import torch
import os
import json

from bilingual_ticket_classifier.models.multi_head_classifier import MultiHeadTicketClassifier
from bilingual_ticket_classifier.processing.data_processor import DataProcessor


# Load model and assets
model_path = "./models/best_model"
encoder_path = os.path.join(model_path, "encoder")
tokenizer_path = os.path.join(model_path, "tokenizer")

# Load label mappings
with open(os.path.join(model_path, "queue_label_map.json")) as f:
    queue_labels = json.load(f)

with open(os.path.join(model_path, "type_label_map.json")) as f:
    type_labels = json.load(f)

# Reverse label maps: index → label
queue_labels = {v: k for k, v in queue_labels.items()}
type_labels = {v: k for k, v in type_labels.items()}

# Load processor and model
processor = DataProcessor(model_name=tokenizer_path)
model = MultiHeadTicketClassifier(
    model_name=encoder_path,
    num_labels_queue=len(queue_labels),
    num_labels_type=len(type_labels)
)
model.eval()

# Prediction function
def classify_ticket(text):
    inputs = processor.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits_queue = outputs["logits_queue"]
    logits_type = outputs["logits_type"]

    probs_queue = torch.softmax(logits_queue, dim=1)[0]
    probs_type = torch.softmax(logits_type, dim=1)[0]

    top_queue = torch.topk(probs_queue, 3)
    top_type = torch.topk(probs_type, 3)

    queue_output = "".join([
        f"<li><b>{queue_labels[idx.item()]}</b>: {score.item():.3f}</li>"
        for idx, score in zip(top_queue.indices, top_queue.values)
    ])
    type_output = "".join([
        f"<li><b>{type_labels[idx.item()]}</b>: {score.item():.3f}</li>"
        for idx, score in zip(top_type.indices, top_type.values)
    ])

    return f"""
    <div style='font-family:Arial; font-size:16px; line-height:1.6'>
        <h2 style='color:#2c3e50;'>🔍 Prediction Results</h2>
        <h3>📂 Top Queue Predictions:</h3>
        <ul>{queue_output}</ul>
        <h3>🔧 Top Type Predictions:</h3>
        <ul>{type_output}</ul>
        <hr>
        <p style='font-size:14px; color:#7f8c8d;'>Model: XLM-RoBERTa | Fine-tuned on bilingual support tickets</p>
    </div>
    """

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🎫 Multilingual Ticket Classifier
    Classifies support tickets into routing **Queue** and issue **Type** using a fine-tuned XLM-RoBERTa model.
    Paste your own ticket below and click <b>Classify</b> to see predictions.
    """)

    with gr.Column():
        input_box = gr.Textbox(lines=5, label="Ticket Text", placeholder="Describe your issue...")
        classify_btn = gr.Button("Classify")
        output_box = gr.HTML(label="Prediction")

        classify_btn.click(fn=classify_ticket, inputs=input_box, outputs=output_box)

demo.launch(share=True)


