# Multitask bilingual ticket classification using XLM-RoBERTa

A pipeline for fine-tuning multilingual transformer models on bilingual customer support tickets. Built with 🤗 Transformers, 🧠 Weights & Biases, and 🔥 PyTorch.

Fine-tuned on the [ale-dp/bilingual-ticket-classification](https://huggingface.co/datasets/ale-dp/bilingual-ticket-classification) dataset using the multilingual model [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base).

## ⚡ Quickstart

**Clone the Repository**
```bash
git clone https://github.com/alikhalajii/bilingual_ticket_classifier.git
cd bilingual_ticket_classifier
```

If you want to skip training and directly run the demo using the fine-tuned model:
```bash
git clone https://github.com/alikhalajii/bilingual_ticket_classifier.git && cd bilingual_ticket_classifier && git lfs install && git lfs pull
```

**Install the repository as a Python package**
    
```bash
pip install -e .
```

**Set up environment variables**

Make sure to add your Weights & Biases API key and Hugging Face token to the .env file in the project root:
**Train the model**

```bash
python src/bilingual_ticket_classifier/training/train.py
```

**Evaluate the Model**
```bash
python src/bilingual_ticket_classifier/evaluation/evaluate.py
```

**Launch the Gradio Demo**
```bash
python demo/app.py
```

## 📊 Results

All training runs are logged to **Weights & Biases** for full transparency and reproducibility.

- [Latest W&B Run Dashboard](https://wandb.ai/alikhalaji-/bilingual_ticket_classifier?nw=nwuseralikhalaji)  
  Explore training curves, evaluation metrics, and system logs.

- [Summary JSON Snapshot](https://github.com/alikhalajii/bilingual_ticket_classifier/blob/main/wandb/run-20250901_123805-r9p9erpj/files/wandb-summary.json)  
  Quick access to final metrics.
