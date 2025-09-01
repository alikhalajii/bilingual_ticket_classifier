# Multitask bilingual ticket classification using XLM-RoBERTa

A pipeline for fine-tuning multilingual transformer models on bilingual customer support tickets. Built with 🤗 Transformers, 🧠 Weights & Biases, and 🔥 PyTorch.

Fine-tuned on the [ale-dp/bilingual-ticket-classification](https://huggingface.co/datasets/ale-dp/bilingual-ticket-classification) dataset using the multilingual model [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base).

## ⚡ Quickstart

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

## Results
All training runs are logged to Weights & Biases.
[Latest W&B run](https://wandb.ai/alikhalaji-/bilingual_ticket_classifier?nw=nwuseralikhalaji)
