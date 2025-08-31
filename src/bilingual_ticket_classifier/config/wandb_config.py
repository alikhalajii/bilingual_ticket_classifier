from dotenv import load_dotenv
import os


def load_wandb_config():
    load_dotenv()

    return {
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "PROJECT_NAME": os.getenv("PROJECT_NAME", "bilingual_ticket_classifier"),
        "RUN_NAME": os.getenv("RUN_NAME", "xlm-roberta-run"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "FacebookAI/xlm-roberta-base"),
        "DATASET_NAME": os.getenv("DATASET_NAME", "ale-dp/bilingual-ticket-classification"),
        "EPOCHS": int(os.getenv("EPOCHS", 10)),
        "BATCH_SIZE": int(os.getenv("BATCH_SIZE", 16)),
        "LEARNING_RATE": float(os.getenv("LEARNING_RATE", 2e-5)),
        "WEIGHT_DECAY": float(os.getenv("WEIGHT_DECAY", 0.01))
    }
