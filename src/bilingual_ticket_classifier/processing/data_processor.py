from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    def __init__(self, model_name, max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder_queue = LabelEncoder()
        self.label_encoder_type = LabelEncoder()
        self.max_length = max_length

    def prepare_data(self, dataset_name, seed=42, test_size=0.2):
        """ Prepare dataset for training and evaluation. """
        raw_ds = load_dataset(dataset_name)["train"]

        ds_split = raw_ds.train_test_split(test_size=test_size, seed=seed)
        ds_test_valid = ds_split["test"].train_test_split(test_size=0.5, seed=seed)
        dataset_splits = DatasetDict({
            "train": ds_split["train"],
            "validation": ds_test_valid["train"],
            "test": ds_test_valid["test"]
        })

        def combine_text(example):
            return {
                "email": (example.get("subject") or "") + " " + (example.get("body") or ""),
                "queue_text": example["queue"],
                "type_text": example["type"]
            }

        for split in dataset_splits:
            dataset_splits[split] = dataset_splits[split].filter(
                lambda x: x.get("queue") and x.get("type")
            )
            dataset_splits[split] = dataset_splits[split].map(combine_text)

        self.label_encoder_queue.fit(dataset_splits["train"]["queue_text"])
        self.label_encoder_type.fit(dataset_splits["train"]["type_text"])

        def encode_labels(example):
            return {
                "label_queue": int(self.label_encoder_queue.transform([example["queue_text"]])[0]),
                "label_type": int(self.label_encoder_type.transform([example["type_text"]])[0])
            }

        dataset_splits = dataset_splits.map(encode_labels)

        def tokenize_function(examples):
            return self.tokenizer(
                examples["email"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        dataset_splits = dataset_splits.map(tokenize_function, batched=True, num_proc=4)

        dataset_splits = dataset_splits.remove_columns([
            "subject", "body", "queue", "type", "queue_text", "type_text", "email", "language"
        ])

        return dataset_splits

    def get_label_maps(self):
        """ Get label maps for queue and type labels."""
        return {
            "queue": {label: idx for idx, label in enumerate(self.label_encoder_queue.classes_)},
            "type": {label: idx for idx, label in enumerate(self.label_encoder_type.classes_)}
        }
