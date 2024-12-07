import argparse
import random
import os
import json
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

def tokenize_and_align_labels(text, entities, tokenizer, label_to_id, max_length=128):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    labels = ["O"] * len(encoding.input_ids[0])

    for entity in entities:
        entity_label = entity['label']
        start, end = entity['start'], entity['end']
        
        token_start_index = encoding.char_to_token(0, start)
        token_end_index = encoding.char_to_token(0, end - 1)
        
        if token_start_index is None or token_end_index is None:
            continue
        
        labels[token_start_index] = f"B-{entity_label}"
        for i in range(token_start_index + 1, token_end_index + 1):
            labels[i] = f"I-{entity_label}"

    label_ids = [label_to_id.get(label, -100) for label in labels]
    encoding["labels"] = torch.tensor(label_ids)
    
    return {k: v.squeeze() for k, v in encoding.items()}


class CustomNERDataset(TorchDataset):
    def __init__(self, data, tokenizer, label_to_id, max_length=128):
        print("Initializing CustomNERDataset...")
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['source_text']
        entities = self.data[idx]['privacy_mask']
        encoded_data = tokenize_and_align_labels(text, entities, self.tokenizer, self.label_to_id, self.max_length)
        encoded_data["source_text"] = text
        encoded_data["privacy_mask"] = self.data[idx]['privacy_mask']
        return encoded_data


def display_random_samples(dataset, tokenizer, num_samples=10, output_file="random_samples.txt"):
    print(f"Displaying {num_samples} random samples to {output_file}...")
    sep_token = tokenizer.sep_token_id
    
    with open(output_file, "w") as file:
        samples = random.sample(list(dataset), num_samples)
        
        for sample in samples:
            file.write("\n--- Random Sample ---\n")
            file.write("Source Text: " + sample["source_text"] + "\n")
            file.write("Privacy Mask:\n")
            for entity in sample["privacy_mask"]:
                file.write(f"  - Label: {entity['label']}, Value: {entity['value']}, Start: {entity['start']}, End: {entity['end']}\n")
            
            tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
            labels = sample["labels"]
            
            last_valid_index = len(tokens) - 1
            for i, token_id in enumerate(sample["input_ids"]):
                if token_id == sep_token:
                    last_valid_index = i
                    break

            file.write("\nToken...Label\n")
            for token, label_id in zip(tokens[:last_valid_index + 1], labels[:last_valid_index + 1]):
                label = label_id_to_name.get(label_id, "PAD")
                file.write(f"{token:<15}...{label}\n")
            file.write("\n")
    
    print(f"Random samples written to {output_file}")


def prepare_data_for_saving(custom_dataset):
    print("Preparing data for saving to Hugging Face dataset format...")
    data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "source_text": [],
        "privacy_mask": []
    }
    
    for i, sample in enumerate(custom_dataset):
        data["input_ids"].append(sample["input_ids"].tolist())
        data["attention_mask"].append(sample["attention_mask"].tolist())
        data["labels"].append(sample["labels"].tolist())
        data["source_text"].append(custom_dataset.data[i]['source_text'])
        data["privacy_mask"].append(custom_dataset.data[i]['privacy_mask'])
        
    print("Data preparation completed.")
    return Dataset.from_pandas(pd.DataFrame(data))


def save_label_mappings(label_to_id, output_dir):
    """Saves label mappings to a JSON file for later use in evaluation and training."""
    label_mappings = {
        "label_to_id": label_to_id,
        "id_to_label": {v: k for k, v in label_to_id.items()}
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "label_mappings.json"), "w") as f:
        json.dump(label_mappings, f, indent=4)
    print(f"Label mappings saved to {os.path.join(output_dir, 'label_mappings.json')}")


def main(args):
    print(f"Loading tokenizer '{args.tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print("Tokenizer loaded successfully.")

    print(f"Loading dataset '{args.dataset_name}'...")
    dataset = load_dataset(args.dataset_name)
    train_data = dataset['train'].filter(lambda x: x['language'] == 'en')
    print(f"Filtered training data to {len(train_data)} English samples.")

    if 'validation' not in dataset:
        print("Validation set not found. Creating validation split from training data...")
        train_data = train_data.shuffle(seed=42)
        validation_size = int(0.2 * len(train_data))
        validation_data = train_data.select(range(validation_size))
        train_data = train_data.select(range(validation_size, len(train_data)))
    else:
        validation_data = dataset['validation'].filter(lambda x: x['language'] == 'en')
    print(f"Training samples: {len(train_data)}, Validation samples: {len(validation_data)}")

    print("Defining label mappings...")
    label_list = sorted(list(set(label for sample in train_data for entity in sample['privacy_mask'] for label in [f"B-{entity['label']}", f"I-{entity['label']}"])))
    label_to_id = {label: i for i, label in enumerate(label_list)}
    label_to_id["O"] = len(label_to_id)
    global label_id_to_name
    label_id_to_name = {v: k for k, v in label_to_id.items()}
    print("Label mappings defined successfully.")

    # Save label mappings
    save_label_mappings(label_to_id, args.output_path)

    print("Processing train and validation splits into CustomNERDataset format...")
    train_dataset = CustomNERDataset(train_data, tokenizer, label_to_id, max_length=args.max_length)
    validation_dataset = CustomNERDataset(validation_data, tokenizer, label_to_id, max_length=args.max_length)
    processed_train_data = prepare_data_for_saving(train_dataset)
    processed_validation_data = prepare_data_for_saving(validation_dataset)
    print("Data processing completed.")

    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    print("Writing random samples to text files...")
    display_random_samples(processed_train_data, tokenizer, num_samples=10, output_file=os.path.join(output_dir, "train_random_samples.txt"))
    display_random_samples(processed_validation_data, tokenizer, num_samples=10, output_file=os.path.join(output_dir, "validation_random_samples.txt"))

    print("Saving processed dataset to disk...")
    processed_dataset = DatasetDict({
        "train": processed_train_data,
        "validation": processed_validation_data
    })
    processed_dataset.save_to_disk(output_dir)
    print(f"Processed dataset and random samples saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save tokenized NER dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to load (e.g., 'ai4privacy/pii-masking-400k')")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the processed dataset and sample files")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use (e.g., 'bert-base-cased', 'roberta-base')")
    
    args = parser.parse_args()
    print("Starting data preparation script with the following arguments:")
    print(f"  Dataset Name: {args.dataset_name}")
    print(f"  Tokenizer Name: {args.tokenizer_name}")
    print(f"  Max Length: {args.max_length}")
    print(f"  Output Path: {args.output_path}")
    main(args)
