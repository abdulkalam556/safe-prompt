import argparse
import random
import os
import json
import ast
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
    
    # Initialize labels as "PAD" for all tokens
    labels = ["PAD"] * len(encoding.input_ids[0])

    for entity in entities:
        if isinstance(entity, dict) and 'entity_type' in entity:
            entity_label = entity['entity_type']
            if entity_label == 'O':
                continue
            start, end = entity['start_position'], entity['end_position']

            token_start_index = encoding.char_to_token(0, start)
            token_end_index = encoding.char_to_token(0, end - 1)

            if token_start_index is None or token_end_index is None:
                continue

            labels[token_start_index] = f"B-{entity_label}"
            for i in range(token_start_index + 1, token_end_index + 1):
                labels[i] = f"I-{entity_label}"

    # Replace labels for non-padding tokens based on attention mask
    for idx, mask in enumerate(encoding["attention_mask"][0]):
        if mask == 0:
            labels[idx] = "PAD"
        elif labels[idx] == "PAD":  # Non-entity tokens default to "O"
            labels[idx] = "O"
    
    # Convert labels to IDs
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
        text = self.data[idx]['full_text']
        entities = self.data[idx]['spans']
        encoded_data = tokenize_and_align_labels(text, entities, self.tokenizer, self.label_to_id, self.max_length)
        encoded_data["source_text"] = text
        encoded_data["privacy_mask"] = self.data[idx]['spans']
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
                if isinstance(entity, dict) and 'entity_type' in entity:
                    file.write(f"  - Label: {entity['entity_type']}, Value: {entity['entity_value']}, Start: {entity['start_position']}, End: {entity['end_position']}\n")
            
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
        data["source_text"].append(custom_dataset.data[i]['full_text'])
        data["privacy_mask"].append(custom_dataset.data[i]['spans'])
        
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


def convert_spans_to_objects(example):
    fixed_spans = []
    for span in example['spans']:
        try:
            # Use ast.literal_eval to safely evaluate strings with mixed quotes
            fixed_span = ast.literal_eval(span)
            fixed_spans.append(fixed_span)
        except (ValueError, SyntaxError) as e:
            # Log or handle problematic spans
            print(f"Skipping invalid span: {span} | Error: {e}")
            continue
    example['spans'] = fixed_spans
    return example


def main(args):
    print(f"Loading tokenizer '{args.tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print("Tokenizer loaded successfully.")

    print(f"Loading dataset '{args.dataset_name}'...")
    dataset = load_dataset(args.dataset_name)
    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].map(convert_spans_to_objects)
    test_data = dataset['test']
    print(f"Filtered training data to {len(test_data)} English samples.")

    print("Defining label mappings...")

    def extract_labels(dataset):
        labels = set()
        for split in ['train', 'validation', 'test']:
            for sample in dataset[split]:
                for span in sample['spans']:
                    # Ensure span is a dictionary with 'entity_type'
                    if isinstance(span, dict) and 'entity_type' in span:
                        if span['entity_type'] != 'O':  # Exclude 'O'
                            labels.add(f"B-{span['entity_type']}")
                            labels.add(f"I-{span['entity_type']}")
        return labels

    label_list = sorted(extract_labels(dataset))
    print("Label list:", label_list)

    # Add the "O" label for tokens outside any entity
    label_to_id = {label: i for i, label in enumerate(label_list)}
    label_to_id["O"] = len(label_to_id)

    # Create a reverse mapping from ID to label name
    global label_id_to_name
    label_id_to_name = {v: k for k, v in label_to_id.items()}

    print("Label mappings defined successfully.")
    print(f"Total labels (excluding 'O' entity type): {len(label_to_id)}")

    # Save label mappings
    save_label_mappings(label_to_id, args.output_path)

    print("Processing test split into CustomNERDataset format...")
    test_dataset = CustomNERDataset(test_data, tokenizer, label_to_id, max_length=args.max_length)
    processed_test_data = prepare_data_for_saving(test_dataset)
    print("Data processing completed.")

    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    print("Writing random samples to text files...")
    display_random_samples(processed_test_data, tokenizer, num_samples=10, output_file=os.path.join(output_dir, "test_random_samples.txt"))
    
    print("Saving processed dataset to disk...")
    processed_dataset = DatasetDict({
        "test": processed_test_data
    })
    processed_dataset.save_to_disk(output_dir)
    print(f"Processed dataset and random samples saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save tokenized NER dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to load (e.g., 'ai4privacy/pii-masking-400k')")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the processed dataset and sample files")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use (e.g., 'bert-base-cased', 'roberta-base')")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization")
    
    args = parser.parse_args()
    print("Starting data preparation script with the following arguments:")
    print(f"  Dataset Name: {args.dataset_name}")
    print(f"  Tokenizer Name: {args.tokenizer_name}")
    print(f"  Max Length: {args.max_length}")
    print(f"  Output Path: {args.output_path}")
    main(args)
