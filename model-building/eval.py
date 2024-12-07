import json
from datasets import load_from_disk
from evaluate import load
from transformers import AutoModelForTokenClassification
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load Dataset and Model
def load_data_and_model(dataset_path, model_path):
    print("Loading dataset and model...")
    dataset = load_from_disk(dataset_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return dataset, model

# Create DataLoader
def create_dataloader(dataset, batch_size=16):
    class NERDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(self.dataset[idx]['input_ids']),
                "attention_mask": torch.tensor(self.dataset[idx]['attention_mask']),
                "labels": torch.tensor(self.dataset[idx]['labels']),
            }

    print("Creating DataLoader...")
    ner_dataset = NERDataset(dataset)
    return DataLoader(ner_dataset, batch_size=batch_size)

# Run Model Inference
def run_inference(model, dataloader):
    print("Running inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        i = 0
        for batch in tqdm(dataloader, desc="Processing Batches"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=-1)

            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            i = i + 1
            if i > 500:
                break

    return all_predictions, all_labels

# Load Label Mappings
def load_label_mappings(test_labels_path, train_labels_path, label_mapping_path):
    print("Loading label mappings...")
    with open(test_labels_path, "r") as f:
        test_label_mapping = json.load(f)
    with open(train_labels_path, "r") as f:
        train_label_mapping = json.load(f)
    with open(label_mapping_path, "r") as f:
        label_mapping = json.load(f)

    id_to_label_test = {int(k): v for k, v in test_label_mapping["id_to_label"].items()}
    id_to_label_train = {int(k): v for k, v in train_label_mapping["id_to_label"].items()}

    return id_to_label_test, id_to_label_train, label_mapping

# Align Predictions and Labels (Handles List of Lists)
def align_predictions_and_labels(predictions, labels, id_to_label_train, id_to_label_test, ignore_index=-100):
    print("Aligning predictions and labels...")
    aligned_predictions = []
    aligned_labels = []

    for pred_seq, label_seq in tqdm(zip(predictions, labels), desc="Aligning Sequences", total=len(predictions)):
        seq_predictions = []
        seq_labels = []
        for pred, label in zip(pred_seq, label_seq):
            if label != ignore_index:  # Exclude padding
                seq_predictions.append(id_to_label_train[int(pred)])
                seq_labels.append(id_to_label_test[int(label)])
            else:
                break
        aligned_predictions.append(seq_predictions)
        aligned_labels.append(seq_labels)

    return aligned_predictions, aligned_labels

# Compute Metrics
def compute_metrics(aligned_predictions, aligned_labels):
    print("Computing metrics...")

    # Use Hugging Face's evaluate metric for seqeval
    metric = load("seqeval")

    results = metric.compute(predictions=aligned_predictions, references=aligned_labels, zero_division=0)

    print("\nEntity-Level Metrics (seqeval):")
    for entity, metrics in results.items():
        if isinstance(metrics, dict):  # Skip overall metrics
            print(f"{entity}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1']:.2f}, Support={metrics['number']}")

    print("\nOverall Metrics:")
    print(f"Precision: {results['overall_precision']:.2f}")
    print(f"Recall: {results['overall_recall']:.2f}")
    print(f"F1 Score: {results['overall_f1']:.2f}")
    print(f"Accuracy: {results['overall_accuracy']:.2f}")

# Main Function
def main():
    # Paths
    dataset_path = "/blue/cnt5410/shaik.abdulkalam/cnt5410/data/test_roberta/test"
    model_path = "/blue/cnt5410/shaik.abdulkalam/cnt5410/output/roberta-400k/sweep_cuoe79pj/model"
    test_labels_path = "/blue/cnt5410/shaik.abdulkalam/cnt5410/data/test_roberta/label_mappings.json"
    train_labels_path = "/blue/cnt5410/shaik.abdulkalam/cnt5410/data/RoBERTa-400k/label_mappings.json"
    label_mapping_path = "/blue/cnt5410/shaik.abdulkalam/cnt5410/data/test_roberta/test_label_mapping_400k.json"

    # Load dataset and model
    dataset, model = load_data_and_model(dataset_path, model_path)

    # Create DataLoader
    dataloader = create_dataloader(dataset)

    # Run inference
    all_predictions, all_labels = run_inference(model, dataloader)

    # Load label mappings
    id_to_label_test, id_to_label_train, label_mapping = load_label_mappings(
        test_labels_path, train_labels_path, label_mapping_path
    )

    # Align predictions and labels
    aligned_predictions, aligned_labels = align_predictions_and_labels(
        all_predictions, all_labels, id_to_label_train, id_to_label_test
    )

    # Compute metrics
    compute_metrics(aligned_predictions, aligned_labels)

# Execute Main
if __name__ == "__main__":
    main()
