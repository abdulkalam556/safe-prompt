import os
import json
import torch
import argparse
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from evaluate import load as load_metric
import matplotlib.pyplot as plt

#global variables
label_to_id = {}
id_to_label = {}

# Function to create output directories dynamically
def create_output_dirs(output_dir):
    model_output_dir = os.path.join(output_dir, "model")
    plots_output_dir = os.path.join(output_dir, "plots")
    logs_output_dir = os.path.join(output_dir, "logs")

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(plots_output_dir, exist_ok=True)
    os.makedirs(logs_output_dir, exist_ok=True)

    return model_output_dir, plots_output_dir, logs_output_dir

def load_datasets(data_dir, model_name, dataset_type):
    """Loads tokenized datasets based on model and dataset type."""
    if "bert" in model_name.lower():
        model_name = "Bert"
    elif "roberta" in model_name.lower():
        model_name = "RoBERTa"
    else:
        print(f"Provided incorrect model name: {model_name}")
        return None, None
    
    dataset_folder = f"{model_name}-{dataset_type}"
    dataset_path = os.path.join(data_dir, dataset_folder)

    print(f"Loading datasets from {dataset_path}...")
    train_dataset = load_from_disk(os.path.join(dataset_path, "train"))
    val_dataset = load_from_disk(os.path.join(dataset_path, "validation"))
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    return train_dataset, val_dataset

def load_label_mappings(data_dir, model_name, dataset_type):
    """Loads label mappings from the specified dataset directory."""
    if "bert" in model_name.lower():
        model_name = "Bert"
    elif "roberta" in model_name.lower():
        model_name = "RoBERTa"
    else:
        print(f"Provided incorrect model name: {model_name}")
        return

    dataset_folder = f"{model_name}-{dataset_type}"
    label_mapping_path = os.path.join(data_dir, dataset_folder, "label_mappings.json")
    print(f"Loading Label Mappings from {label_mapping_path}...")

    with open(label_mapping_path, "r") as f:
        label_mappings = json.load(f)
    
    # Convert id_to_label keys to integers
    id_to_label = {int(k): v for k, v in label_mappings["id_to_label"].items()}
    label_to_id = label_mappings["label_to_id"]

    return label_to_id, id_to_label

def initialize_model(args, num_labels):
    """Initializes the model and training arguments based on the provided arguments."""
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=num_labels)
    exp_directory = os.path.join(args.output_dir, args.experiment_id)

    training_args = TrainingArguments(
        output_dir=exp_directory,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.scheduler_type,
        logging_dir=os.path.join(exp_directory, "logs"),
        logging_steps=500,
        save_total_limit=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        load_best_model_at_end=True if args.load_best_model_at_end else False,
        metric_for_best_model="PII_recall",  # or "PII_f1" for balanced
        greater_is_better=True,
        # fp16=True,
        dataloader_num_workers=4,
    )

    return model, training_args

def compute_metrics(pred):
    global id_to_label
    metric = load_metric("seqeval")
    predictions, labels = pred
    predictions = predictions.argmax(axis=2)

    # Convert prediction and label IDs to multi-class labels
    true_labels = [
        [id_to_label[label] for label in label_row if label != -100]
        for label_row in labels
    ]
    true_predictions = [
        [id_to_label[pred] for pred, label in zip(pred_row, label_row) if label != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]

    # Calculate multi-class metrics
    results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)

    # Binary conversion: "O" vs. "Not-O" (PII)
    true_labels_binary = [
        ["PII" if label != "O" else "O" for label in row]
        for row in true_labels
    ]
    pred_labels_binary = [
        ["PII" if pred != "O" else "O" for pred in row]
        for row in true_predictions
    ]

    # Calculate binary metrics using metric.compute() for "O" vs "PII"
    binary_results = metric.compute(predictions=pred_labels_binary, references=true_labels_binary, zero_division=0)

    return {
        "PII_recall": binary_results["overall_recall"],
        "PII_precision": binary_results["overall_precision"],
        "PII_f1": binary_results["overall_f1"],
        "PII_accuracy": binary_results["overall_accuracy"],
        "multi_class_overall_f1": results["overall_f1"],
        "multi_class_overall_accuracy": results["overall_accuracy"],
        "multi_class_results": results  # Detailed class-wise metrics
    }


def extract_loss_logs(log_history):
    """Separates training and evaluation losses from `log_history`."""
    print("Extracting training and evaluation loss logs...")
    train_loss = []
    eval_loss = []
    steps_train = []
    steps_eval = []

    for entry in log_history:
        # Check and extract training loss
        if "loss" in entry and "eval_loss" not in entry:  
            train_loss.append(entry["loss"])
            steps_train.append(entry["step"])
            # print(f"Training - Step: {entry['step']}, Loss: {entry['loss']}")
        
        # Check and extract evaluation loss
        elif "eval_loss" in entry:  
            eval_loss.append(entry["eval_loss"])
            steps_eval.append(entry["step"])
            # print(f"Evaluation - Step: {entry['step']}, Eval Loss: {entry['eval_loss']}")

    print(f"Extracted {len(train_loss)} training loss entries and {len(eval_loss)} evaluation loss entries.")
    return train_loss, eval_loss, steps_train, steps_eval

def plot_and_save_metrics(logs, plots_output_dir):
    """Plots and saves training/validation loss over steps with log scale on y-axis."""
    print("Starting to plot training and validation losses over steps...")
    
    # Extract training and evaluation loss
    train_loss, eval_loss, steps_train, steps_eval = extract_loss_logs(logs)

    # Check if there’s data to plot
    if not train_loss or not eval_loss:
        print("Warning: No data found for plotting. Ensure log history is populated.")
        return

    # Plot with log scale on the y-axis
    plt.figure(figsize=(10, 6))
    plt.plot(steps_train, train_loss, label="Training Loss", marker="o")
    plt.plot(steps_eval, eval_loss, label="Validation Loss", marker="x")
    plt.xlabel("Steps")
    plt.ylabel("Loss (Log Scale)")
    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.title("Training and Validation Loss (Log Scale)")
    plt.legend()
    print("Plotting completed. Now saving the plot...")
    
    # Save the plot
    output_path = os.path.join(plots_output_dir, "loss_plot_log_scale.png")
    os.makedirs(plots_output_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved log-scale plot to {output_path}")

def train_and_evaluate(args):
    # Step 3: Set up experiment directory
    print("\n--- Setting up experiment directory ---")
    exp_output_dir = os.path.join(args.output_dir, args.experiment_id)
    os.makedirs(exp_output_dir, exist_ok=True)
    print(f"Experiment output directory: {exp_output_dir}")
    
    # Step 1: Set up output directories
    print("\n--- Setting up output directories ---")
    model_output_dir, plots_output_dir, logs_output_dir = create_output_dirs(exp_output_dir)
    print(f"Model output directory: {model_output_dir}")
    print(f"Plots output directory: {plots_output_dir}")
    print(f"Logs output directory: {logs_output_dir}")
    
    # Step 2: Load datasets and label mappings
    print("\n--- Loading datasets and label mappings ---")
    train_dataset, val_dataset = load_datasets(args.data_dir, args.model_name, args.dataset_type)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    global label_to_id, id_to_label
    label_to_id, id_to_label = load_label_mappings(args.data_dir, args.model_name, args.dataset_type)
    num_labels = len(label_to_id)
    print(f"Number of labels: {num_labels}")

    # Step 4: Save configuration
    print("\n--- Saving experiment arguments ---")
    config_dict = vars(args)
    with open(os.path.join(exp_output_dir, "arguments.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    print("Configuration saved successfully.")

    # Step 5: Model Initialization
    print("\n--- Initializing model and training arguments ---")
    device_count = torch.cuda.device_count()
    print(f"Number of devices (GPUs) available: {device_count}")
    
    effective_batch_size = args.batch_size * device_count * args.gradient_accumulation_steps
    print(f"Calculated effective batch size: {effective_batch_size}")
    
    # Calculate eval_steps to evaluate 3 times per epoch
    steps_per_epoch = len(train_dataset) // effective_batch_size
    # eval_steps = max(1, steps_per_epoch // 3)
    print(f"Steps per epoch: {steps_per_epoch}") #, Evaluation every {eval_steps} steps")

    model, training_args = initialize_model(args, num_labels)
    print("Model and training arguments initialized successfully.")
    
    print("\n--- Saving experiment configuration ---")
    training_args_path = os.path.join(exp_output_dir, "training_args.json")
    with open(training_args_path, "w") as f:
        json.dump(training_args.to_dict(), f, indent=4)  # Convert to dict before saving
    print(f"Training arguments saved to {training_args_path}")

    # Step 6: Initialize Trainer
    print("\n--- Initializing Trainer ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    print("Trainer initialized successfully.")

    # Step 7: Train and Evaluate
    print("\n--- Starting Training ---")
    trainer.train()
    print("Training completed successfully.")
    
    print("\n--- Evaluating Model ---")
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")

    # Step 8: Log results
    print("\n--- Logging Evaluation Results ---")
    with open(os.path.join(exp_output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    print("Evaluation results logged successfully.")

    # Step 9: Save Model and Plot Metrics
    print("\n--- Saving Model and Plotting Metrics ---")
    trainer.save_model(model_output_dir)
    print(f"Model saved to {model_output_dir}")
    
    # Save log_history to a JSON file
    log_history_path = os.path.join(exp_output_dir, "log_history.json")
    with open(log_history_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)
    print(f"log_history saved to {log_history_path}")

    # Collect logs for loss tracking (optional for monitoring)
    training_logs = trainer.state.log_history  # Collect the log history
    plot_and_save_metrics(training_logs, plots_output_dir)
    print("Training and evaluation process completed successfully.")
    
    metrics = {
        "PII_f1": eval_results["eval_PII_f1"],
        "PII_recall": eval_results["eval_PII_recall"],
        "PII_precision": eval_results["eval_PII_precision"],
        "PII_accuracy": eval_results["eval_PII_accuracy"],
        "f1": eval_results["eval_multi_class_results"]["overall_f1"],
        "recall": eval_results["eval_multi_class_results"]["overall_recall"],
        "precision": eval_results["eval_multi_class_results"]["overall_precision"],
        "accuracy": eval_results["eval_multi_class_results"]["overall_accuracy"],
    }
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script with manual hyperparameter inputs")

    # Paths
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory for saving outputs (models, plots, logs)")

    # Model, tokenizer, and dataset type settings
    parser.add_argument("--model_name", type=str, default="bert-base-cased", help="Name of the model checkpoint")
    parser.add_argument("--dataset_type", type=str, required=True, help="Dataset type (e.g., '200k' or '400k')")

    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs for training")
    parser.add_argument("--weight_decay", type=float, required=True, help="Weight decay for regularization")
    parser.add_argument("--dropout_rate", type=float, required=True, help="Dropout rate")
    parser.add_argument("--warmup_steps", type=int, required=True, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps to mimic larger batch sizes")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for gradient clipping")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Use early stopping based on validation performance")
    parser.add_argument("--scheduler_type", type=str, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help="Scheduler type for learning rate")

    # Experiment settings
    parser.add_argument("--experiment_id", type=str, required=True, help="Unique identifier for the experiment")

    args = parser.parse_args()
    train_and_evaluate(args)
