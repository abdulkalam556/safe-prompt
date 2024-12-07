import wandb
from train import train_and_evaluate  # Assuming train_and_evaluate is your main training function
from argparse import Namespace

# Objective function for training
def objective():
    # Initialize W&B for this run
    wandb.init()

    # Define training arguments based on W&B sweep configuration
    config = wandb.config
    args = Namespace(
        data_dir="../data",
        output_dir="../output/bert-200k",
        model_name="bert-base-cased",
        dataset_type="200k",
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        dropout_rate=config.dropout_rate,
        warmup_steps=config.warmup_steps,
        gradient_accumulation_steps=1,
        scheduler_type=config.scheduler_type,
        load_best_model_at_end=True,
        max_grad_norm=1.0,
        experiment_id=f"sweep_{wandb.run.id}"
    )

    # Run training and evaluation
    metrics = train_and_evaluate(args)

    # Log metrics to W&B
    wandb.log(metrics)

# Main function to set up W&B Sweep and agent
def main():
    # Log into W&B
    with open("wandb_key.txt", "r") as f:
        wandb_key = f.read().strip()

    wandb.login(key=wandb_key)

    # Define the sweep configuration
    sweep_configuration = {
        "method": "bayes",  # Using Bayesian Optimization
        "metric": {"goal": "maximize", "name": "PII_recall"},
        "parameters": {
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-5},
            "batch_size": {"values": [8, 16]},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
            "warmup_steps": {"distribution": "int_uniform", "min": 100, "max": 500},
            "scheduler_type": {"values": ["linear", "cosine"]},
            "dropout_rate": {"distribution": "uniform", "min": 0.1, "max": 0.3},
            "num_epochs": {"value": 5}  # Fixed number of epochs
        },
    }

    # Start the W&B Sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="pii_detection_sweeps_bert_200k")

    # Start the sweep agent with the specified objective function
    wandb.agent(sweep_id, function=objective, count=10)

# Execute main
if __name__ == "__main__":
    main()
