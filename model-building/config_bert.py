import os
import optuna
import wandb
from train import train_and_evaluate
from argparse import Namespace

# Load W&B API key securely from file
with open("wandb_key.txt", "r") as f:
    wandb_key = f.read().strip()

wandb.login(key=wandb_key)

def objective(trial):
    # Refined search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 1e-2)
    warmup_steps = trial.suggest_int("warmup_steps", 100, 500)
    scheduler_type = trial.suggest_categorical("scheduler_type", ["linear", "cosine"])
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.3)

    # Experiment ID for tracking in W&B
    experiment_id = f"exp_lr{learning_rate}_bs{batch_size}_wd{weight_decay}_ws{warmup_steps}_dr{dropout_rate}_scheduler{scheduler_type}"

    # Define arguments
    args = Namespace(
        data_dir="../data",
        output_dir="../output/bert-400k",
        model_name="bert-base-cased",
        dataset_type="400k",
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=5,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        scheduler_type=scheduler_type,
        gradient_accumulation_steps=1,
        early_stopping=True,
        load_best_model_at_end=True,
        dropout_rate=dropout_rate,
        max_grad_norm=1.0,
        experiment_id=experiment_id,
    )

    # Initialize W&B run for this trial
    with wandb.init(project="pii_detection_optimization", config=args.__dict__) as run:
        metrics = train_and_evaluate(args)
        wandb.log(metrics)
        return metrics["PII_recall"]  # or "PII_f1" 

# Run Bayesian Optimization with Optuna
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters found:", study.best_params)
    print("Best score:", study.best_value)
