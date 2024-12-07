import os
import json
import argparse
import subprocess

def find_top_models(input_dir, num_top_models=3):
    """
    Identifies the top models based on eval_PII_recall from eval_results.json files in a directory.

    Args:
        input_dir (str): Path to the directory containing model run folders.
        num_top_models (int): Number of top models to select.

    Returns:
        list of tuples: A list of tuples containing model path and eval_PII_recall score for top models.
    """
    model_scores = []

    for run_dir in os.listdir(input_dir):
        run_path = os.path.join(input_dir, run_dir)
        eval_results_path = os.path.join(run_path, "eval_results.json")
        model_path = os.path.join(run_path, "model")

        if os.path.isfile(eval_results_path) and os.path.isdir(model_path):
            with open(eval_results_path, "r") as f:
                eval_results = json.load(f)
                eval_pii_recall = eval_results.get("eval_PII_recall", 0)
                model_scores.append((run_path, model_path, eval_pii_recall))

    model_scores.sort(key=lambda x: x[2], reverse=True)

    return model_scores[:num_top_models]

def generate_evaluation_commands(input_root, output_root, batch_size=8):
    """
    Generates evaluation commands for the top models across all input directories.

    Args:
        input_root (str): Path to the root directory containing bert-400k, bert-200k, etc.
        output_root (str): Path to the output directory where test results will be saved.
        batch_size (int): Batch size for evaluation.

    Returns:
        list of str: A list of evaluation commands.
    """
    commands = []

    for folder_name in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder_name)

        if "bert" in folder_name.lower():
            model_name = "BERT"
            tokenizer_name = "bert-base-cased"
        elif "roberta" in folder_name.lower():
            model_name = "RoBERTa"
            tokenizer_name = "roberta-base"
        else:
            print(f"Unknown model in folder: {folder_name}, skipping.")
            continue

        dataset_type = "400k" if "400k" in folder_name.lower() else "200k"
        output_dir = os.path.join(output_root, "test", folder_name)

        os.makedirs(output_dir, exist_ok=True)

        top_models = find_top_models(folder_path)

        for rank, (run_path, model_path, score) in enumerate(top_models, start=1):
            run_output_dir = os.path.join(output_dir, str(rank))
            os.makedirs(run_output_dir, exist_ok=True)

            command = (
                f"python evaluate_model.py "
                f"--model_path {model_path} "
                f"--tokenizer_name {tokenizer_name} "
                f"--model_name {model_name} "
                f"--data_dir ../data "
                f"--dataset_type {dataset_type} "
                f"--output_dir {run_output_dir} "
                f"--batch_size {batch_size}"
            )
            commands.append(command)

    return commands

def run_commands(commands):
    """
    Executes a list of shell commands sequentially.

    Args:
        commands (list): A list of shell commands to execute.
    """
    print(f"Running {len(commands)} commands...")
    for idx, command in enumerate(commands, start=1):
        print(f"\nExecuting command {idx}/{len(commands)}: {command}")
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Command {idx} executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing command {idx}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and run evaluation commands for top models.")
    parser.add_argument("--input_root", type=str, required=True, help="Root directory of model run folders.")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory to save test outputs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")

    args = parser.parse_args()

    # Generate commands for evaluation
    commands = generate_evaluation_commands(
        input_root=args.input_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
    )

    # Save commands to a file for reference
    commands_file = os.path.join(args.output_root, "evaluation_commands.txt")
    with open(commands_file, "w") as f:
        for command in commands:
            f.write(command + "\n")
    print(f"Evaluation commands saved to: {commands_file}")

    # Execute the commands
    run_commands(commands)
