import random
from tqdm import tqdm
from datasets import load_dataset
from piipredictor import PIIPredictor
from piimasker import PIIMasker
from llamahandler import LLAMAHandler
import evaluate

def run_pipeline(sentence, piipredictor, piimasker, llama_handler):
    # Generate privacy mask
    privacy_mask = piipredictor.generate_privacy_mask(sentence)

    # Mask the sentence
    masked_sentence, replacement_dict = piimasker.mask_sentence(sentence, privacy_mask)

    # Generate responses with additional error handling
    pipeline_response = None
    baseline_response = None
    try:
        pipeline_response = llama_handler.generate_response(masked_sentence)
    except Exception as e:
        print(f"Error generating pipeline response for masked sentence: {masked_sentence}. Error: {e}")

    try:
        baseline_response = llama_handler.generate_response(sentence)
    except Exception as e:
        print(f"Error generating baseline response for sentence: {sentence}. Error: {e}")

    if pipeline_response is None or baseline_response is None:
        raise ValueError("Failed to generate both pipeline and baseline responses.")

    # Unmask the pipeline response
    unmasked_pipeline_response = piimasker.unmask_sentence(pipeline_response, replacement_dict)

    return sentence, masked_sentence, unmasked_pipeline_response, baseline_response

def evaluate_metrics(pipeline_responses, baseline_responses):
    # Initialize evaluation metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bert_scorer = evaluate.load("bertscore")
    
    results = {}

    # BLEU
    bleu_score = bleu.compute(predictions=pipeline_responses, references=[[ref] for ref in baseline_responses])
    results["bleu"] = bleu_score["bleu"]

    # ROUGE
    rouge_score = rouge.compute(predictions=pipeline_responses, references=baseline_responses)
    results["rouge1"] = rouge_score["rouge1"]
    results["rougeL"] = rouge_score["rougeL"]

    # BERTScore
    bert_score = bert_scorer.compute(predictions=pipeline_responses, references=baseline_responses, lang="en")
    results["bert_score"] = sum(bert_score["f1"]) / len(bert_score["f1"])

    return results

def process_dataset(dataset, piipredictor, piimasker, llama_handler):
    results = []

    for entry in tqdm(dataset, desc='running pipeline over dataset'):
        sentence = entry["source_text"]
        try:
            sentence, masked_sentence, pipeline_response, baseline_response = run_pipeline(sentence, piipredictor, piimasker, llama_handler)
            results.append({
                "source_text": sentence, 
                "masked_sentence": masked_sentence,
                "pipeline_response": pipeline_response, 
                "baseline_response": baseline_response
            })
        except Exception as e:
            print(f"Error processing sentence: {sentence}. Skipping this example. Error: {e}")

    return results

def main():
    # Load and sample the dataset
    dataset = load_dataset("ai4privacy/pii-masking-300k", split="train")
    dataset = dataset.filter(lambda x: x['language'] == 'en')
    sampled_dataset = dataset.shuffle(seed=42).select(range(1000))
    
    # Configuration for PIIPredictor
    llama_model_path = "/blue/cnt5410/shaik.abdulkalam/cnt5410/pipeline/model/llama" 
    configs = [
        {
            "model_path": "/blue/cnt5410/shaik.abdulkalam/cnt5410/output/bert-200k/sweep_y0a5xakv/model",
            "tokenizer_name": "bert-base-cased",
            "label_mapping_file_path": "/blue/cnt5410/shaik.abdulkalam/cnt5410/data/Bert-200k/label_mappings.json",
        },
        {
            "model_path": "/blue/cnt5410/shaik.abdulkalam/cnt5410/output/bert-400k/exp_lr3.2310505704990376e-05_bs8_wd0.0005633277661591784_ws479_dr0.27239816182045773_schedulerlinear/model",
            "tokenizer_name": "bert-base-cased",
            "label_mapping_file_path": "/blue/cnt5410/shaik.abdulkalam/cnt5410/data/Bert-400k/label_mappings.json",
        },
        {
            "model_path": "/blue/cnt5410/shaik.abdulkalam/cnt5410/output/roberta-400k/sweep_cuoe79pj/model",
            "tokenizer_name": "roberta-base",
            "label_mapping_file_path": "/blue/cnt5410/shaik.abdulkalam/cnt5410/data/RoBERTa-400k/label_mappings.json",
        },
        {
            "model_path": "/blue/cnt5410/shaik.abdulkalam/cnt5410/output/roberta-200k/sweep_3c5czcbk/model",
            "tokenizer_name": "roberta-base",
            "label_mapping_file_path": "/blue/cnt5410/shaik.abdulkalam/cnt5410/data/RoBERTa-200k/label_mappings.json",
        }
    ]

    piimasker = PIIMasker()
    llama_handler = LLAMAHandler(llama_model_path)

    for i, config in tqdm(enumerate(configs), desc="configs"):
        # Initialize components
        piipredictor = PIIPredictor(config)
        
        # Process the dataset
        responses = process_dataset(sampled_dataset, piipredictor, piimasker, llama_handler)

        # Separate pipeline and baseline responses
        pipeline_responses = [r["pipeline_response"] for r in responses]
        baseline_responses = [r["baseline_response"] for r in responses]

        # Evaluate metrics
        metrics = evaluate_metrics(pipeline_responses, baseline_responses)
        print("Metrics:", metrics)

        # Save results
        with open(f"results_{i}.json", "w") as f:
            import json
            json.dump({"metrics": metrics, "responses": responses}, f, indent=4)

if __name__ == "__main__":
    main()
