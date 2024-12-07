import json
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

class PIIPredictor:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model(config["model_path"])
        self.tokenizer = self.load_tokenizer(config["tokenizer_name"])
        self.id_to_label = self.load_id_to_label(config["label_mapping_file_path"])

    def load_model(self, model_path):
        """
        Load and return the pre-trained model.
        """
        print(f"Loading model from {model_path}...")
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        # device=0 if torch.cuda.is_available() else -1
        # model.to(device)
        model.eval()  # Set model to evaluation mode
        print("Model loaded successfully.")
        return model

    def load_tokenizer(self, tokenizer_name):
        """
        Load and return the tokenizer.
        """
        print(f"Loading tokenizer from {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print("Tokenizer loaded successfully.")
        return tokenizer

    def load_id_to_label(self, label_mapping_file_path):
        """
        Load and return the id-to-label mapping.
        """
        print(f"Loading training label mappings from {label_mapping_file_path}...")
        with open(label_mapping_file_path, "r") as f:
            train_label_mappings = json.load(f)

        id_to_label = train_label_mappings["id_to_label"]
        print(f"Training label mappings loaded successfully with {len(id_to_label)} labels.")

        id_to_label = {int(k): v for k, v in id_to_label.items()}

        return id_to_label

    def generate_privacy_mask(self, sentence):
        # Tokenize with offset mapping
        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,  # Enables offset mapping
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze().tolist()
        attention_mask = encoding["attention_mask"].squeeze().tolist()
        offsets = encoding["offset_mapping"].squeeze().tolist()

        # Run model inference
        with torch.no_grad():
            outputs = self.model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

        # Decode tokens and labels
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        # print(tokens[:50])
        bio_labels = [self.id_to_label.get(label, "O") for label in predictions]

        # Generate privacy mask
        privacy_mask = []
        current_entity = None

        for idx, (label, offset) in enumerate(zip(bio_labels, offsets)):
            if attention_mask[idx] == 0 or offset == (0, 0):  # Skip padded and special tokens
                continue

            if label.startswith("B-"):
                if current_entity:
                    privacy_mask.append(current_entity)
                current_entity = {
                    "label": label[2:],
                    "start": offset[0],  # Start from offset mapping
                    "end": offset[1],  # End from offset mapping
                    "value": sentence[offset[0]:offset[1]]  # Extract value from original text
                }
            elif label.startswith("I-") and current_entity and current_entity["label"] == label[2:]:
                current_entity["end"] = offset[1]  # Extend end
                current_entity["value"] += sentence[offset[0]:offset[1]]  # Append to value
            elif current_entity:
                privacy_mask.append(current_entity)
                current_entity = None

        if current_entity:
            privacy_mask.append(current_entity)
        
        # Update the `value` based on the final `start` and `end` indices
        for entity in privacy_mask:
            entity["value"] = sentence[entity["start"]:entity["end"]]
            
        return privacy_mask


# Initialize the predictor
# config = {
#     "model_path": "/blue/cnt5410/shaik.abdulkalam/cnt5410/output/bert-200k/sweep_y0a5xakv/model",
#     "tokenizer_name": "bert-base-cased",
#     "label_mapping_file_path": "/blue/cnt5410/shaik.abdulkalam/cnt5410/data/Bert-200k/label_mappings.json"
# }

# config = {
#     "model_path": "/blue/cnt5410/shaik.abdulkalam/cnt5410/output/bert-400k/exp_lr3.2310505704990376e-05_bs8_wd0.0005633277661591784_ws479_dr0.27239816182045773_schedulerlinear/model",
#     "tokenizer_name": "bert-base-cased",
#     "label_mapping_file_path": "/blue/cnt5410/shaik.abdulkalam/cnt5410/data/Bert-400k/label_mappings.json"
# }
# predictor = PIIPredictor(config)
# # sentence = "Jhon lives in New York."
# # Test with a sample sentence
# sentence = "Addressing Dariush's condition involves working through unresolved conflicts. Regular therapy sessions logged at 10 using ZIPCODE LE14 and update dasnim.lämmlein48@outlook.com. TAXNUM 72676 32249 compliance is mandatory."
# print(sentence)
# # Generate privacy mask
# privacy_mask = predictor.generate_privacy_mask(sentence)

# # Print results
# print("Privacy Mask:", privacy_mask)