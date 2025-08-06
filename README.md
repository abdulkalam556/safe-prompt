# Safe Prompt: A Privacy-Preserving Framework for PII Anonymization in LLM Interactions

Safe Prompt is a framework designed to protect user privacy during interactions with Large Language Models (LLMs) like ChatGPT, Gemini, and Claude. It acts as a protective wrapper that identifies and anonymizes Personally Identifiable Information (PII) in user prompts before they are sent to an LLM, and then restores the original information in the LLM's response. This ensures user privacy without compromising the quality and context of the conversation.

## 📝 About The Project

The rapid adoption of LLMs has introduced significant privacy risks, as users may inadvertently share sensitive data that can be memorized by these models and potentially extracted through adversarial attacks. Regulations like GDPR require stringent protection of personal data, making it crucial to develop solutions that safeguard user privacy.

**Safe Prompt** addresses this challenge by:
* **Detecting PII** in real-time using fine-tuned Named Entity Recognition (NER) models.
* **Masking** sensitive data with context-preserving placeholders to maintain the quality of the LLM's response.
* **Demasking** the response to restore the original data, providing a seamless user experience.

The framework is designed to be scalable and independent of any specific LLM, making it a versatile solution for a wide range of applications.

## 🎤 Presentation

For a visual overview of the project, including our approach, architecture, and results, you can view the final presentation slides.

➡️ **[View the Project Presentation Slides](./presentation/safe-prompt.pdf)**

## ✨ Key Features

* **PII Detection:** Utilizes fine-tuned `BERT` and `RoBERTa` models to accurately identify a wide range of PII categories.
* **Context-Aware Anonymization:** Replaces detected PII not with generic tags, but with contextually similar data to preserve the prompt's semantic meaning.
* **Automated Masking & Demasking:** A three-stage pipeline automatically handles PII detection, masking, and demasking to ensure a smooth workflow.
* **Scalable & LLM-Agnostic:** Designed to work as a web extension or integrated layer across various LLM platforms.

## ⚙️ How It Works

The Safe Prompt framework operates through a three-stage pipeline: PII Detection, Anonymization and Masking, and Demasking.

1.  **Masking:** A fine-tuned NER model scans the user's input to detect PII. For example, in "My name is Azad," the name is identified.
2.  **Anonymization:** The detected PII is replaced with a contextually relevant placeholder. For instance, "My name is Azad" might become "My name is Jhon". A dictionary of these substitutions is stored temporarily.
3.  **Demasking:** The anonymized prompt is sent to the LLM. Once the response is received, the framework uses the stored dictionary to replace the placeholder with the original PII.

## 📦 Models and Datasets

### Models
We fine-tuned two powerful NER models for PII detection:
* **`bert-base-cased`**: Chosen for its effectiveness in handling case-sensitive data like names and emails.
* **`roberta-base`**: Excels at understanding nuanced contextual relationships, making it robust for broader PII detection tasks.

Hyperparameter tuning was performed using **Bayesian optimization** to find the best configurations, achieving a PII recall of nearly 95% during validation runs.

### Datasets
The models were trained and evaluated on large, publicly available datasets from Hugging Face:
* **Training:**
    * `ai4privacy/pii-masking-400k`: A large dataset with 17 common PII categories.
    * `ai4privacy/pii-masking-200k`: A more diverse dataset with 56 PII categories for stricter privacy needs, including `BITCOINADDRESS` and `EYECOLOR`.
* **Testing:**
    * `beki/privy`: A synthetic dataset with varied data formats (JSON, SQL, HTML) used to test the models' generalization capabilities.

## 📊 Results and Analysis

The pipeline's performance was evaluated by comparing the LLM's response to an original prompt with its response to a masked prompt using metrics like BLEU, ROUGE, and BERTScore.

### Model Performance
The models were evaluated based on their ability to correctly identify PII. The final results for accuracy, F1-score, precision, and recall on the test data are summarized below.

| Model | Accuracy | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **BERT-200k** | 0.78 | 0.22 | 0.14 | 0.44 |
| **BERT-400k** | 0.82 | 0.19 | 0.13 | 0.40 |
| **ROBERTa-200k** | 0.86 | 0.24 | 0.16 | 0.42 |
| **ROBERTa-400k** | 0.84 | 0.21 | 0.15 | 0.43 |

### Pipeline Evaluation
The quality of the final, demasked output was measured against the original, non-masked output. [cite_start]The high **BERTScore (around 0.90)** across all configurations indicates a strong semantic alignment between the original and privacy-protected responses[cite: 530, 542].

| Model | BLEU | ROUGE-1 | ROUGE-L | BERTScore |
| :--- | :---: | :---: | :---: | :---: |
| **BERT-200k** | 0.26 | 0.50 | 0.36 | 0.89 |
| **BERT-400k** | 0.28 | 0.52 | 0.38 | 0.89 |
| **ROBERTa-200k** | 0.30 | 0.54 | 0.41 | 0.90 |
| **ROBERTa-400k** | 0.30 | 0.54 | 0.40 | 0.90 |

## 🚀 learning curves and other evaluation plots:

All plots for training and evalutaion are available on GitHub.

**https://github.com/abdulkalam556/safe-prompt-plots/tree/main/plots** 

## 🚀 tokens for sample of texts:

some of samples showcasing all of the data preparation processes are available on GitHub.

**https://github.com/abdulkalam556/safe-prompt-plots/tree/main/random-texts** 
