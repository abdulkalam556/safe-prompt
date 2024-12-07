from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import torch

class LLAMAHandler:
    def __init__(self, model_path, tokenizer_path="meta-llama/Llama-2-7b-chat-hf"):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device=0 if torch.cuda.is_available() else -1,
            return_full_text=False,
        )
        return

    def generate_response(self, input_text):
        prompt = f"[INST] {input_text} [/INST]"
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=200,
            truncation=True
        )
        # print(sequences)
        # print(f"--------------------------------------------------------")
        return sequences[0]['generated_text'].strip()

        
# llama_path = "/blue/cnt5410/shaik.abdulkalam/cnt5410/pipeline/model/llama"
# llama_handler = LLAMAHandler(llama_path)
# message = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?'
# response = llama_handler.generate_response(message)
# print("response:")
# print(response)