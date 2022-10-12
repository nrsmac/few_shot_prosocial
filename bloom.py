from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b3")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b3")

prompt = "It was a dark and stormy night"
result_length = 50
inputs = tokenizer(prompt, return_tensors="pt")

# Sampling Top-k + Top-p
print(tokenizer.decode(model.generate(inputs["input_ids"],
                       max_length=result_length, 
                       do_sample=True, 
                       top_k=50, 
                       top_p=0.9
                      )[0]))
