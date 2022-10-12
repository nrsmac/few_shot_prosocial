import csv
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from RandomProSocial import RandomProSocial

r = RandomProSocial("prosocial_dialog_v1/test.json")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

results = [] 
samples = 10 
for i in tqdm(range(samples)):
    prompt, question, gold = r.get_prompt()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    calculated_no_tokens = math.ceil((len(prompt) - prompt.count(' ')) / 4)
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.1,
        max_length=calculated_no_tokens+5)

    y_hat = tokenizer.batch_decode(gen_tokens)[0][len(prompt):]

    print(f"{question}: \n{y_hat=},{gold=}") 
    results.append({"question":question, "gold":gold, "y_hat":y_hat})

out = f"test_test_gpt-j_{samples}.csv"
try:
    with open(out, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['question', 'gold', 'y_hat'])
        writer.writeheader()
        for data in results:
            writer.writerow(data)
except IOError:
    print("I/O error")