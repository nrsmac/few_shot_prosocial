import csv
from tqdm import tqdm
from transformers import pipeline
from RandomProSocial import RandomProSocial

r = RandomProSocial("prosocial_dialog_v1/test.json")
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

results = [] 
samples = 10 
for i in tqdm(range(samples)):
    prompt, question, gold = r.get_prompt()
    y_hat = generator(f"{prompt}", do_sample=True, max_length=200)[0]['generated_text'][len(prompt):]
    print(f"{question}: {y_hat=} {gold=}") 
    results.append({"question":question, "gold":gold, "y_hat":y_hat})

out = f"test_test_gpt-neo_{samples}.csv"
csv_file = "Names.csv"
try:
    with open(out, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['question', 'gold', 'y_hat'])
        writer.writeheader()
        for data in results:
            writer.writerow(data)
except IOError:
    print("I/O error")