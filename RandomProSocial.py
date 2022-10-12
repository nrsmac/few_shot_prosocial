import json
import random

class RandomProSocial():
    def __init__(self, path) -> None:
        with open(path, "r") as f:
            data = json.load(f)

        texts = [entry[0]['text'] for entry in data]
        labels = [entry[0]['safety_label'] for entry in data]
        self.texts_with_label = list(zip(texts, labels))
        self.key = {
            '__casual__':1,
            '__possibly_needs_caution__':2,
            '__probably_needs_caution__':3,
            '__needs_caution__':4,
            '__needs_intervention__':5
        }
        random.shuffle(self.texts_with_label)

    def get_prompt(self, template="qa_nancy"):
        
        with open(f"templates/{template}.txt") as f:
            template_string = f.read()

        template_string = template_string.replace("\n", " ")

        question, gold = self.texts_with_label.pop(0)
        prompt = template_string.replace("[PROMPT]", question)
        return prompt, question, gold

if __name__ == '__main__':
    r = RandomProSocial('prosocial_dialog_v1/valid.json')
    # print(f"labels = {set([i[1] for i in r.texts_with_label])}")
    _, statement, gold = r.get_prompt() 
    print(statement)
    print(gold)