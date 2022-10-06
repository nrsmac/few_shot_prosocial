import json
import random

class RandomProSocial():
    def __init__(self, path) -> None:
        with open(path, "r") as f:
            data = json.load(f)

        texts = [entry[0]['text'] for entry in data]
        labels = [entry[0]['safety_label'] for entry in data]

        self.texts_with_label = list(zip(texts, labels))
        random.shuffle(self.texts_with_label)