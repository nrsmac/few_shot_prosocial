import os
import openai

openai.api_key_path = '.OPENAI_API_KEY'

prompt = f"I am going on a trip, and with me I am bringing: "

response = openai.Completion.create(
    model='text-davinci-002',
    prompt=prompt,
    temperature=0.6,
    max_tokens=500)

print(response['choices'][0]['text'])
