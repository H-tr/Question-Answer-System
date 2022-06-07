import os
import json
import openai

with open("GPT_SECRET_KEY.json") as f:
    data = json.load(f)

openai.api_key = data["API_KEY"]

start_sequence = "\nA: "
restart_sequence = "\n\nQ: "

def gpt3(prompt, engine="text-davinci-002", response_length=64,
         temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0,
         start_text='', restart_text='', stop_seq=["\n"]):
    response = openai.Completion.create(
        engine=engine,
        prompt = prompt + start_text,
        temperature=0.25,
        max_tokens=response_length,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop_seq
    )
    answer = response.choices[0]['text']
    new_prompt = prompt + start_text + answer + restart_text
    return answer, new_prompt

def chat():
    prompt = "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ:"
    while True:
        prompt += input("You: ")
        answer, prompt = gpt3(prompt,
                              temperature=0.9,
                              frequency_penalty=1,
                              presence_penalty=1,
                              start_text=start_sequence,
                              restart_text=restart_sequence,
                              stop_seq=["\n"])
        print("GPT-3:" + answer)

if __name__ == '__main__':
    chat()
