import os
import json
import openai

with open("GPT_SECRET_KEY.json") as f:
    data = json.load(f)

openai.api_key = data["API_KEY"]

#  start_sequence = "\nAI: "
#  restart_sequence = "\nHuman: "

#  def gpt3(prompt, engine="text-davinci-002", response_length=64,
         #  temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0,
         #  start_text='', restart_text='', stop_seq=["\n"]):
    #  response = openai.Completion.create(
        #  engine=engine,
        #  prompt = prompt + start_text,
        #  temperature=0.25,
        #  max_tokens=response_length,
        #  top_p=top_p,
        #  frequency_penalty=frequency_penalty,
        #  presence_penalty=presence_penalty,
        #  stop=stop_seq
    #  )
    #  answer = response.choices[0]['text']
    #  new_prompt = prompt + start_text + answer + restart_text
    #  return answer, new_prompt

def gpt3(prompt, engine='text-davinci-002', response_length=64,
         temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0,
         start_text='', restart_text='', stop_seq=[]):
    response = openai.Completion.create(
        prompt=prompt + start_text,
        engine=engine,
        max_tokens=response_length,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop_seq,
    )
    answer = response.choices[0]['text']
    new_prompt = prompt + start_text + answer + restart_text
    return answer, new_prompt

