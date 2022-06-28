from tokenize import String
import openai

class DocumentReader:
    def __init__(self) -> None:
        pass

    def chat(self, prompt, engine='text-davinci-002', response_length=64,
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

    def create_answer(self, question, doc_list, examples_context="Larry Page and Sergey Brin.", 
        search_model="davinci", model="curie") -> String:
        response = openai.Answer.create(
            search_model = search_model,
            model = model,
            question = question,
            documents = doc_list,
            examples_context = examples_context,
            examples=[["Who founded Google?"]],
            max_tokens=30,
            stop=["\n", "<|endoftext|>"]
        )
        answer = response["answers"][0]
        return answer