from QAsystemGPT import gpt3

def chat():
    prompt = """Human: Hey, how are you doing?
AI: I'm good! What would you like to chat about?
Human:"""
    while True:
        prompt += input('You: ')
        answer, prompt = gpt3(prompt,
                              temperature=0.9,
                              frequency_penalty=1,
                              presence_penalty=1,
                              start_text='\nAI:',
                              restart_text='\nHuman: ',
                              stop_seq=['\nHuman:', '\n'])
        print('GPT-3:' + answer)


if __name__ == '__main__':
    chat()
