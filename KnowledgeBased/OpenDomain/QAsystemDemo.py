import pandas as pd
import numpy as np
import torch
import wikipedia as wiki
import pprint as pp
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from collections import OrderedDict

model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

question = "What is the wingspan of an albatross?"
results = wiki.search(question)
page = wiki.page(results[0])
text = page.content

# 1. TOKENIZE THE INPUT
# note: if you don't include return_tensors='pt' you'll get a list of lists which is easier for 
# exploration but you cannot feed that into a model. 
inputs = tokenizer.encode_plus(question, text, return_tensors="pt") 

# identify question tokens (token_type_ids = 0)
qmask = inputs["token_type_ids"].lt(1)
qt = torch.masked_select(inputs["input_ids"], qmask)
print(f"The question consists of {qt.size()[0]} tokens.")

chunk_size = model.config.max_position_embeddings - qt.size()[0] - 1
print(f"Each chunk will contain {chunk_size-2} tokens of the Wikipedia articles.")

chunked_input = OrderedDict()
for k, v in inputs.items():
    q = torch.masked_select(v, qmask)
    c = torch.masked_select(v, ~qmask)
    chunks = torch.split(c, chunk_size)

    for i, chunk in enumerate(chunks):
        if i not in chunked_input:
            chunked_input[i] = {}

        thing = torch.cat((q, chunk))
        if i != len(chunks)-1:
            if k == 'input_ids':
                thing = torch.cat((thing, torch.tensor([102])))
            else:
                thing = torch.cat((thing, torch.tensor([1])))

        chunked_input[i][k] = torch.unsqueeze(thing, dim=0)

for i in range(len(chunked_input.keys())):
    print(f"Number of tokens in chunk {i}: {len(chunked_input[i]['input_ids'].tolist()[0])}")

def convert_ids_to_string(tokenizer, input_ids):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))

answer = ''

# now we iterate over out chunks, looking for the best answer from each chunk
for _, chunk in chunked_input.items():
    answer_start_scores,answer_end_scores = model(**chunk, return_dict=False) 
    
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    ans = convert_ids_to_string(tokenizer, chunk['input_ids'][0][answer_start:answer_end])

    # if the ans == [CLS] then the model did not find a real answer in this chunk
    if ans != '[CLS]':
        answer += ans + "/"

print(answer)
