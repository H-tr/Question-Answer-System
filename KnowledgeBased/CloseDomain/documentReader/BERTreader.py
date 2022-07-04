from transformers import pipeline
from transformers import AutoTokenizer
import warnings
import numpy as np
BLOCK_SIZE = 63

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class reader():
    def __init__(self) -> None:
        pass

    def answer(self, question, text):
        question_answering = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased")
        result = question_answering(question=question, context=text)
        return result['answer']

    def split_document(d, tokenizer):
        '''
            d: [['word', '##piece'], ...] # a document of tokenized sentences 
            properties: [
                            [
                                (name: str, value: any), # len(2) tuple, sentence level property
                                (name: str, position: int, value: any) # len(3) tuple, token level property
                            ],
                            []... # len(d) lists
                        ]
        '''
        # d is only a list of tokens, not split. 
        # properties are also a list of tuples.
        cnt = 0
        end_tokens = {'\n':0, '.':1, '?':1, '!':1, ',':2}
        for k, v in list(end_tokens.items()):
            end_tokens[k] = v
        sen_cost, break_cost = 4, 8
        poses = [(i, end_tokens[tok]) for i, tok in enumerate(d) if tok in end_tokens]
        poses.insert(0, (-1, 0))
        if poses[-1][0] < len(d) - 1:
            poses.append((len(d) - 1, 0))
        x = 0
        while x < len(poses) - 1:
            if poses[x + 1][0] - poses[x][0] > BLOCK_SIZE:
                poses.insert(x + 1, (poses[x][0] + BLOCK_SIZE, break_cost))
            x += 1
        # simple dynamic programming
        best = [(0, 0)]
        for i, (p, cost) in enumerate(poses):
            if i == 0:
                continue    
            best.append((-1, 100000))
            for j in range(i-1, -1, -1):
                if p - poses[j][0] > BLOCK_SIZE:
                    break
                value = best[j][1] + cost + sen_cost
                if value < best[i][1]:
                    best[i] = (j, value)
            assert best[i][0] >= 0
        intervals, x = [], len(poses) - 1
        while x > 0:
            l = poses[best[x][0]][0]
            intervals.append((l + 1, poses[x][0] + 1))
            x = best[x][0]
        for st, en in reversed(intervals):
            # copy from hard version
            cnt += 1
            tmp = d[st: en] + [tokenizer.sep_token]
            yield tmp

if __name__ == "__main__":
    s = """I just recently realized that I am bisexual, and also just recently returned to religion, and have a good friend who has pointed out to me that homosexuality is a sin in the bible.  Well, I don't see how it could be considered a sin,
First of all as far as I know, only male homosexuality is explicitly
mentioned in the bibles, so you're off the hook there, I think. In
any event, there are *plenty* of people in many denominations who
do not consider a person's sexual identification of gay/lesbian/bisexual
as an "immoral lifestyle choice"
Also, I have always been a somewhat liberal feminist, and am pro-choice, and it seems that being pro-choice and being religious don't mix either.  I am told
This is another misconception. You are not being told the whole story.
My former minister is a lesbian, and I know personally and
professionally several openly gay and lesbian ministers. I am
a Unitarian-Universalist and like most others in my denomination,
am pro-choice. You needn't go looking to the Unitarian Universalists
(which is a liberal religion) for acceptance of your sexual
identification and pro-choice views, however; there are many of us
who believe in spirituality AND freedom of conscience.
Good Luck on your journey! ADDFSDFDE*(YT(*HO*E))DHF(NKLSHDFDFSFLFJDKSFKSHOFEINLIDS)*Y&(*&(23423534twer54324524)245)4353453777777777777777777777777777777777777777777777777777777777777777777777777777777
4353453777777777777777777777777777777777777777777777777777777777777777777777777777777
"""
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    reader.split_document(tokenizer.tokenize(s), tokenizer)
    # print("Numbers of blocks: " + str(cnt))