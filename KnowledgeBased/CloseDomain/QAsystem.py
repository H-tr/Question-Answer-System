from documentRetriever import retriever as r
from transformers import pipeline
import warnings
import numpy as np

def system():
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
    question = input("\nPlease enter your question: \n")
    while True:
        ret = r.retriever()

        # using cosine similarity to choose the file
        # name, text = ret.cosineSimilarity(question)

        # using simhash distance to choose the file
        name, text = ret.simHash(question) 

        question_answering = pipeline("question-answering")
        result = question_answering(question=question, context=text)
        print("BERT: " + result['answer'])
        ch = input("Do you want to ask another question? (Y/n) ")
        if ch == 'y' or ch == 'Y':
            question = input("\nPlease enter your question: \n")
        else:
            print("Bye!")
            break

def main():
    system()

if __name__ == "__main__":
    main()
