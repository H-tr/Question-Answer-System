from documentRetriever import retriever as r
from transformers import pipeline

def system():
    while True:
        ret = r.retriever()
        question = input("\nPlease enter your question: \n")
        # using cosine similarity to choose the file
        selected = ret.cosineSimilarity(question) 
        text = ret.convert_pdf_2_text("data/" + selected)

        question_answering = pipeline("question-answering")
        result = question_answering(question=question, context=text)
        print("BERT: " + result['answer'])

system()
# def main():
#     system()

# if __name__ == "__name__":
#     main()
