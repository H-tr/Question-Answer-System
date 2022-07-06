from documentRetriever import retriever as ret
from documentReader import BERTreader as rd

def system():
    question = input("\nPlease enter your question: \n")
    retriever = ret.retriever()
    reader = rd.reader()

    # using simhash distance and cosine similarity to choose the file
    text = retriever.engine(question)
    # print(question)
    # text = retriever.convert_pdf_2_text("data/AlexNet.pdf")
    result = reader.answer(question, text)
    print("BERT: " + result)

    while True:
        ch = input("Do you want to ask another question? (Y/n) ")
        if ch == 'y' or ch == 'Y':
            question = input("\nPlease enter your question: \n")
            text = retriever.engine(question)
            result = reader.answer(question, text)
            print("BERT: " + result)
        else:
            print("Bye!")
            break

def main():
    system()

if __name__ == "__main__":
    main()
