from xml.dom.minidom import Document
from documentRetriever import retriever as ret
from documentReader import BERTreader as rd

def system():
    question = input("\nPlease enter your question: \n")
    while True:
        retriever = ret.retriever()
        reader = rd.reader()

        # using simhash distance to choose the file
        text = retriever.engine(question)
        # print(documents)

        print("BERT: ")
        ch = input("Do you want to ask another question? (Y/n) ")
        if ch == 'y' or ch == 'Y':
            question = input("\nPlease enter your question: \n")
            reader.answer(question=question, text=text)
        else:
            print("Bye!")
            break

def main():
    system()

if __name__ == "__main__":
    main()
