import wikipedia as wiki
from QAsystem import DocumentReader
import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

reader = DocumentReader()
question = input("\nPlease enter your question: \n")

while True:
    results = wiki.search(question)
    page = wiki.page(results[0])
    print(f"Top wiki result: {page}")

    text = page.content
    reader.tokenize(question, text)
    print(f"Answer: {reader.get_answer()}")
    
    flag = True
    flag_N = False

    while flag:
        response = input("\nDo you want to ask another question (Y/N)?")
        if response[0] == "Y":
            question = input("\nPlease enter your question: \n")
            flag = False
        elif response[0] == "N":
            print("\nBye!")
            flag = False
            flag_N = True

    if flag_N == True:
        break

