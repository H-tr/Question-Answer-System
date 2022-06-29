from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

class retriever:
    def __init__(self, directory = "data") -> None:
        if not os.path.exists("doc_features.csv"):
            print("Building the structured data...")
            self.directory = directory
            self.uniqueWords = set()
            nameList = []
            for filename in tqdm(os.listdir(directory)): # get filenames to create the dataframe
                f = os.path.join(directory, filename)
                if os.path.isfile(f):
                    text = self.convert_pdf_2_text(f)
                    bagOfWords = text.split(' ')
                    self.uniqueWords = set(bagOfWords).union(self.uniqueWords)
                    nameList.append([filename, text, []])
            self.getKeywords()
            self.doc_features = pd.DataFrame(data = nameList, columns=["name", "text", "key_words"]) # create a dataframe that contain features for each file
            self.numOfWords = dict.fromkeys(self.uniqueWords, 0)
            self.doc_features.to_csv("doc_features.csv")
        else:
            self.doc_features = pd.read_csv("doc_features.csv")
            self.doc_features = self.doc_features.loc[:, ~self.doc_features.columns.str.contains('^Unnamed')] # clean the unamed columns
        print(self.doc_features)

    def convert_pdf_2_text(self, path):

        rsrcmgr = PDFResourceManager()
        retstr = StringIO()

        device = TextConverter(rsrcmgr, retstr, codec='utf-8', laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        with open(path, 'rb') as fp:
            for page in PDFPage.get_pages(fp, set()):
                interpreter.process_page(page)
            text = retstr.getvalue()

        device.close()
        retstr.close()

        return text

    def getKeywords(self): # TF-IDF keyword pickup
        for index, row in self.doc_features.iterrows():
            pass