from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
# the soft cosine
import gensim
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
# import and download stopwords from NLTK
from nltk.corpus import stopwords
from nltk import download
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import WordEmbeddingSimilarityIndex

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
                    #  bagOfWords = text.split(' ')
                    #  self.uniqueWords = set(bagOfWords).union(self.uniqueWords)
                    nameList.append([filename, text])
            # build doc features
            self.doc_features = pd.DataFrame(data = nameList, columns=["name", "text"]) # create a dataframe that contain features for each file
            #  self.numOfWords = dict.fromkeys(self.uniqueWords, 0)
            self.doc_features.to_csv("doc_features.csv")
        else:
            self.doc_features = pd.read_csv("doc_features.csv")
            self.doc_features = self.doc_features.loc[:, ~self.doc_features.columns.str.contains('^Unnamed')] # clean the unamed columns
        #  print(self.doc_features)

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

    def cosineSimilarity(self, text):
        fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
        download('stopwords')
        stop_words = stopwords.words('english')
        maxSmi = 0.0
        maxDoc = ''
        for _, row in self.doc_features.iterrows():
            print("Computing similarity of " + row['name'])
            data = row['text'].lower().split()
            data = [w for w in data if w not in stop_words]
            query = text.lower().split()
            query = [w for w in query if w not in stop_words]

            # prepare a dictionary and a corpus
            documents = [query, data]
            #  print(data)
            #  print(documents)
            dictionary = corpora.Dictionary(documents)

            # Convert the sentences into bag-of-words vectors
            query = dictionary.doc2bow(query)
            data = dictionary.doc2bow(data)
 
            # prepare the similarity matrix
            similarity_index = WordEmbeddingSimilarityIndex(fasttext_model300)
            similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)
            
            # Compute soft cosine similarity
            similarity = similarity_matrix.inner_product(query, data, normalized=(True, True))
            if similarity > maxSmi:
                maxSmi = similarity
                maxDoc = row['name']
            print(f"The similarity to {row['name']} is {similarity}")
        return maxDoc
