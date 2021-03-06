from __future__ import division, unicode_literals
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
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
# import and download stopwords from NLTK
from nltk.corpus import stopwords
from nltk import download
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import WordEmbeddingSimilarityIndex
# import the simhash functions
import collections
import hashlib
import logging
import numbers
import re
import sys
from itertools import groupby
import numpy as np
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from numba import jit

if sys.version_info[0] >= 3:
    basestring = str
    unicode = str
    long = int

    def int_to_bytes(n, length):
        return n.to_bytes(length, 'big')

    def bytes_to_int(b):
        return int.from_bytes(b, 'big')
else:
    range = xrange

    def int_to_bytes(n, length):
        return '{:0{}x}'.format(n, length * 2).decode('hex')

    def bytes_to_int(b):
        return int(b.encode('hex'), 16)

def _hashfunc(x):
    return hashlib.md5(x).digest()


class Simhash(object):
    # Constants used in calculating simhash. Larger values will use more RAM.
    large_weight_cutoff = 50
    batch_size = 200

    def __init__(
            self, value, f=64, reg=r'[\w\u4e00-\u9fcc]+', hashfunc=_hashfunc, log=None
    ):
        """
        `f` is the dimensions of fingerprints, in bits. Must be a multiple of 8.
        `reg` is meaningful only when `value` is basestring and describes
        what is considered to be a letter inside parsed string. Regexp
        object can also be specified (some attempt to handle any letters
        is to specify reg=re.compile(r'\w', re.UNICODE))
        `hashfunc` accepts a utf-8 encoded string and returns either bytes
        (preferred) or an unsigned integer, in at least `f // 8` bytes.
        """
        if f % 8:
            raise ValueError('f must be a multiple of 8')

        self.f = f
        self.f_bytes = f // 8
        self.reg = reg
        self.value = None
        self.hashfunc = hashfunc
        self.hashfunc_returns_int = isinstance(hashfunc(b"test"), numbers.Integral)

        if log is None:
            self.log = logging.getLogger("simhash")
        else:
            self.log = log

        if isinstance(value, Simhash):
            self.value = value.value
        elif isinstance(value, basestring):
            self.build_by_text(unicode(value))
        elif isinstance(value, Iterable):
            self.build_by_features(value)
        elif isinstance(value, numbers.Integral):
            self.value = value
        else:
            raise Exception('Bad parameter with type {}'.format(type(value)))

    def __eq__(self, other):
        """
        Compare two simhashes by their value.
        :param Simhash other: The Simhash object to compare to
        """
        return self.value == other.value

    def _slide(self, content, width=4):
        return [content[i:i + width] for i in range(max(len(content) - width + 1, 1))]

    def _tokenize(self, content):
        content = content.lower()
        content = ''.join(re.findall(self.reg, content))
        ans = self._slide(content)
        return ans

    def build_by_text(self, content):
        features = self._tokenize(content)
        features = {k:sum(1 for _ in g) for k, g in groupby(sorted(features))}
        return self.build_by_features(features)

    def build_by_features(self, features):
        """
        `features` might be a list of unweighted tokens (a weight of 1
                   will be assumed), a list of (token, weight) tuples or
                   a token -> weight dict.
        """
        sums = []
        batch = []
        count = 0
        w = 1
        truncate_mask = 2 ** self.f - 1
        if isinstance(features, dict):
            features = features.items()

        for f in features:
            skip_batch = False
            if not isinstance(f, basestring):
                f, w = f
                skip_batch = w > self.large_weight_cutoff or not isinstance(w, int)

            count += w
            if self.hashfunc_returns_int:
                h = int_to_bytes(self.hashfunc(f.encode('utf-8')) & truncate_mask, self.f_bytes)
            else:
                h = self.hashfunc(f.encode('utf-8'))[-self.f_bytes:]

            if skip_batch:
                sums.append(self._bitarray_from_bytes(h) * w)
            else:
                batch.append(h * w)
                if len(batch) >= self.batch_size:
                    sums.append(self._sum_hashes(batch))
                    batch = []

            if len(sums) >= self.batch_size:
                sums = [np.sum(sums, 0)]

        if batch:
            sums.append(self._sum_hashes(batch))

        combined_sums = np.sum(sums, 0)
        self.value = bytes_to_int(np.packbits(combined_sums > count / 2).tobytes())

    def _sum_hashes(self, digests):
        bitarray = self._bitarray_from_bytes(b''.join(digests))
        rows = np.reshape(bitarray, (-1, self.f))
        return np.sum(rows, 0)

    @staticmethod
    def _bitarray_from_bytes(b):
        return np.unpackbits(np.frombuffer(b, dtype='>B'))

    def distance(self, another):
        assert self.f == another.f
        x = (self.value ^ another.value) & ((1 << self.f) - 1)
        ans = 0
        while x:
            ans += 1
            x &= x - 1
        return ans


class SimhashIndex(object):

    def __init__(self, objs, f=64, k=2, log=None):
        """
        `objs` is a list of (obj_id, simhash)
        obj_id is a string, simhash is an instance of Simhash
        `f` is the same with the one for Simhash
        `k` is the tolerance
        """
        self.k = k
        self.f = f
        count = len(objs)

        if log is None:
            self.log = logging.getLogger("simhash")
        else:
            self.log = log

        self.log.info('Initializing %s data.', count)

        self.bucket = collections.defaultdict(set)

        for i, q in enumerate(objs):
            if i % 10000 == 0 or i == count - 1:
                self.log.info('%s/%s', i + 1, count)

            self.add(*q)

    def get_near_dups(self, simhash):
        """
        `simhash` is an instance of Simhash
        return a list of obj_id, which is in type of str
        """
        assert simhash.f == self.f

        ans = set()

        for key in self.get_keys(simhash):
            dups = self.bucket[key]
            self.log.debug('key:%s', key)
            if len(dups) > 200:
                self.log.warning('Big bucket found. key:%s, len:%s', key, len(dups))

            for dup in dups:
                sim2, obj_id = dup.split(',', 1)
                sim2 = Simhash(long(sim2, 16), self.f)

                d = simhash.distance(sim2)
                if d <= self.k:
                    ans.add(obj_id)
        return list(ans)

    def add(self, obj_id, simhash):
        """
        `obj_id` is a string
        `simhash` is an instance of Simhash
        """
        assert simhash.f == self.f

        for key in self.get_keys(simhash):
            v = '%x,%s' % (simhash.value, obj_id)
            self.bucket[key].add(v)

    def delete(self, obj_id, simhash):
        """
        `obj_id` is a string
        `simhash` is an instance of Simhash
        """
        assert simhash.f == self.f

        for key in self.get_keys(simhash):
            v = '%x,%s' % (simhash.value, obj_id)
            if v in self.bucket[key]:
                self.bucket[key].remove(v)

    @property
    def offsets(self):
        """
        You may optimize this method according to <http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/33026.pdf>
        """
        return [self.f // (self.k + 1) * i for i in range(self.k + 1)]

    def get_keys(self, simhash):
        for i, offset in enumerate(self.offsets):
            if i == (len(self.offsets) - 1):
                m = 2 ** (self.f - offset) - 1
            else:
                m = 2 ** (self.offsets[i + 1] - offset) - 1
            c = simhash.value >> offset & m
            yield '%x:%x' % (c, i)

    def bucket_size(self):
        return len(self.bucket)


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

    def cosineSimilarity(self, documents, text):
        fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
        download('stopwords')
        stop_words = stopwords.words('english')
        maxSmi = 0.0
        selected = ''
        for item in documents:
            # print("Computing similarity of " + row['name'])
            data = item.lower().split()
            data = [w for w in data if w not in stop_words]
            query = text.lower().split()
            query = [w for w in query if w not in stop_words]

            # prepare a dictionary and a corpus
            document = [query, data]
            #  print(data)
            #  print(documents)
            dictionary = corpora.Dictionary(document)

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
                # maxDoc = row['name']
                selected = item
                print("Selected")
            print(f"The similarity is {similarity}")
        return selected

    def simHash(self, text):
        distance = 1e5
        documents = []
        for _, row in self.doc_features.iterrows():
            # print("Computing similarity of " + row['name'])
            sh = Simhash(row['text'])
            sh2 = Simhash(text)
            curDistance = sh.distance(sh2)
            if distance > curDistance:
                distance = curDistance
            print(f"The distance to {row['name']} is {curDistance}")
        # get all the documents with lowest distance
        for _, row in self.doc_features.iterrows():
            sh = Simhash(row['text'])
            sh2 = Simhash(text)
            if distance == sh.distance(sh2):
                documents.append(row['text'])
                print(row['name'] + " is selected.")
        return documents

    def engine(self, question):
        documents = self.simHash(question)
        selected = self.cosineSimilarity(documents, question)
        return selected