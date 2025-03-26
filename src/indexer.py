import os
import re
import xmltodict
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class Indexer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.inverted_index = defaultdict(list)  # token -> list of (doc_id, frequency)
        self.documents = {}                     # doc_id -> list of tokens
        self.doc_lengths = {}                   # doc_id -> document length
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text):
        if not text:
            return []

        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [self.stemmer.stem(word) for word in tokens if word not in stop_words]
        return tokens

    def parse_documents(self):
        file_path = os.path.join(self.dataset_path, "cran.all.1400.xml")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = xmltodict.parse(file.read())

        if 'root' in data and 'doc' in data['root']:
            documents = data['root']['doc']
        else:
            raise KeyError("Unexpected XML structure. Could not find 'root' or 'doc' keys.")

        for doc in documents:
            doc_id = int(doc['docno'])
            text = doc.get('text', '') or ''
            text = text.strip()

            if not text:
                print(f" Skipping empty document: {doc_id}")
                continue

            tokens = self.preprocess_text(text)
            self.documents[doc_id] = tokens
            self.doc_lengths[doc_id] = len(tokens)

            # Calculate term frequencies
            freq = defaultdict(int)
            for token in tokens:
                freq[token] += 1

            # Update inverted index
            for token, count in freq.items():
                self.inverted_index[token].append((doc_id, count))

    def get_inverted_index(self):
        return self.inverted_index

    def get_documents(self):
        return self.documents

    def get_doc_lengths(self):
        return self.doc_lengths
