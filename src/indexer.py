import os
import re
import xmltodict
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class Indexer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.inverted_index = defaultdict(list)
        self.documents = {}

    def preprocess_text(self, text):
        """Lowercase, remove special characters, tokenize, and remove stopwords."""
        if not text:  # Handle empty text safely
            return []
        
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return tokens

    def parse_documents(self):
        """Parses the Cranfield dataset, processes text, and builds an inverted index."""
        with open(os.path.join(self.dataset_path, "cran.all.1400.xml"), 'r', encoding='utf-8') as file:
            data = xmltodict.parse(file.read())

        # Ensure correct key based on XML structure
        if 'root' in data and 'doc' in data['root']:
            documents = data['root']['doc']
        else:
            raise KeyError("Unexpected XML structure. Could not find 'root' or 'doc' keys.")

        # Process each document
        for doc in documents:
            doc_id = int(doc['docno'])  # Ensure doc ID is an integer

            # ✅ Ensure text is never None
            text = doc.get('text', '')  # Use get() to avoid KeyError
            if text is None:
                text = ""  # Assign empty string if missing
            
            text = text.strip()  # Remove extra spaces

            # Skip empty documents
            if not text:
                print(f"⚠️ Skipping empty document: {doc_id}")
                continue

            tokens = self.preprocess_text(text)  # Process text safely
            self.documents[doc_id] = tokens

            # Build inverted index
            for token in set(tokens):
                self.inverted_index[token].append(doc_id)

    def get_inverted_index(self):
        """Returns the inverted index."""
        return self.inverted_index

    def get_documents(self):
        """Returns the parsed and tokenized documents."""
        return self.documents
