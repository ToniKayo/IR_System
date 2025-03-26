import numpy as np
import math
from collections import Counter, defaultdict

class Retrieval:
    def __init__(self, indexer):
        self.indexer = indexer
        self.document_lengths = indexer.get_doc_lengths()
        self.avg_doc_length = np.mean(list(self.document_lengths.values()))
        self.doc_count = len(self.indexer.get_documents())

        self.corpus_term_freq = defaultdict(int)
        total_terms = 0
        for tokens in self.indexer.get_documents().values():
            for token in tokens:
                self.corpus_term_freq[token] += 1
                total_terms += 1

        # Normalize to get term probabilities
        self.corpus_term_prob = {
            term: freq / total_terms for term, freq in self.corpus_term_freq.items()
        }

    def vector_space_model(self, query):
        query_tokens = self.indexer.preprocess_text(query)
        query_freq = Counter(query_tokens)
        scores = defaultdict(float)

        for term in query_freq:
            if term in self.indexer.get_inverted_index():
                doc_list = self.indexer.get_inverted_index()[term]
                idf = math.log((self.doc_count + 1) / (len(doc_list) + 1))

                for doc_id, tf in doc_list:
                    scores[doc_id] += tf * idf  # basic TF-IDF score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def bm25(self, query, k1=1.5, b=0.75):
        query_tokens = self.indexer.preprocess_text(query)
        scores = defaultdict(float)

        for term in query_tokens:
            if term in self.indexer.get_inverted_index():
                doc_list = self.indexer.get_inverted_index()[term]
                idf = math.log((self.doc_count - len(doc_list) + 0.5) / (len(doc_list) + 0.5) + 1)

                for doc_id, tf in doc_list:
                    doc_len = self.document_lengths[doc_id]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_length))
                    score = numerator / denominator
                    scores[doc_id] += idf * score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def language_model(self, query, lambda_param=0.1):
        query_tokens = self.indexer.preprocess_text(query)
        scores = defaultdict(float)

        for doc_id, tokens in self.indexer.get_documents().items():
            doc_counter = Counter(tokens)
            doc_len = len(tokens)
            doc_score = 0

            for term in query_tokens:
                term_prob_doc = doc_counter[term] / doc_len if term in doc_counter else 0
                term_prob_corpus = self.corpus_term_prob.get(term, 1e-8)
                smoothed_prob = (lambda_param * term_prob_doc) + ((1 - lambda_param) * term_prob_corpus)
                doc_score += math.log(smoothed_prob)

            scores[doc_id] = doc_score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
