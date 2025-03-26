import xmltodict

class QueryProcessor:
    def __init__(self, retrieval):
        self.retrieval = retrieval

    def load_queries(self, query_path):
        with open(query_path, 'r', encoding='utf-8') as file:
            data = xmltodict.parse(file.read())

        if 'xml' in data and 'top' in data['xml']:
            queries = {
                int(query['num']): query['title'].strip()
                for query in data['xml']['top']
            }
        else:
            raise KeyError("Unexpected XML structure. Check the printed output for the correct key.")

        return queries

    def run_queries(self, queries, model_name):
        results = []

        for qid, query in queries.items():
            print(f"🔍 Processing Query {qid}: {query[:50]}...")

            # Select model
            if model_name == "vsm":
                ranked_docs = self.retrieval.vector_space_model(query)
            elif model_name == "bm25":
                ranked_docs = self.retrieval.bm25(query)
            elif model_name == "lm":
                ranked_docs = self.retrieval.language_model(query)
                if ranked_docs:
                    min_score = min(ranked_docs, key=lambda x: x[1])[1]
                    max_score = max(ranked_docs, key=lambda x: x[1])[1]
                    if max_score - min_score > 0:
                        ranked_docs = [
                            (doc, (score - min_score) / (max_score - min_score))
                            for doc, score in ranked_docs
                        ]
            else:
                raise ValueError(f"Invalid retrieval model specified: {model_name}")

            # Format top 100 results
            for rank, (doc_id, score) in enumerate(ranked_docs[:100]):
                results.append(f"{qid} 0 {doc_id} {rank + 1} {score:.4f} my_ir_system")

        return results

    def save_results(self, results, output_path):
        with open(output_path, 'w') as file:
            file.write("\n".join(results))
