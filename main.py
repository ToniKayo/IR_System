from src.indexer import Indexer
from src.retrieval import Retrieval
from src.query_processor import QueryProcessor

# Set paths
dataset_path = "data/"
results_path = "results/"

# Parse documents
print("Indexing the documents.")
indexer = Indexer(dataset_path)
indexer.parse_documents()
print("Indexing complete.")

# Retrieval and query processor
retrieval = Retrieval(indexer)
query_processor = QueryProcessor(retrieval)

# Load the queries
print("Loading queries.")
queries = query_processor.load_queries(f"{dataset_path}/cran.qry.xml")
print(f"{len(queries)} queries loaded.")

# Run BM25
print("Running BM25 retrieval...")
bm25_results = query_processor.run_queries(queries, "bm25")
query_processor.save_results(bm25_results, f"{results_path}/bm25_results.txt")
print("BM25 results saved to bm25_results.txt")

# Run VSM
print("Running Vector Space Model retrieval...")
vsm_results = query_processor.run_queries(queries, "vsm")
query_processor.save_results(vsm_results, f"{results_path}/vsm_results.txt")
print("VSM results saved to vsm_results.txt")

# Run LM
print(" Running Language Model retrieval...")
lm_results = query_processor.run_queries(queries, "lm")
query_processor.save_results(lm_results, f"{results_path}/lm_results.txt")
print(" LM results saved to lm_results.txt")
