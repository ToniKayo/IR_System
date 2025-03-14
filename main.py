from src.indexer import Indexer
from src.retrieval import Retrieval
from src.query_processor import QueryProcessor

dataset_path = "data/"
indexer = Indexer(dataset_path)
indexer.parse_documents()

retrieval = Retrieval(indexer)
query_processor = QueryProcessor(retrieval)

queries = query_processor.load_queries(f"{dataset_path}/cran.qry.xml")

# ✅ Run BM25 and save results
bm25_results = query_processor.run_queries(queries, "bm25")
query_processor.save_results(bm25_results, "results/bm25_results.txt")

# ✅ Run VSM and save results
vsm_results = query_processor.run_queries(queries, "vsm")
query_processor.save_results(vsm_results, "results/vsm_results.txt")

# ✅ Run LM and save results
lm_results = query_processor.run_queries(queries, "lm")
query_processor.save_results(lm_results, "results/lm_results.txt")

