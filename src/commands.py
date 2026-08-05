#!/usr/bin/env python3

from src.explorer import Searcher
from src.bm25_applier import Bm25sApplier
import json


class CLI():

	@staticmethod
	def index(max_chunk_size: int) -> None:
		Bm25sApplier.tokenizer(Searcher.analizer(max_chunk_size))
		# se cambia il max_chunk_size e necessario ricreare il corpus ?
		# Ingest data/raw/ and build the index under data/processed/.

	@staticmethod
	def search(query: str, k: int) -> None:
		print("Elaborating query ...\n")
		Bm25sApplier.bm25_index_inizialize()
		print(json.dumps((Bm25sApplier.single_query(query, k)), indent=4))
		# Return the top-k sources for a single query.

	@staticmethod
	def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
		Bm25sApplier.search_dataset_query(dataset_path, k, save_directory)
		# Run search over a whole dataset and write a StudentSearchResults JSON file.


	# @staticmethod qwen
	# def answer(query: str, k: int) -> None:
	# 	# Answer a single query using the retrieved context.
	# 	...

	# @staticmethod qwen
	# def answer_dataset(student_search_results_path: path, save_directory: dir) -> None:
	# 	# Generate answers for a dataset, producing a StudentSearchResultsAndAnswer
	# 	# JSON file.
	# 	...

	# @staticmethod
	# def evaluate(student_search_results_path: path, dataset_path: path) -> None:
	# 	# Report your own recall@k against a ground-truth dataset, for your own testing.
		...


# • For search operations: Use StudentSearchResults model with:
# ◦ search_results: List of MinimalSearchResults containing:
# 	question_id,
# 	question,
# 	retrieved_sources
# ◦ k: Number of results requested


# • For answer generation: Use StudentSearchResultsAndAnswer model with:
# ◦ search_results: List of MinimalAnswer containing:
# 	question_id,
# 	question,
#	retrieved_sources,
# 	and answer
# ◦ k: Number of results requested


# • Source information: Each MinimalSource contains:
# ◦ file_path: path to the source file, (e.g. data/raw/vllm-0.10.1/...); it is # compared verbatim to the reference
# ◦ first_character_index: Starting character position
# ◦ last_character_index: Ending character position