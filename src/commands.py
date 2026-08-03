#!/usr/bin/env python3

from src.explorer import Searcher
from src.bm25_applier import Bm25sApplier

class CLI():

	@staticmethod
	def index(max_chunk_size: int) -> None:
		Bm25sApplier.tokenizer(Searcher.analizer(max_chunk_size))
		# se cambia il max_chunk_size e necessario ricreare il corpus ?
		# Ingest data/raw/ and build the index under data/processed/.

	@staticmethod
	def search(query: str, k: int) -> None:
		Bm25sApplier.single_query(query, k)
		"""serach method"""
		# • search <query> –k <int>
		# Return the top-k sources for a single query.
		...

	# @staticmethod
	# def search_dataset(dataset_path: path, k: int, save_directory: dir) -> None:
	# 	# Run search over a whole dataset and write a StudentSearchResults JSON file.
	# 	...

	# @staticmethod
	# def answer(query: str, k: int) -> None:
	# 	# Answer a single query using the retrieved context.
	# 	...

	# @staticmethod
	# def answer_dataset(student_search_results_path: path, save_directory: dir) -> None:
	# 	# Generate answers for a dataset, producing a StudentSearchResultsAndAnswer
	# 	# JSON file.
	# 	...

	# @staticmethod
	# def evaluate(student_search_results_path: path, dataset_path: path) -> None:
	# 	# Report your own recall@k against a ground-truth dataset, for your own testing.
		...