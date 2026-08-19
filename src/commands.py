#!/usr/bin/env python3

from src.explorer import Searcher
from src.bm25_applier import Bm25sApplier
import json


class CLI():

    @staticmethod
    def index(max_chunk_size: int) -> None:
        """Ingest data/raw/ and build the index under data/processed/.
        """
        Bm25sApplier.bm25_index_inizialize(max_chunk_size)

    @staticmethod
    def search(query: str, k: int) -> str:
        """Return the top-k sources for a single query.
        """
        print(
            "Elaborating query ...\n"
            f"{query}"
        )
        result = json.dumps((Bm25sApplier.search_single_query(query, k)), indent=4)
        # print(result)
        return result

    @staticmethod
    def search_dataset(
            dataset_path: str,
            k: int,
            save_directory: str
            ) -> None:
        """Run search over a whole dataset and write a
        StudentSearchResults JSON file.
        """
        Bm25sApplier.search_dataset_query(dataset_path, k, save_directory)

    @staticmethod
    def answer(query: str, k: int) -> None:
        """Answer a single query using the retrieved context.
        """
        Bm25sApplier.answer_single_query(query, k)

    @staticmethod
    def answer_dataset(
            student_search_results_path: str,
            save_directory: str
            ) -> None:
        """Generate answers for a dataset, producing
        a StudentSearchResultsAndAnswer JSON file.
        """
        Bm25sApplier.answer_dataset_query(
            student_search_results_path,
            save_directory
            )
    	# Generate answers for a dataset, producing a StudentSearchResultsAndAnswer
    	# JSON file.


    # @staticmethod
    # def evaluate(student_search_results_path: path, dataset_path: path) -> None:
    # 	# Report your own recall@k against a ground-truth dataset, for your own testing.
        # ...


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
