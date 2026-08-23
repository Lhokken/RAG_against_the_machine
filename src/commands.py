#!/usr/bin/env python3

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
        result = json.dumps(
            (Bm25sApplier.search_single_query(query, k)),
            indent=4
            )
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

    @staticmethod
    def evaluate(
            student_search_results_path: str, dataset_path: str
            ) -> None:
        """Report your own recall@k against

        a ground-truth dataset, for your own testing."""
        pass
