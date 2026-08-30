#!/usr/bin/env python3

from src.bm25_applier import Bm25sApplier
from src.evaluate import evaluation_step
from src.data_models import EvaluateRequest, MinimalSource
import json


class CLI():

    @staticmethod
    def index(
            max_chunk_size: int,
            save_directory: str = "data/processed"
            ) -> None:
        """Ingest data/raw/ and build the index under data/processed/.
        """
        try:
            int(max_chunk_size)
        except ValueError as e:
            print(e)
            exit("Insert valid data !!!\n")
        Bm25sApplier.bm25_index_inizialize(max_chunk_size, save_directory)

    @staticmethod
    def search(query: str, k: int) -> list[MinimalSource]:
        """Return the top-k sources for a single query.
        """
        try:
            int(k)
        except ValueError as e:
            print(e)
            exit("Insert valid data !!!\n")
        Bm25sApplier.bm25_index_inizialize()
        print(
            "Elaborating query ...\n"
            f"{query}"
        )
        result = Bm25sApplier.search_single_query(query, k)
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
        try:
            int(k)
        except ValueError as e:
            print(e)
            exit("Insert valid data !!!\n")
        Bm25sApplier.search_dataset_query(
            dataset_path,
            k,
            save_directory
            )

    @staticmethod
    def answer(query: str, k: int) -> None:
        """Answer a single query using the retrieved context.
        """
        try:
            int(k)
        except ValueError as e:
            print(e)
            exit("Insert valid data !!!\n")
        Bm25sApplier.answer_single_query(query, k)

    @staticmethod
    def answer_dataset(
            student_search_results_path: str,
            save_directory: str,
            save_file: str = "final_elaborate.json"
            ) -> None:
        """Generate answers for a dataset, producing
        a StudentSearchResultsAndAnswer JSON file.
        """
        Bm25sApplier.answer_dataset_query(
            student_search_results_path,
            save_directory,
            save_file
            )

    @staticmethod
    def evaluate(
            student_search_results_path: str,
            dataset_path: str,
            k: int
            ) -> None:
        """Report your own recall@k against

        a ground-truth dataset, for your own testing.
        """
        data_output = EvaluateRequest.model_validate({
            "student_search_results_path": student_search_results_path,
            "dataset_path": dataset_path,
            "k": k
        })
        try:
            int(data_output.k)
        except ValueError as e:
            print(e)
            exit("Insert valid data !!!\n")
        evaluation_step(data_output)
