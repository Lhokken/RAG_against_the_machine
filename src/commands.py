#!/usr/bin/env python3

from src.bm25_applier import Bm25sApplier
from src.evaluate import evaluation_step
from src.data_models import EvaluateRequest
import json


class CLI():

    @staticmethod
    def index(max_chunk_size: int, save_directory: str) -> None:
        """Ingest data/raw/ and build the index under data/processed/.
        """
        try:
            int(max_chunk_size)
        except ValueError as e:
            print(e)
            exit("Insert valid data !!!\n")
        Bm25sApplier.bm25_index_inizialize(max_chunk_size, save_directory)
        print("Next command:\n--make search_dataset--\n\n")

    @staticmethod
    def search(query: str, k: int) -> str:
        """Return the top-k sources for a single query.
        """
        try:
            int(k)
        except ValueError as e:
            print(e)
            exit("Insert valid data !!!\n")
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
            save_directory: str,
            save_file: str = "prova.json"
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
            save_directory,
            save_file
            )
        print("Next command:\n--make moulinette--\n\n")

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
        print("Next command:\n--make evaluate--\n\n")

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

    @staticmethod
    def moulinette() -> None:
        print("Next command:\n--make answer_dataset--\n\n")
