#!/usr/bin/env python3

import json
from src.data_models import StudentSearchResults, RagDataGround


def dict_student(
        student_search_results_path: str
        ) -> StudentSearchResults:
    try:
        with open(student_search_results_path, encoding="utf-8") as source:
            result = StudentSearchResults.model_validate(json.load(source))
    except FileNotFoundError as e:
        print(e)
        exit()
    result.search_results[0].question
    return result


def dict_dataset(dataset_path: str) -> RagDataGround:
    try:
        with open(dataset_path, encoding="utf-8") as source:
            ground_truth = RagDataGround.model_validate(json.load(source))
    except FileNotFoundError as e:
        print(e)
        exit()
    return ground_truth


def evaluation_step(
        student_search_results_path: str,
        dataset_path: str,
        k: int) -> None:
    result = dict_student(student_search_results_path)
    ground_truth = dict_dataset(dataset_path)
    ground_truth.rag_questions[0]
    n: int = 0
    for elab in result.search_results:
        # print(elab.question)
        i = 0
        id_elab = (elab.question_id)
        while (i < len(ground_truth.rag_questions)):
            if ground_truth.rag_questions[i].question_id == id_elab:
                for source in elab.retrieved_sources[:k]:
                    temp = ground_truth.rag_questions[i].sources[0]
                    if source.file_path == temp.file_path:
                        ground_first = temp.first_character_index
                        ground_last = temp.last_character_index
                        elab_first = source.first_character_index
                        elab_last = source.last_character_index
                        overlap = (
                            min(ground_last, elab_last)
                            - max(ground_first, elab_first))
                        IoU = (overlap / (ground_last - elab_first))
                        if IoU > 0.05:
                            n += 1
                            # print(f"{IoU:.2f}")
                            # print(n)
                            break
            i += 1
    rec = (n / len(result.search_results))
    valid: bool = True
    print(
        f"\nStudent data is valid: {valid}\n"
        f"Total number of questions: {len(result.search_results)}\n"
        "Total number of questions with sources: "
        f"{len(ground_truth.rag_questions)}\n"
        f"Total number of questions with student sources: {n}\n"
        f"Recall@{k}: {rec:.3f} ({(rec * 100):.1f}%)\n\n"
        )
