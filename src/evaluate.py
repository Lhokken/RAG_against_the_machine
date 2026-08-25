#!/usr/bin/env python3

import json
from src.data_models import StudentSearchResults, RagDataGround

def dict_student(
        student_search_results_path: str
        ) -> StudentSearchResults:
    with open(student_search_results_path, encoding="utf-8") as source:
        result = StudentSearchResults.model_validate(json.load(source))
    result.search_results[0].question
    return result

def dict_dataset(dataset_path: str) -> RagDataGround:
    with open(dataset_path, encoding="utf-8") as source:
        ground_truth = RagDataGround.model_validate(json.load(source))
    return ground_truth

def evaluation_step(
        student_search_results_path: str,
        dataset_path: str,
        k: int) -> None:
    result = dict_student(student_search_results_path)
    ground_truth = dict_dataset(dataset_path)
    ground_truth.rag_questions[0]
    for elab in result.search_results:
        i = 0
        id_elab = (elab.question_id)
        while (i < len(ground_truth.rag_questions)):
            if ground_truth.rag_questions[i].question_id == id_elab:
                for source in elab.retrieved_sources:
                    if source.file_path == ground_truth.rag_questions[i].sources[0].file_path:
                        print(source.file_path)
                break
            i += 1
