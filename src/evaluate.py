#!/usr/bin/env python3

import json
from src.data_models import StudentSearchResults, \
    RagDataGround, EvaluateOutput, EvaluateRequest
from fastapi import FastAPI


app = FastAPI()


def dict_student(student_search_results_path: str) -> StudentSearchResults:
    """This function verify and return a student search result"""
    try:
        with open(student_search_results_path, encoding="utf-8") as source:
            result = StudentSearchResults.model_validate(json.load(source))
    except FileNotFoundError as e:
        print(e)
        exit("\nStudent data is valid: False\n")
    result.search_results[0].question
    return result


def dict_dataset(dataset_path: str) -> RagDataGround:
    """This function verify and return a rag data ground_truth result"""
    try:
        with open(dataset_path, encoding="utf-8") as source:
            ground_truth = RagDataGround.model_validate(json.load(source))
    except FileNotFoundError as e:
        print(e)
        exit("\nStudent data is valid: False\n")
    return ground_truth


@app.post("/local_path")  # type: ignore[untyped-decorator]
def evaluation_step(data_output: EvaluateRequest) -> EvaluateOutput:
    """This function build the recall@k result

    In order to obtain the result, launche k times the
    single evaluate function"""
    result = dict_student(data_output.student_search_results_path)
    ground_truth = dict_dataset(data_output.dataset_path)
    n: int = 0
    i: int = 0
    j: int = 0
    IoU: float = 0
    eval_result: list[list[str | float]] = []
    while (j < data_output.k):
        j = j + 1
        n = 0
        for elab in result.search_results:
            i = 0
            id_elab = (elab.question_id)
            while (i < len(ground_truth.rag_questions)):
                if ground_truth.rag_questions[i].question_id == id_elab:
                    for source in elab.retrieved_sources[:j]:
                        temp = ground_truth.rag_questions[i].sources[0]
                        if source.file_path == temp.file_path:
                            ground_first = temp.first_character_index
                            ground_last = temp.last_character_index
                            elab_first = source.first_character_index
                            elab_last = source.last_character_index
                            intersection = (
                                min(ground_last, elab_last)
                                - max(ground_first, elab_first))
                            union = (
                                max(ground_last, elab_last)
                                - min(ground_first, elab_first)
                            )
                            try:
                                IoU = (intersection / union)
                            except ZeroDivisionError as e:
                                IoU = 0.0
                                print(e)
                            if IoU > 0.05:
                                n += 1
                                break
                i += 1
        rec = (n / len(result.search_results))
        eval_result.append([f"Recall@{j}: {rec:.3f}", (rec * 100)])
    rek = ""
    rekall: list[str] = []
    for text in eval_result:
        rek += (f"{text[0]} ({text[1]:.1f}%)\n")
        rekall.append(f"{text[0]} ({text[1]:.1f}%)")
    output = EvaluateOutput.model_validate({
        "is_valid": True,
        "total_student_questions": len(result.search_results),
        "total_ground_truth_questions": len(ground_truth.rag_questions),
        "recall_results": rekall
    })
    output_str = (
        f"\nStudent data is valid: True\n"
        f"Total number of questions: {len(result.search_results)}\n"
        f"Total number of questions with sources: "
        f"{len(ground_truth.rag_questions)}\n\n"
        "Evaluation results\n"
        "======================================\n"
        f"Questions evaluated: {i}\n\n"
        f"{rek}"
        )
    print(output_str)
    print()
    return output
