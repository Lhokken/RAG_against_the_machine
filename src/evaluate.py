#!/usr/bin/env python3

import json

def evaluation_step(student_search_results_path: str, dataset_path: str) -> None:
    with open(student_search_results_path, encoding="utf-8") as source:
        text_content = json.load(source)
        result = []
        for elem in (text_content["search_results"]):
            text_1 = {}
            list_1 = []
            for elem2 in elem['retrieved_sources']:
                list_1.append(elem2)
            text_1.update({
                "question_id": f"{elem['question_id']}",
                "retrieved_sources": list_1,
            })
            result.append((text_1))

    # print(json.dumps(result, indent=2))
    # for elem in result:
    #     for elem1 in elem["retrieved_sources"]:
    #         print(elem1["last_character_index"])

    with open(dataset_path, encoding="utf-8") as source:
        text_content2 = json.load(source)
        ground = []
        for elem in (text_content2["rag_questions"]):
            text_1 = {}
            text_1.update({
                "question_id": f"{elem['question_id']}",
                "sources": elem['sources'],
            })
            ground.append((text_1))

    print(json.dumps(ground, indent=2))
    for elem in ground:
        for elem1 in elem["sources"]:
            print(elem1["last_character_index"])
