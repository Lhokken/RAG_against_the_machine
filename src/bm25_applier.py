#!/usr/bin/env python3

import os
import bm25s
import torch
import transformers
import json
from pydantic import ValidationError
from typing import Any
from tqdm import tqdm
from langchain_core.documents import Document
from transformers import pipeline, logging
from src.explorer import Searcher
from src.data_models import MinimalSource, MinimalSearchResults
from src.data_models import StudentSearchResults, MinimalAnswer
from src.data_models import RagDataGround, MinimalRagData
from pathlib import Path


class Bm25sApplier():
    """This class contain all methods that use bm25s"""
    retriever: bm25s.BM25

    @classmethod
    def tokenizer(
            cls,
            chunk_list: list[Document],
            save_directory: str = "data/processed"
            ) -> None:
        """This method create the corpus based on a list of given text chunks

        It save the result in the ricght directory and return an error message
        if something goes wrong
        """
        print("Calculating Index ...\n")
        try:
            corpus_saved = [
                {
                    "Text": doc.page_content,
                    "metadati": doc.metadata
                } for doc in chunk_list]
            text = [doc.page_content for doc in chunk_list]
            corpus_tokens = bm25s.tokenize(text)
            cls.retriever = bm25s.BM25(k1=1.5, b=0.75)
            cls.retriever.index(corpus_tokens)
            cls.retriever.save(
                save_directory, corpus=corpus_saved
                )
            print("Ingestion complete! Indices saved under data/processed/")
        except (ValidationError, RuntimeWarning, Exception) as e:
            print(f"Error while index creation: {e}\n")
            exit()

    @classmethod
    def bm25_index_inizialize(
            cls,
            k: int = 2000,
            save_directory: str = "data/processed"
            ) -> None:
        """This method create the index

        It search the full database, split all documents
        with the right function for each document type
        and save the result
        """
        if os.path.exists("data/processed"):
            print("---\nIndex already exist! Ultrafast loading.\n---")
        else:
            Bm25sApplier.tokenizer(Searcher.analizer(k), save_directory)
        try:
            cls.retriever = bm25s.BM25.load(
                "data/processed", load_corpus=True
                )
        except (ValidationError, FileNotFoundError) as e:
            print(f"Loading error: {e}\nVerify and retry.\n")
            exit()

    @classmethod
    def search_single_query(cls, query: str, n: int) -> list[MinimalSource]:
        """This method return the top k number resources

        Based on a single query this method uses bm25s criteria to return
        a list of MinimalSource with the most significant data based on the
        given query.
        """
        query_tokens = bm25s.tokenize([query])
        if cls.retriever.corpus is None:
            raise Exception("Corpus not loaded or created. Verify.")
        result, scores = cls.retriever.retrieve(
            query_tokens, corpus=cls.retriever.corpus, k=n
            )
        docs_found = result[0]
        scores_found = scores[0]
        min_res: list[MinimalSource] = []
        for position, (doc, score) in enumerate(
                zip(docs_found, scores_found), 1
                ):
            metadata = doc["metadati"]["source"]
            first = doc["metadati"]["first_char_index"]
            last = doc["metadati"]["last_char_index"]
            min_res.append(MinimalSource.model_validate({
                "file_path": metadata[2:],
                "first_character_index": first,
                "last_character_index": last,
                }))
        return min_res

    @classmethod
    def search_dataset_query(
            cls,
            dataset_path: str,
            k: int,
            save_directory: str
            ) -> None:
        """This method search corpus about a list of question

        First apply search_single_query to each question of the list.
        Then save result in a file in the given directory.
        """
        print("Elaborating query ...\n")
        result: list[MinimalSearchResults] = []
        cls.bm25_index_inizialize()
        try:
            with open(dataset_path, encoding="utf-8") as source:
                text_content = json.load(source)
                for elem in tqdm(text_content["rag_questions"]):
                    text = cls.search_single_query(elem["question"], k)
                    result.append(MinimalSearchResults.model_validate({
                        "question_id": f"{elem['question_id']}",
                        "question": f"{elem['question']}",
                        "retrieved_sources": text
                    }))
                os.makedirs(save_directory, exist_ok=True)
                file_path = Path(dataset_path)
                file_path.name
                dict_result = StudentSearchResults.model_validate({
                    "search_results": result,
                    "k": k})
                Path(f"{save_directory}/{file_path.name}").write_text(
                    dict_result.model_dump_json(indent=2)
                    )
                print(
                    "---\nStudentSearchResults JSON file saved in:\n"
                    f"{save_directory}\n---\n"
                    )
        except (ValidationError, FileNotFoundError) as e:
            print(e)
            exit()

    @classmethod
    def answer_single_query(cls, query: str, k: int) -> None:
        """This method coordinate two methods

        First search_single_query to obtain the right data to analize,
        then call single_query with question and data.
        """
        cls.bm25_index_inizialize()
        try:
            if k <= 0:
                raise IndexError
            else:
                data_list = cls.search_single_query(query, k)
                result = (MinimalSearchResults.model_validate({
                    "question_id": "",
                    "question": query,
                    "retrieved_sources": data_list
                }))
        except IndexError as e:
            print(f"Parameter k must be >0: {e}")
            exit()
        print("\n", cls.single_query(result).answer)

    @classmethod
    def single_query(cls, text: MinimalSearchResults) -> MinimalAnswer:
        """This method use qwen to answer a single question

        After buildind the prompt with query and context, decide to use
        cuda or cpu, activate the llm model qwen and finally return the result.
        """
        context: str = ""
        for source in tqdm(text.retrieved_sources):
            context += (
                Searcher.extractor(MinimalSource.model_validate(source))
                + "\n"
                )
        query = text.question
        chat = f"""<|im_start|>system
Output ONLY the factual answer.
Start your response immediately with the requested information.
{context}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
        if len(context) > 9000:
            dev_type = "cpu"
        else:
            dev_type = "cuda"
        print(
            f"Context lenght: {len(context)}\n"
            f"Device used: {dev_type}\n"
            )
        logging.set_verbosity_error()
        llm = pipeline(
            task="text-generation",
            model="Qwen/Qwen3-0.6B",
            dtype=torch.bfloat16,
            device=dev_type
            )
        blocked_ids: list[list[int] | Any] = []
        if llm.tokenizer is not None:
            blocked_ids.append(llm.tokenizer.encode(
                "<think>", add_special_tokens=False
                ))
            blocked_ids.append(llm.tokenizer.encode(
                "</think>", add_special_tokens=False
                ))
        result: str = ""
        while(True):
            try:
                result = llm(
                    chat,
                    max_new_tokens=64,
                    return_full_text=False,
                    bad_words_ids=blocked_ids
                    )[0]['generated_text']
            except (ValidationError, RuntimeError, Exception) as e:
                print(f"\nMemory error: {e}=====\n")
                exit()
            if len(result.strip()) > 50:
                break
        answer = MinimalAnswer.model_validate({
            "question_id": text.question_id,
            "question": text.question,
            "retrieved_sources": text.retrieved_sources,
            "answer": result.rpartition('\n')[0].strip()
        })
        return answer

    @classmethod
    def answer_dataset_query(
            cls,
            student_search_results_path: str,
            save_directory: str
            ) -> None:
        """This method apply single_query to each question.

        After obtaining result from single_query, it append the result
        to answer_list.
        """
        transformers.logging.set_verbosity_error()
        question_list = []
        try:
            with open(student_search_results_path, encoding="utf-8") as source:
                question_list = json.load(source)["search_results"]
        except (ValidationError, FileNotFoundError) as e:
            print(e)
            exit()
        answer_list = RagDataGround(rag_questions=[])

        cls.bm25_index_inizialize()
        counter: int = 0
        for elem in tqdm(question_list):
            counter += 1
            sources = (elem)["retrieved_sources"]
            question: str = (elem)["question"]
            question_id: str = (elem)["question_id"]

            print(f"----------\nElaborating query number: {counter}")

            result = (MinimalSearchResults.model_validate({
                "question_id": question_id,
                "question": question,
                "retrieved_sources": sources
            }))

            answer = cls.single_query(result).answer
            print(
                "\n\n=== Answer: ===\n"
                f"{answer}\n"
                "======\n"
                )

            n_result = (MinimalRagData.model_validate({
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "sources": sources,
                "difficulty": "easy",
                "is_valid": True
            }))
            answer_list.rag_questions.append(n_result)
        file_path = Path(student_search_results_path)
        os.makedirs(save_directory, exist_ok=True)
        Path(f"{save_directory}/{file_path.name}").write_text(
                    answer_list.model_dump_json(indent=2)
                    )
        print(
            f"\nLoaded 100 questions ... Processed {counter} "
            F"of {len(question_list)} questions\n"
            f"Save student_search_results_and_answer to .../{save_directory}")

        print("\a")
