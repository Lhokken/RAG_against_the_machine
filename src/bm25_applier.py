#!/usr/bin/env python3

import os
from pydantic import ValidationError
from typing import Any
import bm25s
from tqdm import tqdm
from langchain_core.documents import Document
import torch
import transformers
from transformers import pipeline
from src.explorer import Searcher
import json
# from src import data_models as dm


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
        except (ValidationError, Exception) as e:
            print(f"Error while index cration: {e}")

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
        cls.retriever = bm25s.BM25.load(
            "data/processed", load_corpus=True
            )

    @classmethod
    def search_single_query(cls, query: str, n: int) -> list[dict[str, str]]:
        """This method return the top k number resources

        Based on a single query this method uses bm25s criteria to return
        a list of dictionaries with the most significant data based on the
        given query.
        """
        query_tokens = bm25s.tokenize([query])
        if cls.retriever.corpus is None:
            raise ValueError("Corpus not loaded or created. Verify.")
        result, scores = cls.retriever.retrieve(
            query_tokens, corpus=cls.retriever.corpus, k=n
            )
        docs_found = result[0]
        scores_found = scores[0]
        text_list: list[dict[str, str]] = []
        for position, (doc, score) in enumerate(
                zip(docs_found, scores_found), 1
                ):
            metadata = doc["metadati"]["source"]
            first = doc["metadati"]["first_char_index"]
            last = doc["metadati"]["last_char_index"]
            text_list.append({
                "file_path": metadata[2:],
                "first_character_index": first,
                "last_character_index": last
                })
        return ((text_list))

    @classmethod
    def search_dataset_query(
            cls,
            dataset_path: str,
            k: int,
            save_directory: str,
            save_file: str
            ) -> None:
        """This method search corpus about a list of question

        First apply search_single_query to each question of the list.
        Then save result in a file in the given directory.
        """
        print("Elaborating query ...\n")
        result = []
        cls.bm25_index_inizialize()
        with open(dataset_path, encoding="utf-8") as source:
            text_content = json.load(source)
            for elem in tqdm(text_content["rag_questions"]):
                text_1 = {}
                text_2 = cls.search_single_query(elem["question"], k)
                text_1.update({
                    "question_id": f"{elem['question_id']}",
                    "question": f"{elem['question']}",
                    "retrieved_sources": text_2
                })
                result.append((text_1))
            os.makedirs(save_directory, exist_ok=True)
            dict_result = {"search_results": result, "k": k}
            with open(
                    f"{save_directory}/{save_file}", "w"
                    ) as file_output:
                json.dump(dict_result, file_output, indent=2)
            print("---\nStudentSearchResults JSON file saved in:\n"
                  f"{save_directory}\n---\n")

    @classmethod
    def answer_single_query(cls, query: str, k: int) -> None:
        """This method coordinate two methods

        First search_single_query to obtain the right data to analize,
        then call single_query with question and data.
        """
        context: str = ""
        try:
            if k <= 0:
                raise IndexError
            else:
                data_list = cls.search_single_query(query, k)
                context = json.dumps(data_list, indent=2)
        except IndexError as e:
            print(f"Parameter k must be >0: {e}")
            exit()
        cls.single_query(query, context)

    @classmethod
    def single_query(cls, query: str, context: str) -> str:
        """This method use qwen to answer a single question

        After buildind the prompt with query and context, decide to use
        cuda or cpu, activate the llm model qwen and finally return the result.
        """
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
        answer = result.rpartition('\n')[0].strip()
        return answer

    @classmethod
    def answer_dataset_query(
            cls,
            student_search_results_path: str,
            save_directory: str,
            save_file: str
            ) -> None:
        """This method apply single_query to each question.

        After obtaining result from single_query, it append the result
        to answer_list.
        """
        transformers.logging.set_verbosity_error()
        with open(student_search_results_path, encoding="utf-8") as source:
            question_list = json.load(source)["search_results"]
        answer_list = []
        n_result: dict[str, str] = {}
        cls.bm25_index_inizialize()
        counter: int = 0
        for elem in tqdm(question_list):
            counter += 1
            sources = (elem)["retrieved_sources"]
            question = (elem)["question"]
            question_id = (elem)["question_id"]
            context: str = ""
            for source in tqdm(sources):
                context += Searcher.extractor(source)
            print(f"----------\nElaborating query number: {counter}")
            answer = cls.single_query(question, context)
            print(
                "\n=== Answer: ===\n"
                f"{answer}\n"
                "======\n"
                )
            n_result = {
                "question_id": question_id,
                "question": question,
                "retrieved_sources": sources,
                "answer": answer
            }
            answer_list.append(n_result)
            # if counter == 10:
            #     break
        final_list = {"search_results": answer_list,
                      "k": len(question_list[0]["retrieved_sources"])}
        os.makedirs(save_directory, exist_ok=True)
        with open(
                f"{save_directory}/{save_file}", "w"
                ) as file_output:
            json.dump(final_list, file_output, indent=2)

        print(
            f"\nLoaded 100 questions ... Processed {counter} "
            F"of {len(question_list)} questions\n"
            f"Save student_search_results_and_answer to .../{save_directory}")

        print("\a")
