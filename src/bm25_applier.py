#!/usr/bin/env python3

import os
from pydantic import ValidationError
import bm25s
from tqdm import tqdm
from langchain_core.documents import Document
import torch
from transformers import pipeline
from src.explorer import Searcher
import json
from src import data_models as dm


class Bm25sApplier():
    """This class contain all methods that use bm25s"""
    retriever: bm25s.BM25

    @classmethod
    def tokenizer(cls, chunk_list: list[Document]) -> None:
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
                "./data/processed/Index_bm25s", corpus=corpus_saved
                )
            print("Index created and saved for next run!")
        except (ValidationError, Exception) as e:
            print(f"Error while index cration: {e}")

    @classmethod
    def bm25_index_inizialize(cls, k=2000) -> None:
        """This method create the index
        
        It search the full database, split all documents with the right function
        for each document type and save the result
        """
        if os.path.exists("./data/processed/Index_bm25s"):
            print("Index already exist! Ultrafast loading.")
        else:
            Bm25sApplier.tokenizer(Searcher.analizer(k))
        cls.retriever = bm25s.BM25.load(
            "./data/processed/Index_bm25s", load_corpus=True
            )

    @classmethod
    def search_single_query(cls, query: str, n: int) -> list[dict[str, str]]:
        """This method return the top k number resources
        
        Based on a single query this method uses bm25s criteria to return
        a list of dictionaries with the most significant data based on the
        given query.
        """
        cls.bm25_index_inizialize()
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
                "Source:": f"{metadata}",
                "First character index:": f"{first}",
                "Last character index:": f"{last}"
                })
        return ((text_list))

    @classmethod
    def search_dataset_query(
            cls,
            dataset_path: str,
            k: int,
            save_directory: str
            ) -> None:
        cls.bm25_index_inizialize()
        print("Elaborating query ...\n")
        result = []
        with open(dataset_path, encoding="utf-8") as source:
            text_content = json.load(source)
            for elem in tqdm(text_content["rag_questions"]):
                text_1 = []
                text_2 = cls.search_single_query(elem["question"], k)
                text_1.append({
                    "question_id:": f"{elem['question_id']}",
                    "question:": f"{elem['question']}",
                    "retrieved_sources:": text_2
                })
                result.append((text_1))
            cls.retriever.save(save_directory, corpus=result)
            print(json.dumps((result), indent=4))
            print("\n\n", save_directory, "\n")

    @classmethod
    def answer_single_query(cls, query: str, k: int) -> str:
        Bm25sApplier.bm25_index_inizialize()
        data_list = cls.search_single_query(query, k)
        context = Searcher.extractor(data_list[0])
        chat = f"""<|im_start|>system
This is your knowledge:
You must answer to a query from the user about your knowledge.
Do not think, give short direct answer.
{context}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
        llm = pipeline(
            task="text-generation",
            model="Qwen/Qwen3-0.6B",
            dtype=torch.bfloat16,
            device_map="auto"
            )
        blocked_ids: list[list] = []
        end_ids: list[list] = []
        if llm.tokenizer is not None:
            blocked_ids.append(llm.tokenizer.encode(
                "<think>", add_special_tokens=False
                ))
            blocked_ids.append(llm.tokenizer.encode(
                "</think>", add_special_tokens=False
                ))
            end_ids.append(llm.tokenizer.encode(
                ".", add_special_tokens=False))
        while(True):
            result = llm(
                chat, max_new_tokens=256,
                return_full_text=False,
                bad_words_ids=blocked_ids
                )[0]['generated_text']
            if len(result) > 120:
                break
        print(f"\n\n--{query}--\n")
        answer = result.rpartition('\n')[0]
        print(f"\n===\n{answer}\n===\n")
        return result

    @classmethod
    def answer_dataset_query(
            cls,
            student_search_results_path: str,
            save_directory: str
            ) -> None:
        with open(student_search_results_path, encoding="utf-8") as source:
            text_content = json.load(source)