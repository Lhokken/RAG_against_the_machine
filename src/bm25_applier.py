#!/usr/bin/env python3

import os
from pydantic import ValidationError
import bm25s
# import acc
from langchain_core.documents import Document
import torch
from transformers import pipeline
from src.explorer import Searcher
import json
from pprint import pprint

class Bm25sApplier():
    retriever: bm25s.BM25 

    @classmethod
    def bm25_index_inizialize(cls) -> None:
        if os.path.exists("./data/processed/Index_bm25s"):
            print("Index already exist! Ultrafast loading.")
        else:
            Bm25sApplier.tokenizer(Searcher.analizer(1900))
        cls.retriever = bm25s.BM25.load(
        "./data/processed/Index_bm25s", load_corpus=True
        )

    @classmethod
    def tokenizer(cls, chunk_list: list[Document]) -> list[Document] | None:
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
    def single_query(cls, query: str, n: int) -> list[dict]:
        # Bm25sApplier.bm25_index_inizialize()
        query_tokens = bm25s.tokenize([query])
        if cls.retriever.corpus is None:
            raise ValueError("Corpus not loaded or created. Verify.")
        result, scores = cls.retriever.retrieve(
            query_tokens, corpus=cls.retriever.corpus, k=n
            )
        docs_found = result[0]
        scores_found = scores[0]
        text_list = []
        for position, (doc, score) in enumerate(zip(docs_found, scores_found), 1):
            metadata = doc["metadati"]["source"]
            first = doc["metadati"]["first_char_index"]
            last = doc["metadati"]["last_char_index"]
            text_list.append({
                f"Results {position}:": f"score BM25: {score:.2f}",
                f"Source:": f"{metadata}",
                f"First character index:": f"{first}",
                f"Last character index:": f"{last}"
                })
        return ((text_list))

    @classmethod
    def search_dataset_query(cls, dataset_path: str, k: int, save_directory: str) -> None:
        cls.bm25_index_inizialize()
        print("Elaborating query ...\n")
        result = []
        with open(dataset_path, encoding="utf-8") as source:
            text_content = json.load(source)
            for elem in text_content["rag_questions"]:
                text_1 = []
                text_2 = cls.single_query(elem["question"], k)
                text_1.append({
                    f"question:": f"{elem['question']}",
                    f"question_id:": f"{elem['question_id']}",
                    "retrieved_sources:": text_2
                })
                result.append((text_1))
            cls.retriever.save(save_directory, corpus=result)
            print(json.dumps((result), indent=4))
            print("\n\n", save_directory, "\n")


    @classmethod
    def answer_single_query(cls, query: str, k) -> None:
        Bm25sApplier.bm25_index_inizialize()
        data_list = cls.single_query(query, k)
        # print(json.dumps(data_list[0], indent=2))
        context = Searcher.extractor(data_list[0])
        # print(context)

        chat = [
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ]
        llm = pipeline(
            task="text-generation",
            model="Qwen/Qwen3-0.6B",
            dtype=torch.bfloat16,
            device_map="auto"
            )
        result = llm(chat, max_new_tokens=64)
        print("-"*10)
        print(f"--\n{query}\n--\n")
        print(result[0]["generated_text"][-1]["content"])
        print("-"*10)

        