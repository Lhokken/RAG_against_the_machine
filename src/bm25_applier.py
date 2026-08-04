#!/usr/bin/env python3

import os
from pydantic import ValidationError
import bm25s
from langchain_core.documents import Document
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
    def single_query(cls, query: str, n) -> None:
        query_tokens = bm25s.tokenize([query])
        if cls.retriever.corpus is None:
            raise ValueError("Corpus not loaded or created. Verify.")
        result, scores = cls.retriever.retrieve(
            query_tokens, corpus=cls.retriever.corpus, k=n
            )
        docs_found = result[0]
        scores_found = scores[0]
        print(f"\n\nResults for --- {query} ---\n")
        text_list = []
        for position, (doc, score) in enumerate(zip(docs_found, scores_found), 1):
            text = doc["Text"]
            metadata = doc["metadati"]["source"]
            first = doc["metadati"]["first_char_index"]
            last = doc["metadati"]["last_char_index"]
            text_list.append({
                f"Results {position}:": f"score BM25: {score:.2f}",
                f"Source:": f"{metadata}",
                f"First character index:": f"{first}",
                f"Last character index:": f"{last}"
                })
        print(json.dumps(text_list, indent=4))

    @classmethod
    def search_dataset_query(cls, dataset_path: str, k: int, save_directory: str) -> None:
        print("Elaborating query ...\n")
        cls.bm25_index_inizialize()
        with open(dataset_path, encoding="utf-8") as source: 
            text_content = json.load(source)
            for elem in text_content["rag_questions"]:
                print("question_id: ", elem["question_id"])
                print("question: ", elem["question"], "\n")
                print("retrieved_sources: ")
                cls.single_query(elem["question"], k)
                # print(elem, elem["question_id"], "\n")
            print("\n\n", save_directory, "\n")



            # StudentSearchResults(BaseModel):
            # {search_results: list[
            #     question_id: str
            #     question: str
            #     retrieved_sources: list[MinimalSource]
            #     ]
            # k: int}
