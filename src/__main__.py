#!/usr/bin/env python3

import os
import bm25s
import fire
# from commands import CommandList
from explorer import Searcher
from bm25_applier import Bm25sApplier


if __name__ == "__main__":
    if os.path.exists("./data/processed/Index_bm25s"):
        print("Index already exist! Ultrafast loading.")
        retriever = bm25s.BM25.load(
            "./data/processed/Index_bm25s", load_corpus=True
            )
        # print(retriever.corpus)
    else:
        print("Index do not exist! Calculating ...")
        explorer = Searcher()
        explorer.search_all()
        # explorer.print_file_list()
        explorer.split_all(explorer.file_list)
        # explorer.print_chunk_list()

        bm25_indexer = Bm25sApplier()
        bm25_indexer.tokenizer(explorer.chunk_list)
        retriever = bm25s.BM25.load(
            "./data/processed/Index_bm25s", load_corpus=True
            )
    # Commander = CommandList()

    question: str = ""
    question = input("Enter your question: ")
    query_tokens = bm25s.tokenize([question])
    if retriever.corpus is None:
        raise ValueError("Corpus not loaded or created. Verify.")
    result, scores = retriever.retrieve(
        query_tokens, corpus=retriever.corpus, k=4
        )
    docs_found = result[0]
    scores_found = scores[0]
    print(f"Results for --- {question} ---\n")
    for position, (doc, score) in enumerate(zip(docs_found, scores_found), 1):
        text = doc["Text"]
        metadata = doc["metadati"]["source"]
        print(
            f"results {position} - score BM25: {score:.2f}"
            " - source: {metadata}\n"
            )

        print(f"Text: {text[:200]}\n")
