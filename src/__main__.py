#!/usr/bin/env python3

import os
import bm25s
import fire
from src.commands import CLI
from src.explorer import Searcher
from src.bm25_applier import Bm25sApplier
import sys


if __name__ == "__main__":
    fire.Fire(CLI)
    # sys.exit(1)

    # CLI.index(1900)
    # retriever = bm25s.BM25.load(
    #     "./data/processed/Index_bm25s", load_corpus=True
    #     )

    # question: str = ""
    # question = input("Enter your question: ")
    # query_tokens = bm25s.tokenize([question])
    # if retriever.corpus is None:
    #     raise ValueError("Corpus not loaded or created. Verify.")
    # result, scores = retriever.retrieve(
    #     query_tokens, corpus=retriever.corpus, k=4
    #     )
    # docs_found = result[0]
    # scores_found = scores[0]
    # print(f"\n\n\nResults for --- {question} ---\n")
    # for position, (doc, score) in enumerate(zip(docs_found, scores_found), 1):
    #     text = doc["Text"]
    #     metadata = doc["metadati"]["source"]
    #     print(
    #         f"results {position} - score BM25: {score:.2f}"
    #         " - source: {metadata}\n"
    #         )

    #     print(f"Text: {text[:200]}\n")
