#!/usr/bin/env python3

import os
import bm25s
from explorer import Searcher
from bm25_applier import Bm25sApplier


if __name__ == "__main__":
    if os.path.exists("Data/data_processed/Index_bm25s"):
        print("Index already exist! Ultrafast loading.")
        retriever = bm25s.BM25.load(
            "Data/data_processed/Index_bm25s", load_corpus=True
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
