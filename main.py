#!/usr/bin/env python3

import os
import bm25s
from explorer import Searcher
from splitter import Splitter
from bm25_applier import Bm25sApplier


if __name__ == "__main__":
    if os.path.exists("Index_bm25s"):
        print("Index already exist! Ultrafast loading.")
        retriever = bm25s.BM25.load("Index_bm25s", load_corpus=True)
        # print(retriever.corpus)
    else:
        print("Index do not exist! Calculating ...")
        explorer = Searcher()
        splitter = Splitter()
        explorer.search_all()
        # explorer.print_file_list()
        splitter.split_all(explorer.file_list)
        # splitter.print_chunk_list()

        bm25_indexer = Bm25sApplier()
        bm25_indexer.tokenizer(splitter.chunk_list)
