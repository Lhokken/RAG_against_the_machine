#!/usr/bin/env python3

from explorer import Searcher
from splitter import Splitter


if __name__ == "__main__":
	explorer = Searcher()
	splitter = Splitter()
	explorer.search_all()
	# explorer.print_file_list()
	splitter.split_all(explorer.file_list)
	splitter.print_chunk_list()