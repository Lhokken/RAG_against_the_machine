#!/usr/bin/env python3

from pydantic import BaseModel, ValidationError
import bm25s
import bm25s.tokenization


class Bm25sApplier(BaseModel):

	def tokenizer(self, chunk_list: list[str]) -> None:
		try:
			chk_list_tokenized = bm25s.tokenize(chunk_list)
			retriever = bm25s.BM25()
			retriever.index(chk_list_tokenized)
			retriever.save("Index_bm25s", corpus=chunk_list)
		except ValidationError as e:
			print(e)
