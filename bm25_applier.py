#!/usr/bin/env python3

from typing import Any
from pydantic import BaseModel, ValidationError
import bm25s


class Bm25sApplier(BaseModel):

	def tokenizer(self, chunk_list: list[str]) -> None:
		try:
			chk_list_tokenized = bm25s.tokenize(chunk_list)
			retriever = bm25s.BM25()
			retriever.index(chk_list_tokenized)
			retriever.save("Index_bm25s", corpus=chunk_list)
			print("Index created and saved for next run!")
		except ValidationError as e:
			print(e)
