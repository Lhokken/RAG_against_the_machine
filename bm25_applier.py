#!/usr/bin/env python3

from pydantic import BaseModel, ValidationError
import bm25s
import bm25s.tokenization


class Bm25sApplier(BaseModel):
	txt_tokens: list

	def tokenizer(self, chunk_list: list[str]) -> None:
