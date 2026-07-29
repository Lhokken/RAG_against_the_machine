#!/usr/bin/env python3

from pydantic import BaseModel, ValidationError
from langchain_text_splitters import RecursiveCharacterTextSplitter # .txt
from langchain_text_splitters import MarkdownHeaderTextSplitter # .md
from langchain_text_splitters import PythonCodeTextSplitter #  Language # con rec_car_text_splitter .py
from pprint import pprint

class Splitter(BaseModel):
	chunk_list_txt: list[str] = [];
	chunk_list_md: list[str] = [];
	chunk_list_py: list[str] = [];

	def split_all(self, file_list: list[str]) -> None:
		txt_chunker = RecursiveCharacterTextSplitter(
			chunk_size=1900, chunk_overlap=150)
		try:
			for file in file_list:
				if file.lower().endswith(".txt"):
					with open(file) as source:
						temp = txt_chunker.split_text(source.read())
						self.chunk_list_txt.extend(temp)
						
				elif file.lower().endswith(".md"):
					with open(file) as source:
						temp = txt_chunker.split_text(source.read())
						self.chunk_list_md.extend(temp)
				elif file.lower().endswith(".py"):
					with open(file) as source:
						temp = txt_chunker.split_text(source.read())
						self.chunk_list_py.extend(temp)
		except ValidationError as e:
			print(e)

	def print_chunk_list(self) -> None:
		for string in self.chunk_list_txt:
			print("-" * 50)
			pprint(string)
			print("-" * 50)
		for string in self.chunk_list_md:
			print("-" * 50)
			pprint(string)
			print("-" * 50)
		for string in self.chunk_list_py:
			print("-" * 50)
			pprint(string)
			print("-" * 50)

