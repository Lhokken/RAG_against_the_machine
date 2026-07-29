#!/usr/bin/env python3

import os
from pydantic import BaseModel, ValidationError
from pprint import pprint


class Searcher(BaseModel):
	root_directory: str = "vllm-0.10.1";
	file_list: list[str] = [];
	valid_extensions: tuple[str, str, str] = (".py", ".txt", ".md")

	def search_all(self) -> None:
		try:
			for root, _, files in os.walk(self.root_directory):
				for file in files:
					if file.lower().endswith(self.valid_extensions):
						file_path = os.path.join(root, file);
						self.file_list.append(file_path);
		except ValidationError as e:
			print(e)

	def print_file_list(self) -> None:
		pprint(self.file_list)
