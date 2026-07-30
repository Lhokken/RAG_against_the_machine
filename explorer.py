#!/usr/bin/env python3

import os
import re
from pydantic import BaseModel, ValidationError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_text_splitters import PythonCodeTextSplitter
from langchain_core.documents import Document
from pprint import pprint


class Searcher(BaseModel):
    root_directory: str = "vllm-0.10.1"
    file_list: list[str] = []
    chunk_list: list[(Document)] = []
    valid_extensions: tuple[str, str, str] = (".py", ".txt", ".md")

    def search_all(self) -> None:
        try:
            for root, _, files in os.walk(self.root_directory):
                for file in files:
                    if file.lower().endswith(self.valid_extensions):
                        file_path = os.path.join(root, file)
                        self.file_list.append(file_path)
        except ValidationError as e:
            print(e)

    def print_file_list(self) -> None:
        pprint(self.file_list)

    def split_all(self, file_list: list[str]) -> None:
        html_pattern = r'</?(h[1-6]|p|div|span|br|table)[^>]*>'
        txt_chunker = RecursiveCharacterTextSplitter(
            chunk_size=1900,
            chunk_overlap=150,
            add_start_index=True,
            length_function=len
            )
        md_chunker = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Title_1_H1"),
                ("##", "Title_2_H2"),
                ("###", "Title_3_H3")])
        html_chunker = HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "Title_1_H1"),
                ("h2", "Title_2_H2"),
                ("h3", "Title_3_H3")
            ]
        )
        py_chunker = PythonCodeTextSplitter()
        temp: list[Document]
        for file in file_list:
            try:
                temp = []
                with open(file, encoding="utf-8") as source:
                    if file.lower().endswith(".txt"):
                        text = Document(
                            page_content=source.read(),
                            metadata={"source": file}
                            )
                        temp = txt_chunker.split_documents([text])

                    elif file.lower().endswith(".md"):
                        text_content = source.read()
                        if re.search(
                                html_pattern, text_content, re.IGNORECASE
                                ):
                            temp = html_chunker.split_text(text_content)
                        else:
                            temp = md_chunker.split_text(text_content)
                        for doc in temp:
                            doc.metadata["source"] = file
                        temp = txt_chunker.split_documents(temp)

                    elif file.lower().endswith(".py"):
                        temp = py_chunker.create_documents(
                            texts=[source.read()],
                            metadatas=[{"source": file}]
                            )
                        temp = txt_chunker.split_documents(temp)
                    if temp != []:
                        self.chunk_list.extend(temp)

            except (
                ValidationError,
                FileNotFoundError,
                PermissionError,
                UnicodeDecodeError,
                Exception
                ) as e:
                print(f"Error in elaborating file {file}: {e}")

    def print_chunk_list(self) -> None:
        for string in self.chunk_list:
            print("-" * 50)
            pprint(string)
            print("-" * 50)

