#!/usr/bin/env python3

import os
import re
from pydantic import ValidationError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_text_splitters import PythonCodeTextSplitter
from langchain_core.documents import Document
from pprint import pprint


class Searcher():
    root_directory: str = "./data/raw/vllm-0.10.1"
    file_list: list[str] = []
    chunk_list: list[(Document)] = []
    valid_extensions: tuple[str, str, str] = (".py", ".txt", ".md")
    max_chunk_size = 2000

    @classmethod
    def analizer(cls, new_max_chunk_size: int) -> list[Document]:
        cls.max_chunk_size = new_max_chunk_size
        cls.search_all()
        cls.split_all(cls.file_list)

        return cls.chunk_list

    @classmethod
    def search_all(cls) -> None:
        try:
            for root, _, files in os.walk(cls.root_directory):
                for file in files:
                    if file.lower().endswith(cls.valid_extensions):
                        file_path = os.path.join(root, file)
                        cls.file_list.append(file_path)
        except (ValidationError, Exception) as e:
            print(e)

    @classmethod
    def print_file_list(cls) -> None:
        pprint(cls.file_list)

    @classmethod
    def split_all(cls, file_list: list[str]) -> None:
        html_pattern = r'</?(h[1-6]|p|div|span|br|table)[^>]*>'
        txt_chunker = RecursiveCharacterTextSplitter(
            chunk_size=cls.max_chunk_size,
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
                search_start = 0
                with open(file, encoding="utf-8") as source:
                    text_content = source.read()
                    if file.lower().endswith(".txt"):
                        text = Document(
                            page_content=text_content,
                            metadata={"source": file}
                            )
                        temp = txt_chunker.split_documents([text])

                    elif file.lower().endswith(".md"):
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
                        for doc in temp:
                            first_index = text_content.find(
                                doc.page_content, search_start
                                )
                            if first_index != -1:
                                last_index = first_index + len(
                                    doc.page_content
                                    )
                                search_start = first_index + 1
                            else:
                                first_index = doc.metadata.get(
                                    "start_index", 0
                                    )
                                last_index = first_index + len(
                                    doc.page_content
                                    )
                            doc.metadata["first_char_index"] = first_index
                            doc.metadata["last_char_index"] = last_index
                            if "start_index" in doc.metadata:
                                del doc.metadata["start_index"]

                        cls.chunk_list.extend(temp)

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

    @classmethod
    def extractor(cls, data_file: dict[str, str]) -> str:
        with open(data_file["Source:"],  encoding="utf-8") as source:
            text_content = source.read()
        start = int(data_file["First character index:"])
        end = int(data_file["Last character index:"])
        return text_content[start:end]
