#!/usr/bin/env python3

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import PythonCodeTextSplitter
from langchain_core.documents import Document
from pydantic import ValidationError
from src.data_models import MinimalSource
from pprint import pprint


class Searcher():
    """This class contains methods for database chunking
    """
    root_directory: str = "./data/raw/vllm-0.10.1"
    file_list: list[str] = []
    chunk_list: list[(Document)] = []
    valid_extensions: tuple[str, str, str] = (".py", ".txt", ".md")
    max_chunk_size = 2000
    overlap = (max_chunk_size * 15 // 100)

    @classmethod
    def analizer(cls, new_max_chunk_size: int) -> list[Document]:
        """This method collect other methods of this class
        """
        try:
            if new_max_chunk_size < 2000:
                cls.max_chunk_size = new_max_chunk_size
                cls.overlap = (cls.max_chunk_size * 15 // 100)
            cls.search_all()
            cls.split_all(cls.file_list)
            if len(cls.file_list) == 0:
                raise FileNotFoundError("\nAbsent data, file not found.\n")
        except (ValidationError, FileNotFoundError) as e:
            print(e)
            exit()
        return cls.chunk_list

    @classmethod
    def search_all(cls) -> None:
        """This method create the list of files

        and relatives filepath
        """
        try:
            for root, _, files in os.walk(cls.root_directory):
                for file in files:
                    if file.lower().endswith(cls.valid_extensions):
                        file_path = os.path.join(root, file)
                        cls.file_list.append(file_path)
        except (ValidationError, RuntimeWarning, Exception) as e:
            print(e)
            exit()

    @classmethod
    def print_file_list(cls) -> None:
        """Simple print method"""
        pprint(cls.file_list)

    @classmethod
    def split_all(cls, file_list: list[str]) -> None:
        """Main method of this class

        This method apply the right splitter function for file types:
        text, html, and .py.
        Then create file chunks.
        """
        txt_chunker = RecursiveCharacterTextSplitter(
            chunk_size=cls.max_chunk_size,
            chunk_overlap=cls.overlap,
            add_start_index=True,
            length_function=len
            )

        py_chunker = PythonCodeTextSplitter(
            chunk_size=cls.max_chunk_size,
            chunk_overlap=cls.overlap
        )
        temp: list[Document]
        for file in file_list:
            try:
                temp = []
                search_start = 0
                with open(file, encoding="utf-8") as source:
                    text_content = source.read()
                    if file.lower().endswith(".txt") \
                            or file.lower().endswith(".md"):
                        text = Document(
                            page_content=text_content,
                            metadata={"source": file}
                            )
                        temp = txt_chunker.split_documents([text])

                    elif file.lower().endswith(".py"):
                        temp = py_chunker.create_documents(
                            texts=[text_content],
                            metadatas=[{"source": file}]
                            )

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
                exit()

    def print_chunk_list(self) -> None:
        """Debugging function, used for test control
        """
        for string in self.chunk_list:
            print("-" * 50)
            pprint(string)
            print("-" * 50)

    @classmethod
    def extractor(cls, data_file: MinimalSource) -> str:
        """This method is used to extract chunk from the corpus
        """
        try:
            with open(data_file.file_path,  encoding="utf-8") as source:
                text_content = source.read()
        except FileNotFoundError as e:
            print(e)
            exit()
        start = int(data_file.first_character_index)
        end = int(data_file.last_character_index)
        return text_content[start:end]
