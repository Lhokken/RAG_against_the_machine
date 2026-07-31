# #!/usr/bin/env python3

# import re
# from pydantic import BaseModel, ValidationError
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters import MarkdownHeaderTextSplitter
# from langchain_text_splitters import HTMLHeaderTextSplitter
# from langchain_text_splitters import PythonCodeTextSplitter
# from langchain_core.documents import Document
# from pprint import pprint


# class Splitter(BaseModel):
#     chunk_list: list[Document] = []

#     def split_all(self, file_list: list[str]) -> None:
#         html_pattern = r'</?(h[1-6]|p|div|span|br|table)[^>]*>'
#         txt_chunker = RecursiveCharacterTextSplitter(
#             chunk_size=1900, chunk_overlap=150, add_start_index=True)
#         md_chunker = MarkdownHeaderTextSplitter(
#             headers_to_split_on=[
#                 ("#", "Title_1_H1"),
#                 ("##", "Title_2_H2"),
#                 ("###", "Title_3_H3")])
#         html_chunker = HTMLHeaderTextSplitter(
#             headers_to_split_on=[
#                 ("h1", "Title_1_H1"),
#                 ("h2", "Title_2_H2"),
#                 ("h3", "Title_3_H3")
#             ]
#         )
#         py_chunker = PythonCodeTextSplitter()
#         try:
#             for file in file_list:
#                 if file.lower().endswith(".txt"):
#                     with open(file) as source:
#                         temp = txt_chunker.create_documents([source.read()])
#                         self.chunk_list.extend(temp)

#                 elif file.lower().endswith(".md"):
#                     with open(file) as source:
#                         if re.search(
#                                 html_pattern, source.read(), re.IGNORECASE
#                                 ):
#                             temp = html_chunker.split_text(source.read())
#                         else:
#                             temp = md_chunker.split_text(source.read())
#                         self.chunk_list.extend(
#                             txt_chunker.split_documents(temp)
#                             )

#                 elif file.lower().endswith(".py"):
#                     with open(file) as source:
#                         temp = py_chunker.create_documents([source.read()])
#                         self.chunk_list.extend(temp)
#         except ValidationError as e:
#             print(e)

#     def print_chunk_list(self) -> None:
#         for string in self.chunk_list:
#             print("-" * 50)
#             pprint(string)
#             print("-" * 50)
