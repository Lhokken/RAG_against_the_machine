#!/usr/bin/env python3

from pydantic import BaseModel, ValidationError
import bm25s
from langchain_core.documents import Document


class Bm25sApplier(BaseModel):

    def tokenizer(self, chunk_list: list[Document]) -> None:
        try:
            corpus_saved = [
                {
                    "Text": doc.page_content,
                    "metadati": doc.metadata
                } for doc in chunk_list]
            corpus_tokens = bm25s.tokenize(
                [doc.page_content for doc in chunk_list]
                )
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)
            retriever.save("Data/data_processed/Index_bm25s", corpus=corpus_saved)
            print("Index created and saved for next run!")
        except ValidationError as e:
            print(e)
