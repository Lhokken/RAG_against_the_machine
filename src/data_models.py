#!/usr/bin/env python3

from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """Class for chunk data"""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Class for question without answer, assign unique identifier"""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Class for question with answer"""
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Class for dta block with answered and unanswered questions"""
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Class for questions and N sources of file chunks"""
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Class used to add answer to a MinimalSearchResults dataset"""
    answer: str


class StudentSearchResults(BaseModel):
    """Class to create a dataset of N questions and relative N chunks"""
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Class used to create a dataset of questions, answers and chunks"""
    search_results: list[MinimalAnswer]
    k: int


class MinimalRagData(BaseModel):
    """Class used to create a complete dataset

    with question, answer, chunks and other relevant data"""
    question_id: str
    question: str
    answer: str
    sources: list[MinimalSource]
    difficulty: str
    is_valid: bool


class RagDataGround(BaseModel):
    """Class used to contain MinimalRagData"""
    rag_questions: list[MinimalRagData]
