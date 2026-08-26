*This project was created as part of the 42 curriculum by gcerrete.*

# RAG Against the Machine

## Description
This project implements a local Retrieval-Augmented Generation (RAG) system designed to index, search, and answer questions based on a codebase or text repository (defaulting to the `vllm-0.10.1` source code). The system extracts relevant information using the BM25 retrieval algorithm and synthesizes factual answers using a locally hosted Large Language Model (Qwen 3 0.6B). The pipeline is highly modular, separating document chunking, indexing, searching, and evaluation into distinct, well-defined stages.

## Instructions
### Requirements and Installation
To run this project, ensure you have Python 3.10+ installed. You will need the following key libraries:
* `bm25s`
* `torch`
* `transformers`
* `langchain-text-splitters`
* `pydantic`
* `fire`
* `tqdm`

Install dependencies via make:
```bash
make install
```

### Execution
The system provides a Command Line Interface (CLI) built with `fire`, accessible via your main entry point in the project root:

1. **Build the Index:**
   ```bash
   make index
   ```
2. **Search a Single Query:**
   ```bash
   make search
   ```
3. **Answer a Single Query (RAG):**
   ```bash
   make answer
   ```
4. **Process a Whole Dataset:**
   ```bash
   make search_dataset
   ```
5. **Evaluate System (Recall@k):**
   ```bash
   make evaluate
   ```

## System architecture
The pipeline is structured around several core components:
* **`Searcher` (Data Ingestion & Chunking):** Scans the raw data directories, filters for valid file types (`.py`, `.txt`), and applies type-specific segmentation.
* **`Bm25sApplier` (Retrieval Engine):** Manages the tokenization, index building, and saving/loading of the BM25 corpus using the `bm25s` library.
* **`Qwen 3 LLM` (Generation):** A Hugging Face `transformers` pipeline using `Qwen/Qwen3-0.6B` loaded in `bfloat16`. It dynamically switches between `cuda` and `cpu` based on the context length to prevent VRAM overflow.
* **`Pydantic Models` (Validation):** Ensures strict schema adherence for datasets, queries, and chunk structures (e.g., `StudentSearchResults`, `RagDataGround`).

## Chunking strategy
Document segmentation is handled by LangChain splitters tailored to the file format:
* **Python Files:** Uses `PythonCodeTextSplitter` to respect classes and functions, preventing code blocks from being arbitrarily sliced.
* **Markdown & Text Files:** Uses `RecursiveCharacterTextSplitter`.
* **Sizing:** The maximum chunk size is dynamically adjustable (defaulting to 2000 characters) with a 15% overlap to ensure context continuity. 
* **Metadata Tracking:** Each chunk strictly tracks its `first_character_index` and `last_character_index` relative to the original source file. This is critical for precise downstream evaluation.

## Retrieval method
The retrieval mechanism relies on the **BM25 algorithm** (via the `bm25s` library) configured with $k_1 = 1.5$ and $b = 0.75$.
* When a query is passed, it is tokenized and matched against the pre-computed corpus tokens.
* The system retrieves the top $k$ chunks and extracts their exact file paths and character indices.
* The extracted metadata is then passed to the generation pipeline to form the exact context window for the LLM.

## Performance analysis
System retrieval accuracy is benchmarked using a custom **Recall@k** evaluation script.
* **IoU (Intersection over Union) equivalent:** Instead of requiring exact index matches, the evaluation calculates the geometric overlap between the retrieved chunk's character range and the ground truth's character range.
* **Match criteria:** A retrieval is counted as a success if the overlap exceeds a strict 5% threshold (IoU > 0.05).
* The evaluation iterates up to $k$, providing a granular view of how recall improves as the number of retrieved documents increases.

## Design decisions
* **Dynamic Hardware Allocation:** To handle massive contexts without crashing, the system checks context length before LLM inference. If the context exceeds 9,000 characters, it automatically shifts from CUDA to CPU.
* **Strict Data Validation:** Adopting Pydantic ensures that corrupted or misformatted JSON ground-truth files fail fast, rather than causing silent errors during evaluation.
* **Character-Level Tracking:** Instead of tracking overlaps by arbitrary chunk IDs, the system calculates exact string indices. This decouples the evaluation logic from the specific chunking parameters used during ingestion.

## Challenges faced
* **Pydantic Validation Exceptions:** Handling nested JSON validations required refining the `StudentSearchResults` and `RagDataGround` schemas to cleanly decouple evaluation parameters (like $k$) from raw data schemas.
* **Double-Chunking Code:** Initially, processing code files through generic text splitters after code splitters resulted in broken logic. This was solved by applying the `PythonCodeTextSplitter` independently with strict length constraints.
* **Calculating Overlap for Recall:** Determining if the model found the "right" answer required implementing a custom mathematical intersection logic for character indices to fairly evaluate partial chunk overlaps.

## Example usage
To evaluate the provided codebase on a custom dataset:
```bash
# 1. Build the BM25 index with 2000-character chunks
python main.py index 2000 ./processed_index

# 2. Run the retrieval test over a list of questions
python main.py search_dataset ./test_questions.json 5 ./results student_retrieval.json

# 3. Generate answers using Qwen
python main.py answer_dataset ./results/student_retrieval.json ./results final_answers.json

# 4. Measure retrieval accuracy against ground truth
python main.py evaluate ./results/student_retrieval.json ./test_questions.json 5
```

## Resources
* **BM25s Documentation:** For fast, memory-efficient sparse retrieval implementation.
* **LangChain Text Splitters:** Classic reference for semantic document segmentation.
* **Hugging Face Transformers:** Standard documentation for loading and managing causal language models.
* **AI Assistance:** Artificial Intelligence was utilized during the development of this project primarily as a debugging and conceptual tutor. Specifically, AI helped to:
  * Clarify the mathematical logic behind standardizing **Recall@k** and **IoU (Intersection over Union)** for text boundaries.
  * Debug and architect strict schema validation using **Pydantic** to trace `ValidationError` exceptions.
  * Optimize the file parsing and chunking loops to prevent "empty slice" indexing errors.