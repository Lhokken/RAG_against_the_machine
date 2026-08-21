# Makefile

.PHONY: all install run index search test lint lint-strict clean fclean

UV      = uv
PYTHON  = .venv/bin/python


all: install

install:
	$(UV) sync

run:
	$(UV) run python -m src

index:
	$(UV) run python -m src index --max_chunk_size 1500

search:
	$(UV) run python -m src search "database search" 3

search_dataset:
	$(UV) run python -m src search_dataset \
	--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
	--k 1 \
	--save_directory data/output/search_results/UnansweredQuestions

answer:
	$(UV) run python -m src answer "What activation formats does the fused batched MoE layer return in vLLM?" 1

answer_dataset:
	$(UV) run python -m src answer_dataset "./data/output/search_result/corpus.json" "/data/output/search_result_and_answer"

evaluate:
	$(UV) run python -m src answer_dataset "./data/datasets/AnsweredQuestions" "./data/raw/vllm-0.10.1"

debug:
	$(UV) run python -m pdb src

moul:
	./moulinette evaluate_student_search_results \
	data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
	data/datasets/AnsweredQuestions/dataset_docs_public.json \
	--k 10 --max_context_length 2000

# Run pdb in a shell
# 	Command		Short	What it does
# 	next		n		Execute next line (don't step into calls)
# 	step		s		Step into a function call
# 	continue	c		Run until next breakpoint
# 	quit		q		Exit debugger
# 	list		l		Show surrounding source code
# 	where		w		Print call stack
# 	up / down	u / d	Move up/down the call stack
# 	return		r		Run until current function returns
	
clean:
	rm -rf data/processed/Index_bm25s

fclean:
	rm -rf .venv
	rm -rf data/processed/Index_bm25s

re: clean install

lint:
	-$(UV) run flake8 src
	$(UV) run mypy --explicit-package-bases src

lint-strict:
	-$(UV) run flake8 src
	$(UV) run mypy --strict --explicit-package-bases src

