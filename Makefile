# Makefile

.PHONY: all install run search answer evaluate debug \
	lint lint-strict clean fclean re \
	index search_dataset moulinette answer_dataset \
	fastapi pipe

UV      = uv
PYTHON  = .venv/bin/python


all: install

install:
	$(UV) sync

run:
	$(UV) run python -m src

search:
	$(UV) run python -m src search "database search" 3

answer:
	$(UV) run python -m src answer "What activation formats does the fused batched MoE layer return in vLLM?" 1

debug:
	$(UV) run python -m pdb src

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

DOCS	= dataset_docs_public.json
CODE	= dataset_code_public.json
DATA	= $(DOCS) # DOCS or CODE
FILE	= $(DATA)
FINAL	= final_elaborate.json
K		= 9 # number of chunks, max 10
OUTUN	= data/output/search_results/UnansweredQuestions

index:
	$(UV) run python -m src index \
	--max_chunk_size 2000 \
	--save_directory data/processed

search_dataset:
	$(UV) run python -m src search_dataset \
	--dataset_path data/datasets/UnansweredQuestions/$(DATA) \
	--k $(K) \
	--save_directory $(OUTUN)

moulinette:
	./moulinette evaluate_student_search_results \
	$(OUTUN)/$(FILE) \
	data/datasets/AnsweredQuestions/$(DATA) \
	--k $(K) --max_context_length 2000

answer_dataset:
	$(UV) run python -m src answer_dataset \
	--student_search_results_path $(OUTUN)/$(FILE) \
	--save_directory data/output/search_results_and_answer/UnansweredQuestions \
	--save_file $(FINAL)

evaluate:
	$(UV) run python -m src evaluate \
	--student_search_results_path $(OUTUN)/$(FILE) \
	--dataset_path data/datasets/AnsweredQuestions/$(DATA) \
	--k $(K)

fastapi:
	$(UV) run fastapi dev --entrypoint src.evaluate:app

fastapic:
	$(UV) run fastapi dev --entrypoint src.commands:app


clean:
	rm -rf data/processed
	rm -rf data/output

fclean:
	rm -rf .venv
	rm -rf data/processed
	rm -rf data/output

re: clean install

lint:
	-$(UV) run flake8 src
	$(UV) run mypy --explicit-package-bases src

lint-strict:
	-$(UV) run flake8 src
	$(UV) run mypy --strict --explicit-package-bases src

pipe:
	@echo "\033[\n\n32mindex\033[0m"
	$(UV) run python -m src index \
	--max_chunk_size 2000 \
	--save_directory data/processed
	@echo "\033[32msearch_dataset\033[0m"
	$(UV) run python -m src search_dataset \
	--dataset_path data/datasets/UnansweredQuestions/$(DATA) \
	--k $(K) \
	--save_directory $(OUTUN)
	@echo "\033[32mmoulinette\033[0m"
	./moulinette evaluate_student_search_results \
	$(OUTUN)/$(FILE) \
	data/datasets/AnsweredQuestions/$(DATA) \
	--k $(K) --max_context_length 2000
	@echo "\033[32mevaluate\033[0m"
	$(UV) run python -m src evaluate \
	--student_search_results_path $(OUTUN)/$(FILE) \
	--dataset_path data/datasets/AnsweredQuestions/$(DATA) \
	--k $(K)
	@echo "\033[32manswer_dataset\033[0m"
	$(UV) run python -m src answer_dataset \
	--student_search_results_path $(OUTUN)/$(FILE) \
	--save_directory data/output/search_results_and_answer/UnansweredQuestions \
	--save_file $(FINAL)

exam:
	./exams/scripts/exam_retrieval.sh \
	--student-path ./ \
	--moulinette-path ./moulinette

