# Makefile

.PHONY: all install run test lint clean

UV      = uv
PYTHON  = .venv/bin/python


all: install

install:
	$(UV) sync

run:
	$(UV) run python main.py

debug:
	$(UV) run python -m pdb main.py config.txt

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
	rm -rf .venv

re: clean install

lint:
	-$(UV) run flake8 *.py
	$(UV) run mypy *.py

lint-strict:
	-$(UV) run flake8 *.py
	$(UV) run mypy --strict *.py

