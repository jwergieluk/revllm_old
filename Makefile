export PYTHONPATH = .

black-format:
	black -t py311 -S -l 100 revllm *.py

black: black-format

isort-format:
	isort -m VERTICAL_HANGING_INDENT --py auto --tc revllm *.py

lint: black-format flake

check:
	pytest -v test

pretty: isort-format black-format check

