.SILENT: install run clean fclean lint lint-strict

.PHONY: install run clean fclean lint lint-strict



install:
	uv sync

run:
	uv run python -m src $(ARGS)

clean:
	rm -rf */*/*/__pycache__
	rm -rf */*/__pycache__
	rm -rf */__pycache__
	rm -rf __pycache__
	rm -rf */*/*/.mypy_cache
	rm -rf */*/.mypy_cache
	rm -rf */.mypy_cache
	rm -rf .mypy_cache
	rm -rf */*/*/.pytest_cache
	rm -rf */*/.pytest_cache
	rm -rf */.pytest_cache
	rm -rf .pytest_cache

fclean: clean
	rm -rf .venv
	rm -rf uv.lock
	uv cache clean

debug:
	uv run python -m pdb -m src $(ARGS)

lint:
	uv run python -m flake8 src --exclude=.venv && printf '\033[1;32mFlake8 all good !\033[0m\n'; uv run python -m mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	uv run python -m flake8 src --exclude=.venv && printf '\033[1;32mFlake8 all good !\033[0m\n'; uv run python -m mypy src --strict
