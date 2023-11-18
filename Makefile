poetry-install:
	poetry install --sync

poetry-up:
	poetry up --latest

pre-commit-install:
	pre-commit install

pre-commit-up:
	pre-commit autoupdate

pre-commit-run:
	pre-commit run -a

lint:
	poetry run poe lint

tests:
	poetry run pytest
