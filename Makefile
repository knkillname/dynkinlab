PIPENV_VENV_IN_PROJECT := 1
PIPENV_VERBOSITY := -1
export PIPENV_VENV_IN_PROJECT PIPENV_VERBOSITY

env: .venv

.venv: Pipfile
	pipenv install --dev
	touch .venv

.PHONY: format
format: .venv
	pipenv run isort .
	pipenv run black .

test: .venv
	pipenv run python -m unittest discover -s tests -p "test_*.py" -v

.PHONY: clean
clean:
	git clean -Xdf