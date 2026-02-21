install:
		pip install -r requirements.txt
test:
		python -m pytest -vv test_app.py
format:
		black *.py 
lint:
		pylint --disable=R,C app.py
all: 	install format lint