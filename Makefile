.PHONY: test gate serve load-test drift clean

gate:
	python3 run_quality_gate.py

test:
	python3 -m unittest discover -s tests -v

serve:
	python3 -m aegisqa.cli serve --port 8080

load-test:
	python3 -m aegisqa.cli load-test --requests 100 --concurrency 15

drift:
	python3 -m aegisqa.cli drift
