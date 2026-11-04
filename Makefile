PY ?= python
CFG ?= config.yaml

.PHONY: install test smoke pipeline demo api

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

smoke:            ## whole pipeline on synthetic faces, a few minutes on a laptop
	$(PY) -m shadesense make-demo --config configs/smoke.yaml
	$(PY) -m shadesense pipeline  --config configs/smoke.yaml

pipeline:
	$(PY) -m shadesense pipeline --config $(CFG)

demo:
	streamlit run app/streamlit_app.py -- --config $(CFG)

api:
	SHADESENSE_CONFIG=$(CFG) uvicorn app.api:app --reload
