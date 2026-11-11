PY ?= python
CFG ?= config.yaml

.PHONY: install test smoke pipeline cards report demo gradio api space

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

smoke:            ## whole pipeline on synthetic faces, a few minutes on a laptop
	$(PY) -m shadesense make-demo --config configs/smoke.yaml
	$(PY) -m shadesense pipeline  --config configs/smoke.yaml

pipeline:
	$(PY) -m shadesense pipeline --config $(CFG)

cards:            ## preview the white card found in each reference photo
	$(PY) -m shadesense check-cards --config $(CFG)

report:
	$(PY) -m shadesense report --config $(CFG)

gradio:
	SHADESENSE_CONFIG=$(CFG) $(PY) app/gradio_app.py

space:            ## folder ready to push to a Hugging Face Space
	$(PY) scripts/build_space.py --config $(CFG)

demo:
	streamlit run app/streamlit_app.py -- --config $(CFG)

api:
	SHADESENSE_CONFIG=$(CFG) uvicorn app.api:app --reload
