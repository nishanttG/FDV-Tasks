.PHONY: install preprocess train eval streamlit fastapi

# Install dependencies
install:
	pip install -r requirements.txt

# Run preprocessing
preprocess:
	python -m src.main preprocess

# Train the model
train:
	python -m src.main train

# Evaluate the model
eval:
	python -m src.main evaluate

# Launch Streamlit app
streamlit:
	streamlit run src/streamlit_app/app.py

# Launch FastAPI server
api:
	uvicorn src.models.serve_fastapi:app --reload --host 127.0.0.1 --port 8000