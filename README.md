# SmartHire - Resume Matcher

This project is a classical machine-learning based resume matcher and recommender built with Streamlit and scikit-learn.

Features
- TF-IDF vectorization of resumes
- Logistic Regression classifier for resume category prediction
- Model evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
- Resume-job similarity matching using cosine similarity
- Top-5 job recommendations with match score and inferred company/location
- Skill gap report comparing resume skills with job requirements

Quick start
1. Create a Python virtual environment and activate it.

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate # macOS / Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model (this will create `models/` with trained artifacts):

```bash
python train_model.py
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

Environment Variables
- `DATA_URL`: remote URL to the resume dataset CSV (optional)
- `DATA_PATH`: local path to the resume dataset CSV (optional)
- `RESUME_DATASET_PATH`: alternate local path to the resume dataset CSV (optional)
- `JOBS_PATH`: local path to the job listings CSV (optional, default `jobs.csv`)
- `MODEL_PATH`: path to an external serialized model if loading externally (optional)
- `LOG_LEVEL`: logging level, e.g. `INFO`
- `STREAMLIT_SERVER_HEADLESS`: set to `true` for deployment

Render Deployment
- Set the start command to:

```bash
streamlit run app.py --server.port $PORT --server.headless true
```

- Ensure the `jobs.csv` and dataset file are available in Render either by uploading them or by using `DATA_URL`/`JOBS_PATH`.
- If files are unavailable, the app now uses built-in sample fallback data for deployment.

Notes
- The repository expects `ResumeDataset.csv.csv` and `jobs.csv` in the project root for training and job recommendations, but they are ignored in git and should be kept locally or provided via env variables.
- Do not use generative AI or external LLMs — this project uses only scikit-learn.

Author: yeswanth
