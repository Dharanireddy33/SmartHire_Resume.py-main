import json
import os
import re

import joblib
import pandas as pd
import PyPDF2
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
try:
    import numpy as np
except Exception:
    np = None


def sanitize_for_json(obj):
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)


def load_or_train_model_artifacts():
    """Load the serialized model files, or train them if they do not exist."""
    required_files = [
        resolve_path(os.path.join("models", "model.pkl")),
        resolve_path(os.path.join("models", "vectorizer.pkl")),
        resolve_path(os.path.join("models", "label_encoder.pkl")),
        resolve_path(os.path.join("models", "evaluation_metrics.json")),
    ]

    if all(os.path.exists(path) for path in required_files):
        model = joblib.load(required_files[0])
        vectorizer = joblib.load(required_files[1])
        label_encoder = joblib.load(required_files[2])
        with open(required_files[3], "r", encoding="utf-8") as handle:
            evaluation_metrics = json.load(handle)
        return model, vectorizer, label_encoder, evaluation_metrics

    # If artifacts are missing, attempt to train. If training fails for any reason
    # (network download, file not present, etc), catch the exception and build
    # a small fallback model from the sample dataset so the UI does not crash.
    try:
        from train_model import train_and_evaluate_model

        train_and_evaluate_model()
        return load_or_train_model_artifacts()
    except Exception as exc:  # fallback to in-memory sample model
        print(f"Warning: training failed during deployment: {exc}")
        try:
            from train_model import load_sample_dataset

            sample_df = load_sample_dataset()
            # Train a minimal TF-IDF + LogisticRegression on the sample data
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import LabelEncoder
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

            resumes = sample_df["Resume"].fillna("").astype(str)
            categories = sample_df["Category"].fillna("").astype(str)

            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            X = vectorizer.fit_transform(resumes)

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(categories)

            # Only stratify when every class has at least 2 samples to avoid
            # sklearn ValueError for very small sample sets.
            try:
                y_counts = pd.Series(y).value_counts()
                stratify_arg = y if (y_counts >= 2).all() else None
            except Exception:
                stratify_arg = None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_arg
            )

            model = LogisticRegression(max_iter=5000, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # When using small sample data, some classes may be absent from the
            # test split. Build reports only for labels present to avoid errors.
            labels_present = sorted(set(y_test))
            try:
                target_names_present = label_encoder.inverse_transform(labels_present)
            except Exception:
                target_names_present = [str(l) for l in labels_present]

            evaluation_metrics = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, average="weighted")), 4),
                "recall": round(float(recall_score(y_test, y_pred, average="weighted")), 4),
                "f1_score": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
                "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels_present).tolist(),
                "classification_report": classification_report(y_test, y_pred, labels=labels_present, target_names=target_names_present),
                "labels": list(target_names_present),
                "label_indices": [int(x) for x in labels_present],
            }

            # Persist artifacts so subsequent loads use them
            models_dir = resolve_path("models")
            os.makedirs(models_dir, exist_ok=True)
            joblib.dump(model, os.path.join(models_dir, "model.pkl"))
            joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
            joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))
            with open(os.path.join(models_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as fh:
                json.dump(sanitize_for_json(evaluation_metrics), fh, indent=2)

            return model, vectorizer, label_encoder, evaluation_metrics
        except Exception as ee:
            # As a last resort, re-raise the original exception to surface useful logs
            print(f"Fallback training also failed: {ee}")
            raise


def get_jobs_csv_path() -> str:
    env_path = os.getenv("JOBS_PATH") or os.getenv("JOBS_URL") or "jobs.csv"
    if env_path.startswith(("http://", "https://")):
        raise ValueError(
            "Remote JOBS_URL is not supported by the current job loader. "
            "Please provide a local JOBS_PATH instead."
        )
    return resolve_path(env_path)


def load_sample_jobs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Job": "Data Scientist",
                "Description": "Lead data science projects using Python, pandas, scikit-learn, machine learning, and statistical modeling.",
            },
            {
                "Job": "Software Engineer",
                "Description": "Build backend systems and REST APIs with Java, Python, Docker, and cloud-native deployment practices.",
            },
            {
                "Job": "Web Developer",
                "Description": "Develop responsive web applications using HTML, CSS, JavaScript, React, and modern UX patterns.",
            },
            {
                "Job": "Business Analyst",
                "Description": "Analyze business processes, prepare SQL reports, and deliver actionable insights to stakeholders.",
            },
            {
                "Job": "Machine Learning Engineer",
                "Description": "Deploy machine learning models, manage feature pipelines, and optimize model performance in production.",
            },
        ]
    )


def extract_pdf_text(uploaded_file):
    """Extract text from an uploaded PDF resume."""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    pages_text = []

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


def extract_skills(text):
    """Extract a simple set of technical skills from resume/job text."""
    if not text:
        return []

    text_lower = text.lower()
    skill_library = [
        "python",
        "sql",
        "java",
        "javascript",
        "html",
        "css",
        "pandas",
        "numpy",
        "scikit-learn",
        "machine learning",
        "deep learning",
        "data science",
        "data analysis",
        "tableau",
        "power bi",
        "excel",
        "git",
        "docker",
        "flask",
        "django",
        "aws",
        "cloud",
        "spark",
        "hadoop",
        "statistics",
        "nlp",
        "natural language processing",
        "opencv",
        "matlab",
        "r",
        "c++",
        "linux",
        "api",
        "rest",
    ]

    discovered_skills = []
    for skill in skill_library:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            discovered_skills.append(skill)

    return sorted(set(discovered_skills))


def infer_company(job_title, description):
    """Create a simple company display value when the dataset does not provide one."""
    text = f"{job_title} {description}".lower()
    company_map = {
        "google": "Google",
        "microsoft": "Microsoft",
        "amazon": "Amazon",
        "meta": "Meta",
        "apple": "Apple",
        "netflix": "Netflix",
        "ibm": "IBM",
        "accenture": "Accenture",
        "infosys": "Infosys",
        "tcs": "TCS",
        "wipro": "Wipro",
        "ey": "EY",
    }

    for keyword, company in company_map.items():
        if keyword in text:
            return company

    return "Company not listed"


def infer_location(description):
    """Create a simple location display value when the dataset does not provide one."""
    text = description.lower()
    if "remote" in text:
        return "Remote"
    if "hyderabad" in text:
        return "Hyderabad"
    if "bangalore" in text or "bengaluru" in text:
        return "Bangalore"
    if "mumbai" in text:
        return "Mumbai"
    if "delhi" in text:
        return "Delhi"
    if "pune" in text:
        return "Pune"
    return "Location not listed"


def build_job_recommendations(resume_text, jobs_df, vectorizer):
    """Score each job against the uploaded resume using cosine similarity."""
    if resume_text.strip() == "":
        return []

    resume_vector = vectorizer.transform([resume_text])
    recommendation_rows = []

    for _, row in jobs_df.iterrows():
        job_text = str(row.get("Description", ""))
        job_vector = vectorizer.transform([job_text])
        similarity = cosine_similarity(resume_vector, job_vector)[0][0]
        match_score = round(float(similarity * 100), 2)

        recommendation_rows.append(
            {
                "job_title": row.get("Job", "Unknown Role"),
                "company": infer_company(str(row.get("Job", "")), job_text),
                "location": infer_location(job_text),
                "match_score": match_score,
                "required_skills": extract_skills(job_text),
                "description": job_text,
            }
        )

    recommendation_rows.sort(key=lambda item: item["match_score"], reverse=True)
    return recommendation_rows[:5]


def build_skill_gap_report(resume_text, selected_job_description):
    """Compare resume skills with a selected job and return a simple gap report."""
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(selected_job_description))

    present_skills = sorted(resume_skills & job_skills)
    missing_skills = sorted(job_skills - resume_skills)
    suggested_skills = missing_skills[:5]

    if missing_skills:
        recommendation = (
            f"Your resume already covers {len(present_skills)} key skills for this role. "
            f"To improve your match, focus on learning {', '.join(suggested_skills)}."
        )
    else:
        recommendation = "Your profile already aligns strongly with this opportunity."

    return {
        "present_skills": present_skills,
        "missing_skills": missing_skills,
        "suggested_skills": suggested_skills,
        "recommendation": recommendation,
    }


def main():
    """Run the Streamlit app UI."""
    st.set_page_config(page_title="SmartHire", page_icon="📄", layout="wide")

    model, vectorizer, label_encoder, evaluation_metrics = load_or_train_model_artifacts()

    if os.path.exists("logo.png"):
        logo = Image.open("logo.png")
        st.image(logo, width=140)

    st.title("📄 SmartHire - Resume Matcher")
    st.markdown("Upload a resume, classify it into a professional category, and compare it with job opportunities using classical machine learning.")

    st.markdown("---")

    # Input Section
    with st.container():
        st.subheader("1. Resume and Job Details")
        col_left, col_right = st.columns([1.1, 0.9])

        with col_left:
            job_description = st.text_area(
                "Enter Job Description",
                height=150,
                placeholder="Paste the job description here...",
            )

        with col_right:
            uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
            resume_text = st.text_area(
                "Or Paste Resume Text",
                height=190,
                placeholder="Paste your resume text here if you do not want to upload a PDF...",
            )

    if uploaded_file is not None:
        extracted_text = extract_pdf_text(uploaded_file)
        if extracted_text.strip():
            resume_text = extracted_text
            st.success("✅ Resume uploaded successfully and text extracted.")

    # Resume Classification
    if resume_text and resume_text.strip():
        resume_vector = vectorizer.transform([resume_text])
        predicted_category = label_encoder.inverse_transform(model.predict(resume_vector))[0]
        probability_scores = model.predict_proba(resume_vector)[0]
        confidence = round(float(max(probability_scores) * 100), 2)

        st.markdown("---")
        st.subheader("2. Resume Category Classification")

        col_a, col_b = st.columns([1.3, 0.7])
        with col_a:
            st.success(f"Predicted Category: {predicted_category}")
            st.caption(f"Model confidence: {confidence:.2f}%")
        with col_b:
            st.metric("Model Accuracy", f"{evaluation_metrics.get('accuracy', 0) * 100:.2f}%")

        with st.expander("View model evaluation results"):
            metrics_display = pd.DataFrame(
                {
                    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                    "Value": [
                        f"{evaluation_metrics.get('accuracy', 0) * 100:.2f}%",
                        f"{evaluation_metrics.get('precision', 0) * 100:.2f}%",
                        f"{evaluation_metrics.get('recall', 0) * 100:.2f}%",
                        f"{evaluation_metrics.get('f1_score', 0) * 100:.2f}%",
                    ],
                }
            )
            st.dataframe(metrics_display, use_container_width=True, hide_index=True)

            confusion_matrix_data = evaluation_metrics.get("confusion_matrix", [])
            metric_label_names = evaluation_metrics.get("labels", [])
            metric_label_indices = evaluation_metrics.get("label_indices", None)

            # Determine the appropriate label names for the confusion matrix.
            labels_to_use = None
            try:
                # If metric provides explicit indices (numeric labels), use them to map
                # to human-readable names when possible.
                if metric_label_indices is not None and hasattr(label_encoder, "inverse_transform"):
                    if len(metric_label_indices) == len(confusion_matrix_data):
                        labels_to_use = list(label_encoder.inverse_transform(metric_label_indices))

                # If still not determined, and the stored names length matches matrix size, use them.
                if labels_to_use is None and metric_label_names and len(metric_label_names) == len(confusion_matrix_data):
                    labels_to_use = metric_label_names

                # As a last resort, if label_encoder has classes and counts align, attempt to use a prefix of them.
                if labels_to_use is None and hasattr(label_encoder, "classes_"):
                    if len(label_encoder.classes_) >= len(confusion_matrix_data):
                        labels_to_use = list(label_encoder.classes_)[: len(confusion_matrix_data)]

                # If still None, generate numeric labels matching matrix dimensions.
                if labels_to_use is None:
                    labels_to_use = [str(i) for i in range(len(confusion_matrix_data))]
            except Exception:
                labels_to_use = [str(i) for i in range(len(confusion_matrix_data))]

            if confusion_matrix_data and labels_to_use:
                try:
                    confusion_df = pd.DataFrame(confusion_matrix_data, index=labels_to_use, columns=labels_to_use)
                    st.markdown("**Confusion Matrix**")
                    st.dataframe(confusion_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render confusion matrix: {e}")

            st.markdown("**Classification Report**")
            st.text_area("", evaluation_metrics.get("classification_report", ""), height=220)

    # Match Resume Button
    if st.button("Match Resume", use_container_width=True):
        if job_description.strip() == "" or resume_text.strip() == "":
            st.warning("Please enter both a job description and a resume before running the match.")
        else:
            vectors = vectorizer.transform([job_description, resume_text])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            score = round(float(similarity * 100), 2)

            st.markdown("---")
            st.subheader("3. Resume Match Score")
            st.progress(min(int(score), 100))
            st.metric("Resume Match Score", f"{score:.2f}%")

            if score >= 80:
                st.success("⭐⭐⭐⭐⭐ Excellent match for this role.")
            elif score >= 60:
                st.info("👍 Strong match. The profile fits the role well.")
            elif score >= 40:
                st.warning("⚠ Moderate match. Some skills may need improvement.")
            else:
                st.error("❌ Weak match. Consider strengthening your profile for this job.")

            job_csv_path = get_jobs_csv_path()
            if not os.path.exists(job_csv_path):
                st.info(
                    "Job dataset not found. Using built-in sample jobs. "
                    "Set JOBS_PATH to a real jobs CSV for production use."
                )
                jobs = load_sample_jobs()
            else:
                jobs = pd.read_csv(job_csv_path)
            top_jobs = build_job_recommendations(resume_text, jobs, vectorizer)

            st.markdown("---")
            st.subheader("4. Top 5 Matching Jobs")
            top_jobs_df = pd.DataFrame(top_jobs)
            if not top_jobs_df.empty:
                display_df = top_jobs_df[["job_title", "company", "match_score", "required_skills", "location"]].copy()
                display_df.rename(
                    columns={
                        "job_title": "Job Title",
                        "company": "Company",
                        "match_score": "Match Score (%)",
                        "required_skills": "Required Skills",
                        "location": "Location",
                    },
                    inplace=True,
                )
                display_df["Required Skills"] = display_df["Required Skills"].apply(lambda skills: ", ".join(skills) if skills else "No skills listed")
                display_df["Match Score (%)"] = display_df["Match Score (%)"].round(2)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                selected_job = st.selectbox("Select a job for skill-gap analysis", options=top_jobs_df["job_title"].tolist())
                selected_job_row = top_jobs_df[top_jobs_df["job_title"] == selected_job].iloc[0]
                skill_gap_report = build_skill_gap_report(resume_text, selected_job_row["description"])

                st.markdown("---")
                st.subheader("5. Skill Gap Report")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Skills already present", len(skill_gap_report["present_skills"]))
                with col2:
                    st.metric("Missing skills", len(skill_gap_report["missing_skills"]))
                with col3:
                    st.metric("Suggested skills to learn", len(skill_gap_report["suggested_skills"]))

                st.markdown("**Skills already present**")
                if skill_gap_report["present_skills"]:
                    st.success(", ".join(skill_gap_report["present_skills"]))
                else:
                    st.warning("No overlapping skills detected from the uploaded resume.")

                st.markdown("**Missing skills**")
                if skill_gap_report["missing_skills"]:
                    st.warning(", ".join(skill_gap_report["missing_skills"]))
                else:
                    st.success("No missing skills detected for this role.")

                st.markdown("**Suggested skills to learn**")
                if skill_gap_report["suggested_skills"]:
                    st.info(", ".join(skill_gap_report["suggested_skills"]))
                else:
                    st.success("No additional skill suggestions needed.")

                st.markdown("**Recommendation**")
                st.write(skill_gap_report["recommendation"])
            else:
                st.info("No matching jobs were found for the provided resume.")

    st.markdown("---")
    st.caption("Developed as a final-year machine learning project using classical scikit-learn techniques.")


if __name__ == "__main__":
    main()
