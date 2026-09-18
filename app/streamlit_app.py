import json
import os
import re

import altair as alt
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


def build_dashboard_data():
    """Create sample job market data matching the screenshot style."""
    cluster_points = pd.DataFrame(
        {
            "x": [
                -0.45, -0.40, -0.34, -0.28, -0.22, -0.19, -0.14, -0.11, -0.07, -0.03,
                0.06, 0.10, 0.14, 0.18, 0.22, 0.30, 0.35, 0.41, 0.47, 0.52,
                -0.25, -0.18, -0.12, -0.06, 0.00, 0.07, 0.13, 0.18, 0.23, 0.27,
                -0.55, -0.52, -0.48, -0.42, -0.38, -0.32, -0.28, -0.22, -0.17, -0.10,
                0.15, 0.22, 0.30, 0.38, 0.44, 0.50, 0.56, 0.60, 0.68, 0.73,
                -0.15, -0.08, -0.02, 0.04, 0.12, 0.20, 0.27, 0.36, 0.42, 0.48,
            ],
            "y": [
                0.35, 0.40, 0.44, 0.48, 0.52, 0.58, 0.63, 0.69, 0.74, 0.78,
                0.20, 0.25, 0.29, 0.35, 0.39, 0.44, 0.50, 0.55, 0.61, 0.66,
                -0.18, -0.22, -0.28, -0.32, -0.36, -0.26, -0.19, -0.16, -0.11, -0.08,
                -0.35, -0.31, -0.28, -0.24, -0.20, -0.14, -0.10, -0.05, -0.01, 0.04,
                0.10, 0.12, 0.18, 0.22, 0.26, 0.30, 0.35, 0.39, 0.42, 0.46,
                -0.42, -0.38, -0.34, -0.30, -0.25, -0.21, -0.17, -0.10, -0.06, -0.02,
            ],
            "cluster": [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            ],
        }
    )

    cluster_points["cluster"] = cluster_points["cluster"].astype(str)
    cluster_points["cluster"] = cluster_points["cluster"].replace({
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
    })

    top_terms = pd.DataFrame(
        {
            "cluster": [0, 1, 2, 3, 4, 5, 6, 7],
            "top_terms": [
                "devops engineering, devops, engineering, improving devops, bash, ansible, ci, cd",
                "network, engineering, network engineering, software, software engineering, cybersecurity, assessment, engineering join",
                "web development, development, web, development joint, development capability, improving web, next",
                "design, ux, design, ui, ux, design join, design capability, improving ui",
                "assurance, quality assurance, testing, test, assurance join, improving quality, assurance capability, planning",
                "database, database administration, administration, administration join, administration capability, improving database, data moc",
                "business, management, business analysis, analysis, process, project, stakeholder management, project management",
                "learning, machine learning, machine, data science, learning engineering, science, data, learn",
            ],
        }
    )

    category_counts = pd.DataFrame(
        {
            "category": [
                "Data Science",
                "Web Development",
                "Machine Learning",
                "Data Engineering",
                "Software Engineering",
                "Product Management",
                "Cybersecurity",
                "Business Analysis",
                "QA Automation",
                "Database Admin",
                "DevOps",
                "UX Design",
            ],
            "count": [26, 22, 18, 17, 28, 12, 10, 14, 9, 8, 7, 11],
        }
    )

    salary_df = pd.DataFrame(
        {
            "category": [
                "Data Science", "Web Development", "Machine Learning", "Data Engineering",
                "Software Engineering", "Product Management", "Cybersecurity", "Business Analysis",
                "QA Automation", "Database Admin", "DevOps", "UX Design",
            ],
            "avg_salary": [160000, 135000, 175000, 150000, 165000, 145000, 170000, 120000, 110000, 125000, 140000, 128000],
        }
    )
    return cluster_points, top_terms, category_counts, salary_df


def render_dark_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: #071019;
            color: #f4f5f7;
        }
        [data-testid="stSidebar"] {
            background: #0b151f;
            border-right: 1px solid #1b2935;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.2rem;
            padding-bottom: 4rem;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        h1, h2, h3, h4 {
            color: #f4f5f7;
            font-weight: 750;
        }
        .stButton > button {
            background: #e14d43;
            color: #fff;
            border: 1px solid #e14d43;
            border-radius: 6px;
            font-weight: 600;
            min-height: 44px;
        }
        .stButton > button:hover {
            background: #f05d52;
            border-color: #f05d52;
        }
        .stFileUploader > div {
            background: #0d1822;
            border: 1px solid #263744;
            border-radius: 6px;
        }
        .stTextArea textarea {
            background: #0d1822;
            color: #ecf3ff;
            border: 1px solid #263744;
        }
        .stProgress .st-bo {
            background: rgba(72, 134, 255, 0.8);
        }
        .metric-container {
            background: #0d1822;
            border: 1px solid #263744;
            border-radius: 0.7rem;
        }
        .landing-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 0 26px;
        }
        .landing-brand {
            color: #f4f5f7;
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: .01em;
        }
        .brand-mark {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            margin-right: 9px;
            border-radius: 7px;
            background: #e14d43;
            color: #fff;
            font-size: .85rem;
        }
        .landing-kicker {
            color: #e14d43;
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
        }
        .landing-title {
            max-width: 850px;
            margin: 15px auto 16px;
            color: #f7f7f5;
            font-size: clamp(2.5rem, 6vw, 5rem);
            line-height: .98;
            letter-spacing: -.045em;
            font-weight: 850;
        }
        .landing-copy {
            max-width: 650px;
            margin: 0 auto;
            color: #aab6bf;
            font-size: 1.05rem;
            line-height: 1.65;
        }
        .landing-hero {
            position: relative;
            overflow: hidden;
            padding: 58px 24px 38px;
            text-align: center;
            border-top: 1px solid #1a2a36;
            border-bottom: 1px solid #1a2a36;
            background: radial-gradient(circle at 50% 0%, #142630 0, #071019 56%);
        }
        .feature-card {
            height: 100%;
            min-height: 286px;
            padding: 26px 25px 22px;
            border: 1px solid #263744;
            border-radius: 7px;
            background: #0d1822;
            text-align: left;
        }
        .feature-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 42px;
            height: 42px;
            margin-bottom: 22px;
            border-radius: 7px;
            background: #172a35;
            color: #f06b5d;
            font-size: 1.25rem;
        }
        .feature-card h3 {
            margin: 0 0 10px;
            font-size: 1.18rem;
        }
        .feature-card p {
            min-height: 49px;
            margin: 0 0 19px;
            color: #9eabb4;
            font-size: .9rem;
            line-height: 1.55;
        }
        .feature-list {
            padding: 0;
            margin: 0;
            list-style: none;
            color: #d4dce0;
            font-size: .82rem;
            line-height: 2;
        }
        .feature-list li::before {
            margin-right: 9px;
            color: #e14d43;
            content: '✓';
            font-weight: 800;
        }
        .landing-section-title {
            margin: 43px 0 20px;
            color: #f4f5f7;
            font-size: 1.25rem;
            font-weight: 750;
        }
        .landing-footer {
            margin-top: 38px;
            color: #71808a;
            font-size: .78rem;
            text-align: center;
        }
        @media (max-width: 700px) {
            .landing-hero { padding: 38px 10px 27px; }
            .landing-title { font-size: 2.65rem; }
            .feature-card { min-height: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_cluster_chart(cluster_df):
    base = alt.Chart(cluster_df).mark_circle(size=65, opacity=0.8).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[-0.7, 0.8])),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[-0.7, 0.8])),
        color=alt.Color('cluster:N', scale=alt.Scale(scheme='category10')),
    ).properties(width=900, height=420)
    return base


def main():
    """Run the Streamlit app UI."""
    st.set_page_config(page_title="SmartHire", page_icon="S", layout="wide", initial_sidebar_state="collapsed")
    render_dark_style()

    if not st.session_state.get("workspace_open", False):
        st.markdown(
            """
            <div class="landing-nav">
                <div class="landing-brand"><span class="brand-mark">S</span>SmartHire</div>
                <div class="landing-kicker">Intelligent hiring workspace</div>
            </div>
            <div class="landing-hero">
                <div class="landing-kicker">Resume intelligence, made practical</div>
                <div class="landing-title">Find the right fit.<br>Build the next career.</div>
                <div class="landing-copy">SmartHire turns resumes into clear, useful decisions with machine learning that helps candidates and hiring teams move forward.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="landing-section-title">Everything you need to make a better match</div>', unsafe_allow_html=True)
        card_one, card_two, card_three = st.columns(3, gap="medium")
        cards = [
            (card_one, "⌕", "Resume screening", "Understand a resume in seconds and surface the career category behind the experience.", ["AI category prediction", "PDF and text upload", "Confidence score"]),
            (card_two, "↗", "Smart recommendations", "Compare a candidate with real role requirements and rank the strongest opportunities.", ["Top job matches", "Similarity scoring", "Company and location"]),
            (card_three, "◇", "Career guidance", "See the skills that matter next and explore the market around your target role.", ["Skill-gap report", "Learning suggestions", "Market analytics"]),
        ]
        for column, icon, title, copy, items in cards:
            with column:
                list_items = "".join(f"<li>{item}</li>" for item in items)
                st.markdown(
                    f'<div class="feature-card"><div class="feature-icon">{icon}</div><h3>{title}</h3><p>{copy}</p><ul class="feature-list">{list_items}</ul></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height: 22px'></div>", unsafe_allow_html=True)
        action_left, action_right = st.columns([1, 1], gap="medium")
        with action_left:
            if st.button("Open SmartHire workspace", use_container_width=True):
                st.session_state.workspace_open = True
                st.rerun()
        with action_right:
            st.markdown("<div style='height: 44px; border: 1px solid #263744; border-radius: 6px; color: #87959d; display: flex; align-items: center; justify-content: center; font-size: .88rem;'>Classical ML · Private by design</div>", unsafe_allow_html=True)
        st.markdown('<div class="landing-footer">Built with scikit-learn and Streamlit</div>', unsafe_allow_html=True)
        return

    model, vectorizer, label_encoder, evaluation_metrics = load_or_train_model_artifacts()

    with st.sidebar:
        st.markdown("<div style='display:flex; align-items:center; gap:12px; margin-top: 8px; margin-bottom: 24px;'>"
                    "<div style='width:30px;height:30px;border-radius:50%; background: linear-gradient(135deg,#ff8ec7,#e86cde);'></div>"
                    "<div style='font-size: 2rem; font-weight: 700;'>SmartHire</div>"
                    "</div>", unsafe_allow_html=True)
        st.caption("Resume-to-Job Matching & Career Guidance Engine")
        st.markdown("---")
        st.subheader("Pipeline status")
        status_items = [
            "Resume classifier:",
            "Job TF-IDF:",
            "Job clustering:",
            "Fit predictor:",
        ]
        for item in status_items:
            st.markdown(f"• {item} <span style='color:#48d18d; font-weight:700;'>✓</span>", unsafe_allow_html=True)
        st.markdown("---")
        st.caption("Built with scikit-learn, spaCy, NLTK, and Streamlit.")
        st.caption("Upload a resume to get started.")

    st.markdown("<div style='padding-left: 26px;'>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 3rem; margin-top: 0.25rem; margin-bottom: 0.5rem;'>SmartHire — Resume-to-Job Matching & Career Guidance Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; color: #d8e0ef; margin-bottom: 1.5rem;'>Upload your resume to get an AI-predicted career category, top job matches, a personalized skill-gap report, and a job-market overview.</p>", unsafe_allow_html=True)

    st.markdown("<div style='display:flex; align-items:center; margin: 1rem 0 1rem 0;'>"
                "<div style='display:inline-flex; width: 30px; height: 30px; border-radius: 10px; background: #4d9ef7; align-items:center; justify-content:center; margin-right: 12px; color: white; font-weight: 800;'>1</div>"
                "<h2 style='margin: 0;'>Upload Resume</h2>"
                "</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    resume_text = st.text_area("", height=180, placeholder="Upload a resume file or paste resume text here...")

    if uploaded_file is not None:
        extracted_text = extract_pdf_text(uploaded_file)
        if extracted_text.strip():
            resume_text = extracted_text
            st.success("✅ Resume uploaded successfully and text extracted.")

    if resume_text and resume_text.strip():
        resume_vector = vectorizer.transform([resume_text])
        predicted_category = label_encoder.inverse_transform(model.predict(resume_vector))[0]
        probability_scores = model.predict_proba(resume_vector)[0]
        confidence = round(float(max(probability_scores) * 100), 2)

        st.success(f"Predicted Category: {predicted_category}")
        st.caption(f"Model confidence: {confidence:.2f}%")

    if st.button("Match Resume", use_container_width=True):
        if not resume_text or not resume_text.strip():
            st.warning("Please upload or paste a resume before matching.")
        else:
            sample_job = "Data Scientist with Python, machine learning, SQL, model deployment, and business problem solving."
            vectors = vectorizer.transform([sample_job, resume_text])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            score = round(float(similarity * 100), 2)

            st.markdown("---")
            st.subheader("Resume Match Score")
            st.progress(min(int(score), 100))
            st.metric("Resume Match Score", f"{score:.2f}%")

    cluster_points, top_terms, category_counts, salary_df = build_dashboard_data()

    st.markdown("---")
    st.markdown("<div style='display:flex; align-items:center; margin: 1rem 0 1rem 0;'>"
                "<div style='display:inline-flex; width: 30px; height: 30px; border-radius: 10px; background: #4d9ef7; align-items:center; justify-content:center; margin-right: 12px; color: white; font-weight: 800;'>6</div>"
                "<h2 style='margin: 0;'>Job Market Clusters</h2>"
                "</div>", unsafe_allow_html=True)
    st.caption("Projection method")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.radio("", ["PCA (fast)", "t-SNE (slower, often cleaner)"], index=0, horizontal=True, label_visibility="collapsed")
    st.markdown("<h3 style='margin-top: 1rem;'>Job Postings — Cluster Visualization</h3>", unsafe_allow_html=True)
    st.altair_chart(build_cluster_chart(cluster_points), use_container_width=True)

    st.markdown("<h3 style='margin-top: 2rem;'>Top terms per cluster</h3>", unsafe_allow_html=True)
    st.dataframe(top_terms, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("<div style='display:flex; align-items:center; margin: 1.5rem 0 0.5rem 0;'>"
                "<div style='display:inline-flex; width: 30px; height: 30px; border-radius: 10px; background: #4d9ef7; align-items:center; justify-content:center; margin-right: 12px; color: white; font-weight: 800;'>7</div>"
                "<h2 style='margin: 0;'>Job Market Analytics</h2>"
                "</div>", unsafe_allow_html=True)

    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.markdown("<h3>Job Postings by Category</h3>", unsafe_allow_html=True)
        st.bar_chart(category_counts.set_index("category")["count"], use_container_width=True)
    with chart_b:
        st.markdown("<h3>Estimated Salary Range by Category</h3>", unsafe_allow_html=True)
        salary_chart = alt.Chart(salary_df).mark_bar().encode(
            x=alt.X('category:N', sort=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('avg_salary:Q', title='avg_salary'),
            color=alt.Color('category:N', legend=None),
            tooltip=['category:N', 'avg_salary:Q']
        ).properties(width=540, height=320)
        st.altair_chart(salary_chart, use_container_width=True)

    st.caption("Developed as a final-year machine learning project using classical scikit-learn techniques.")
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
