"""Top-level wrapper to maintain compatibility with existing commands.

This file calls into app.streamlit_app.main() so running `streamlit run app.py`
continues to work while the application code is organized under `app/`.
"""

from app.streamlit_app import main


if __name__ == "__main__":
    main()