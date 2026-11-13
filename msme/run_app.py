"""Start the demo app:  python run_app.py"""
import os, sys
from streamlit.web import cli
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)
    sys.argv = ["app", "run", os.path.join(here, "app.py")]
    sys.exit(cli.main())
