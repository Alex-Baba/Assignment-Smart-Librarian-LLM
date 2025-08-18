import argparse, subprocess, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cli", action="store_true", help="Run CLI recommender")
    p.add_argument("--ui", action="store_true", help="Run Streamlit UI")
    args = p.parse_args()

    if args.cli:
        from chatbot.cli import main as cli_main
        cli_main()
    elif args.ui:
        cmd = [sys.executable, "-m", "streamlit", "run", "ui/app_streamlit.py"]
        subprocess.run(cmd, check=False)
    else:
        print("Use --cli or --ui")

if __name__ == "__main__":
    main()