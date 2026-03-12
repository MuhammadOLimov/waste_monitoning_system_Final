# main.py
from app.config import Settings
from app.runner import run_app


def main():
    s = Settings()
    run_app(s)


if __name__ == "__main__":
    main()