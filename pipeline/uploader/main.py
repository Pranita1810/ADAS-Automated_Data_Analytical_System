import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from uploader.uploader import upload_data


def run():
    return upload_data()


if __name__ == "__main__":
    run()
