"""
Вспомогательные утилиты проекта.
"""

from pathlib import Path
import gdown


def get_project_root() -> Path:
    """
    Возвращает абсолютный путь к корню проекта (project/).
    Работает из любого файла внутри project/: ноутбуков, src/, tests/.
    """
    return Path(__file__).resolve().parent.parent


def download_preprocessed_dataset() -> None:
    DATASET_PATH = get_project_root() / "data" / "preprocessed"
    FILE_ID = "16ktFLwOplE17TMsCBcPctELWeiI-41rW"
    URL = f"https://drive.google.com/uc?id={FILE_ID}"
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(URL, str(DATASET_PATH), quiet=False)
