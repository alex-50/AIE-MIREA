"""
Вспомогательные утилиты проекта.
"""

from pathlib import Path
import os
import zipfile

import gdown


def get_project_root() -> Path:
    """
    Возвращает абсолютный путь к корню проекта (project/).
    Работает из любого файла внутри project/: ноутбуков, src/, tests/.
    """
    return Path(__file__).resolve().parent.parent


def _download_from_gdrive(file_id: str, output_path: Path) -> None:
    """
    Универсальная функция для скачивания файлов с Google Drive.

    Args:
        file_id: ID файла на Google Drive
        output_path: Путь для сохранения файла
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = gdown.download(url, str(output_path), quiet=False)
    if output is None:
        raise RuntimeError(f"Failed to download from {url}")


def download_preprocessed_dataset() -> None:
    """
    Подгрузка готового очищенного датасета для обучения
    """
    DATASET_PATH = get_project_root() / "data" / "preprocessed" / "clean_data.parquet"
    FILE_ID = "16ktFLwOplE17TMsCBcPctELWeiI-41rW"

    _download_from_gdrive(FILE_ID, DATASET_PATH)


def download_finetune_model() -> None:
    """
    Скачивает и распаковывает веса переобученной модели
    """
    PROJECT_ROOT = get_project_root()
    ARCHIVE_PATH = PROJECT_ROOT / "artifacts" / "rubert_fine-tune" / "rubert_tiny2_finetune.zip"
    EXTRACT_TO = PROJECT_ROOT / "artifacts" / "rubert_fine-tune"
    FILE_ID = "1rvWhWsYjJ6l3X9wUSx34Go9qmSXwRlYO"

    _download_from_gdrive(FILE_ID, ARCHIVE_PATH)

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_TO)

    os.remove(ARCHIVE_PATH)
