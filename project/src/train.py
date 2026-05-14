"""
Обучение моделей классификации статей
Обучаются:
1. SVM baseline (fallback model)
2. RuBERT fine-tuned model (primary model)
"""
import os
import sys
import json
import random
import shutil
from pathlib import Path

import torch
import joblib
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import Dataset, DatasetDict
from transformers import (
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoModelForSequenceClassification
)

sys.path.append(str(Path(__file__).parent))
from utils import get_project_root, download_preprocessed_dataset

RANDOM_SEED = 42
MODEL_NAME = "cointegrated/rubert-tiny2"
BATCH_SIZE = 32
MAX_LENGTH = 512
NUM_CLASSES = 11
EPOCHS = 5

PROJECT_ROOT = get_project_root()
PREPROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "preprocessed" / "clean_data.parquet"
BASELINE_ARTIFACTS_PATH = PROJECT_ROOT / "artifacts" / "baselines"
RUBERT_ARTIFACTS_PATH = PROJECT_ROOT / "artifacts" / "rubert_fine-tune"

os.makedirs(BASELINE_ARTIFACTS_PATH, exist_ok=True)
os.makedirs(RUBERT_ARTIFACTS_PATH, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Фиксирует случайные сиды для воспроизводимости результатов.

    Args:
        seed: Значение сида для всех генераторов случайных чисел
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data() -> pd.DataFrame:
    """
    Загружает предобработанный датасет.

    Returns:
        DataFrame с колонками 'news', 'labels', 'news_tfidf_ready' и др.
    """
    try:
        df = pd.read_parquet(PREPROCESSED_DATASET_PATH)
        print(f"Dataset loaded: {len(df)} samples, {df['labels'].nunique()} classes")
        return df
    except FileNotFoundError:
        print("Preprocessed dataset not found. Downloading...")
        download_preprocessed_dataset()
        df = pd.read_parquet(PREPROCESSED_DATASET_PATH)
        print(f"Dataset loaded: {len(df)} samples")
        return df


def train_fallback(df: pd.DataFrame) -> Pipeline:
    """
    Обучает SVM модель как fallback (резервный вариант).

    Args:
        df: датафрейм с колонками 'news_tfidf_ready' и 'labels'

    Returns:
        Обученный sklearn Pipeline
    """
    print("\n" + "=" * 60)
    print("TRAINING FALLBACK MODEL (LinearSVC)")
    print("=" * 60)

    X = df["news_tfidf_ready"]
    y = df["labels"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Training samples: {len(X_train)}")

    pipeline = Pipeline([
        ("vec", TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            max_features=70000,
            min_df=3,
            max_df=0.8
        )),
        ("svm", LinearSVC(C=0.3, random_state=RANDOM_SEED))
    ])

    print("Training SVM pipeline...")
    pipeline.fit(X_train, y_train)

    model_path = BASELINE_ARTIFACTS_PATH / "svm_best_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Fallback model saved to: {model_path}")

    params = {
        "C": 0.3,
        "max_features": 70000,
        "min_df": 3,
        "max_df": 0.8,
        "ngram_range": [1, 2],
        "random_state": RANDOM_SEED
    }
    with open(BASELINE_ARTIFACTS_PATH / "svm_best_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    return pipeline


def prepare_rubert_datasets(df: pd.DataFrame) -> DatasetDict:
    """
    Подготавливает датасеты для дообучения rubert-tiny-2.

    Args:
        df: DataFrame с колонками 'news' и 'labels'

    Returns:
        DatasetDict с разбивкой train/validation/test
    """
    print("\n" + "=" * 60)
    print("PREPARING RUBERT-TINY-2 DATASETS")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        df["news"],
        df["labels"],
        test_size=0.2,
        stratify=df["labels"],
        random_state=RANDOM_SEED
    )

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["label"],
        random_state=RANDOM_SEED
    )

    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True))
    })

    return dataset


def tokenize_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    """
    Токенизирует датасет для rubert-tiny-2.

    Args:
        dataset: DatasetDict с новостями
        tokenizer: токенизатор HuggingFace

    Returns:
        Токенизированный DatasetDict
    """

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    return tokenized_datasets


def train_rubert(df: pd.DataFrame) -> None:
    """
    Выполняет дообучение (fine-tuning) модели rubert-tiny-2.

    Args:
        df: DataFrame с колонками 'news' и 'labels'
    """
    print("\n" + "=" * 60)
    print("TRAINING RUBERT-TINY-2 MODEL")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = prepare_rubert_datasets(df)

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES
    ).to(device)

    tokenized_datasets = tokenize_dataset(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=RUBERT_ARTIFACTS_PATH / "output",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        seed=RANDOM_SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("\nStarting training...")
    trainer.train()

    model_path = RUBERT_ARTIFACTS_PATH / "rubert_tiny2_finetune"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"rubert-tiny-2 model saved to: {model_path}")

    checkpoints_dir = RUBERT_ARTIFACTS_PATH / "output"
    if checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir)
        print(f"Removed checkpoints directory")


def main():
    """
    Запускает обучение fallback модели и дообучение rubert-tiny-2
    """

    print("\n" + "=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)

    set_seed()
    df = load_data()

    train_fallback(df)
    train_rubert(df)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
