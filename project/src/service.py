"""
FastAPI сервис для классификации новостей
"""

import time
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent))

from preprocessing import preprocess_batch
from utils import get_project_root

############ Конфигурация ############

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Климат", "Конфликты", "Культура", "Экономика", "Глянец", "Здоровье",
    "Политика", "Наука", "Общество", "Спорт", "Путешествия"
]
PROJECT_ROOT = get_project_root()
BASELINE_ARTIFACTS_PATH = PROJECT_ROOT / "artifacts" / "baselines"
RUBERT_ARTIFACTS_PATH = PROJECT_ROOT / "artifacts" / "rubert_fine-tune"

############ Глобальные переменные для моделей ############
fallback_pipeline = None
bert_model = None
bert_tokenizer = None
model_type = "fallback"


############ Модели данных ############
class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="Список текстов новостей")


class PredictionResult(BaseModel):
    category_id: int
    category_name: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: List[List[PredictionResult]]
    model_used: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_type: str
    device: str
    fallback_loaded: bool


############ Инициализация ############
app = FastAPI(
    title="News Classifier API",
    description="Классификация новостей с использованием fine-tuned rubert-tiny-2 или fallback TF-IDF + SVM",
    version="1.0.0"
)


def load_models():
    """
    Загрузка моделей при старте сервиса
    """
    global fallback_pipeline, bert_model, bert_tokenizer, model_type

    print("=" * 50)
    print("Loading models...")
    print("=" * 50)

    fallback_path = BASELINE_ARTIFACTS_PATH / "svm_best_model.joblib"
    if fallback_path.exists():
        fallback_pipeline = joblib.load(fallback_path)
        print(f"Fallback model loaded from {fallback_path}")
    else:
        print(f"Fallback model not found at {fallback_path}")

    bert_path = RUBERT_ARTIFACTS_PATH / "rubert_tiny2_finetune"
    if bert_path.exists():
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            bert_tokenizer = AutoTokenizer.from_pretrained(str(bert_path))
            bert_model = AutoModelForSequenceClassification.from_pretrained(str(bert_path))
            bert_model.to(DEVICE)
            bert_model.eval()
            model_type = "bert"
            print(f"BERT model loaded from {bert_path} on {DEVICE}")
        except Exception as e:
            print(f"Failed to load BERT model: {e}")
            print("Using fallback model only")
    else:
        print(f"BERT model not found at {bert_path}")
        print("Using fallback model only")

    print(f"Active model: {model_type.upper()}")
    print("=" * 50)


############ Предсказания ############
def predict_fallback(texts: List[str], top_k: int = 3) -> List[List[PredictionResult]]:
    """
    Предсказание с помощью fallback модели (TF-IDF + SVM).
    Использует decision_function + softmax для получения confidence.
    """

    processed_texts = preprocess_batch(texts, max_words=500)

    distances = fallback_pipeline.decision_function(processed_texts)

    if distances.ndim == 1:
        distances = distances.reshape(-1, 1)

    results = []
    for dist in distances:
        exp_dist = np.exp(dist - np.max(dist))
        probs = exp_dist / exp_dist.sum()

        top_indices = np.argsort(probs)[::-1][:top_k]
        preds = []
        for idx in top_indices:
            preds.append(PredictionResult(
                category_id=int(idx),
                category_name=CLASS_NAMES[idx],
                confidence=float(probs[idx])
            ))
        results.append(preds)

    return results


def predict_bert(texts: List[str], top_k: int = 3) -> List[List[PredictionResult]]:
    """
    Предсказание с помощью fine-tuned BERT.
    """

    results = []
    for text in texts:
        inputs = bert_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

        preds = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            preds.append(PredictionResult(
                category_id=int(idx),
                category_name=CLASS_NAMES[idx],
                confidence=float(prob)
            ))
        results.append(preds)

    return results


@app.on_event("startup")
async def startup_event():
    """Загрузка моделей при старте"""
    load_models()


############ Эндпоинты ############
@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Проверка работоспособности сервиса
    """
    return HealthResponse(
        status="healthy",
        model_type=model_type,
        device=str(DEVICE) if model_type == "bert" else "cpu",
        fallback_loaded=fallback_pipeline is not None
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Классификация списка текстов новостей.

    Возвращает для каждого текста top-3 предсказания с уверенностью.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    start_time = time.time()

    if model_type == "bert" and bert_model is not None:
        predictions = predict_bert(request.texts, top_k=3)
        model_used = "bert"
    elif fallback_pipeline is not None:
        predictions = predict_fallback(request.texts, top_k=3)
        model_used = "fallback"
    else:
        raise HTTPException(status_code=503, detail="No model available")

    processing_time = (time.time() - start_time) * 1000

    return PredictResponse(
        predictions=predictions,
        model_used=model_used,
        processing_time_ms=round(processing_time, 2)
    )


@app.post("/predict_single")
async def predict_single(text: str):
    """
    Упрощённый эндпоинт для одного текста.
    """
    response = await predict(PredictRequest(texts=[text]))
    return {
        "text": text,
        "predictions": response.predictions[0],
        "model_used": response.model_used,
        "processing_time_ms": response.processing_time_ms
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
