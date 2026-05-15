"""
Тесты для API сервиса
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from src.service import app

client = TestClient(app)


class TestHealthEndpoint:
    """Тесты для /health"""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_type" in data
        assert "device" in data
        assert "fallback_loaded" in data

    def test_health_status_healthy(self):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"


class TestPredictEndpoint:
    """Тесты для /predict"""

    def test_predict_single_text(self):
        response = client.post(
            "/predict",
            json={"texts": ["Российские пожарные спасли детеныша косули"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "model_used" in data
        assert "processing_time_ms" in data
        assert len(data["predictions"]) == 1
        assert len(data["predictions"][0]) == 3  # top-3

    def test_predict_multiple_texts(self):
        texts = [
            "Сборная России победила в футбольном матче",
            "Ученые открыли новый метод лечения рака",
            "Центробанк снизил ключевую ставку"
        ]
        response = client.post("/predict", json={"texts": texts})
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3

    def test_predict_max_batch(self):
        """Проверка максимального количества текстов (100)"""
        texts = [f"Текст номер {i}" for i in range(100)]
        response = client.post("/predict", json={"texts": texts})
        assert response.status_code == 200

    def test_predict_empty_list(self):
        response = client.post("/predict", json={"texts": []})
        assert response.status_code == 422

    def test_predict_missing_field(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422  # ошибка валидации

    def test_predict_prediction_structure(self):
        response = client.post(
            "/predict",
            json={"texts": ["Футбольный матч завершился победой хозяев"]}
        )
        data = response.json()
        prediction = data["predictions"][0][0]
        assert "category_id" in prediction
        assert "category_name" in prediction
        assert "confidence" in prediction
        assert isinstance(prediction["category_id"], int)
        assert isinstance(prediction["confidence"], float)
        assert 0 <= prediction["confidence"] <= 1

    def test_predict_model_used(self):
        response = client.post(
            "/predict",
            json={"texts": ["Тестовый текст для определения модели"]}
        )
        data = response.json()
        assert data["model_used"] in ["bert", "fallback"]


class TestPredictSingleEndpoint:
    """Тесты для /predict_single"""

    def test_predict_single_endpoint(self):
        response = client.post(
            "/predict_single",
            params={"text": "Российские пожарные спасли детеныша косули"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "predictions" in data
        assert "model_used" in data
        assert "processing_time_ms" in data

    def test_predict_single_empty_text(self):
        response = client.post("/predict_single", params={"text": ""})
        assert response.status_code in [200, 400]

    def test_predict_single_missing_param(self):
        response = client.post("/predict_single")
        assert response.status_code == 422


class TestResponseConsistency:
    """Тесты согласованности ответов"""

    def test_same_input_same_output(self):
        """
        Проверка детерминированности
        """

        text = "Футбольный клуб Зенит выиграл чемпионат"

        response1 = client.post("/predict", json={"texts": [text]})
        response2 = client.post("/predict", json={"texts": [text]})

        pred1 = response1.json()["predictions"][0][0]["category_id"]
        pred2 = response2.json()["predictions"][0][0]["category_id"]
        assert pred1 == pred2

    def test_batch_vs_single_consistency(self):
        """
        Проверка, что batch и single дают одинаковые результаты
        """

        text = "Ученые открыли новый метод лечения рака"

        batch_response = client.post("/predict", json={"texts": [text]})
        single_response = client.post("/predict_single", params={"text": text})

        batch_pred = batch_response.json()["predictions"][0][0]["category_id"]
        single_pred = single_response.json()["predictions"][0]["category_id"]
        assert batch_pred == single_pred
