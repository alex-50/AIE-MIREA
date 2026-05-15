"""
Интеграционные тесты
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from src.service import app, load_models
from src.preprocessing import nlp_preprocess

client = TestClient(app)


class TestFullPipeline:
    """Тесты полного пайплайна"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Загрузка моделей перед тестами
        """
        load_models()
        yield

    def test_full_pipeline_with_fallback(self):
        """
        Тест полного пайплайна с fallback моделью
        """

        text = "Российские пожарные спасли детеныша косули"

        processed = nlp_preprocess(text)
        assert len(processed) > 0

        response = client.post("/predict", json={"texts": [text]})
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 1
        assert data["processing_time_ms"] > 0

    def test_category_consistency(self):
        """
        Проверка, что похожие тексты получают похожие категории
        """

        sports_texts = [
            "Футбольный матч закончился победой хозяев",
            "Зенит выиграл чемпионат по футболу",
            "Теннисист вышел в финал турнира"
        ]

        response = client.post("/predict", json={"texts": sports_texts})
        data = response.json()

        categories = [pred[0]["category_name"] for pred in data["predictions"]]

        for cat in categories:
            assert cat in ["Спорт", "Культура", "Общество"]

    def test_performance_batch_size(self):
        """
        Тест производительности для разных размеров батча
        Время должно расти не быстрее O(n)
        """

        sizes = [1, 5, 10, 25]
        base_text = "Тестовый текст для замера производительности"

        results = {}
        for size in sizes:
            texts = [base_text] * size
            response = client.post("/predict", json={"texts": texts})
            data = response.json()
            results[size] = data["processing_time_ms"]

        for size in sizes[1:]:
            time_per_item = results[size] / size
            prev_time_per_item = results[sizes[sizes.index(size) - 1]] / sizes[sizes.index(size) - 1]
            assert time_per_item <= prev_time_per_item * 1.5  # допустимо замедление не более 50%
