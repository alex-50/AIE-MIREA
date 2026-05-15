"""
Фикстуры для тестов
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from src.service import app


@pytest.fixture
def client():
    """
    Клиент для тестирования API
    """
    return TestClient(app)


@pytest.fixture
def sample_texts():
    """
    Примеры текстов для тестов
    """

    return {
        "sport": "Футбольный клуб Зенит выиграл чемпионат России",
        "science": "Ученые открыли новый метод лечения рака",
        "economy": "Центробанк снизил ключевую ставку",
        "politics": "Госдума приняла новый закон",
        "culture": "В Эрмитаже открылась выставка картин"
    }


@pytest.fixture
def sample_texts_batch():
    """
    Пакет примеров для batch тестов
    """

    return [
        "Сборная России победила в футбольном матче",
        "Ученые обнаружили новую экзопланету",
        "Нефть подорожала до 85 долларов",
        "В Москве прошел фестиваль искусств"
    ]
