"""
Тесты для модуля preprocessing
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
from src.preprocessing import nlp_preprocess, preprocess_batch


class TestNLPPreprocess:
    """Тесты для nlp_preprocess"""

    def test_basic_cleaning(self):
        """
        Проверка базовой очистки: регистр, пунктуация
        """
        text = "Российские! пожарные, спасли детеныша? косули."
        result = nlp_preprocess(text)

        assert result.islower()
        assert "!" not in result
        assert "," not in result
        assert "?" not in result
        assert "." not in result

    def test_stop_words_removal(self):
        """
        Проверка удаления стоп-слов
        """
        text = "Я иду в школу сегодня утром"
        result = nlp_preprocess(text)
        # Стоп-слова должны быть удалены
        assert "я" not in result.split()
        assert "и" not in result.split()
        assert "в" not in result.split()
        assert "сегодня" not in result.split()

    def test_lemmatization(self):
        """
        Проверка лемматизации
        """
        text = "Российские пожарные спасали детенышей"
        result = nlp_preprocess(text)
        # Леммы: российский, пожарный, спасать, детёныш
        assert "российский" in result or "российские" not in result
        assert "пожарный" in result or "пожарные" not in result

    def test_short_words_removal(self):
        """
        Проверка удаления слов короче 3 букв
        """
        text = "кот в доме"
        result = nlp_preprocess(text)
        assert "в" not in result.split()
        assert "кот" in result.split()
        assert "дом" in result.split()

    def test_max_words_limit(self):
        """
        Проверка ограничения на количество слов
        """
        text = " ".join([f"слово{i}" for i in range(1000)])
        result = nlp_preprocess(text, max_words=100)
        assert len(result.split()) <= 100

    def test_empty_text(self):
        """
        Проверка обработки пустого текста
        """
        result = nlp_preprocess("")
        assert result == ""

    def test_text_with_numbers(self):
        """
        Проверка обработки текста с числами
        """
        text = "В 2024 году цены выросли на 15 процентов"
        result = nlp_preprocess(text)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "цена" in result or "цены" in result

    def test_news_noise_removal(self):
        """
        Проверка удаления новостного шума
        """
        text = "По данным РИА Новости, источник сообщил, что..."
        result = nlp_preprocess(text)
        assert "риа" not in result
        assert "новости" not in result
        assert "сообщил" not in result
        assert "источник" not in result


class TestPreprocessBatch:
    """Тесты для preprocess_batch"""

    def test_batch_processing(self):
        """
        Проверка пакетной обработки
        """
        texts = [
            "Российские пожарные спасли детеныша",
            "Ученые открыли новый метод лечения",
            "Центробанк сохранил ключевую ставку"
        ]
        results = preprocess_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
        assert all(len(r) > 0 for r in results)

    def test_batch_with_empty(self):
        """
        Проверка пакетной обработки с пустыми текстами
        """
        texts = ["", "Нормальный текст", ""]
        results = preprocess_batch(texts)

        assert len(results) == 3
        assert results[0] == ""
        assert len(results[1]) > 0
        assert results[2] == ""
        assert results[2] == ""
