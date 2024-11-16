from io import StringIO

import pandas as pd
import pytest

from ml.model import SentimentPrediction, batch_inference, load_model, process_csv


@pytest.fixture(scope="function")
def model():
    # Load the model once for each test function
    return load_model()


def test_batch_inference(model):
    texts = ["очень плохо", "очень хорошо", "по-разному"]
    predictions = batch_inference(texts, batch_size=2)

    # Проверка, что длина результатов соответствует входным данным
    assert len(predictions) == len(texts)

    # Проверка типа результатов
    for pred in predictions:
        assert isinstance(pred, SentimentPrediction)

    # Дополнительная проверка для первой строки
    assert predictions[0].label in {"negative", "positive", "neutral"}
    assert 0.0 <= predictions[0].score <= 1.0


def test_process_csv(tmp_path):
    # Пример данных для тестирования
    csv_data = """id,text
                    1,очень плохо
                    2,очень хорошо
                    3,по-разному
                    """
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    # Сохраняем тестовые данные в CSV
    with open(input_file, "w") as f:
        f.write(csv_data)

    # Запуск функции обработки CSV
    process_csv(
        input_csv=input_file, output_csv=output_file, text_column="text", batch_size=2
    )

    # Проверка, что файл был создан
    assert output_file.exists()

    # Проверка содержимого выходного CSV
    df_output = pd.read_csv(output_file)
    assert "label" in df_output.columns
    assert "score" in df_output.columns

    # Проверка содержимого строк
    for _, row in df_output.iterrows():
        assert row["label"] in {"negative", "positive", "neutral"}
        assert 0.0 <= row["score"] <= 1.0


def test_empty_batch():
    # Проверка обработки пустого списка текстов
    predictions = batch_inference([], batch_size=2)
    assert predictions == []


def test_invalid_column_csv(tmp_path):
    # Пример данных для тестирования с отсутствующим столбцом
    csv_data = """id,content
                    1,очень плохо
                    2,очень хорошо
                    3,по-разному
                    """
    input_file = tmp_path / "input_invalid.csv"
    output_file = tmp_path / "output_invalid.csv"

    # Сохраняем тестовые данные в CSV
    with open(input_file, "w") as f:
        f.write(csv_data)

    # Проверка, что возникает ошибка для отсутствующего столбца
    with pytest.raises(ValueError, match="Column 'text' not found in CSV file."):
        process_csv(
            input_csv=input_file,
            output_csv=output_file,
            text_column="text",
            batch_size=2,
        )