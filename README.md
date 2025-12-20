# Описание

Это проект по распознаванию животных с использованием свёрточной нейронной сети (CNN) и FastAPI для бэка. Модель обучена классифицировать изображения 10 видов животных.

## Использованный датасет

- https://www.kaggle.com/alessiocorrado99/animals10



## Структура проекта: ML-часть 

```
ml-dev/
├── data/              # Данные
│   ├── processed/     # Предобработанные
│   └── raw-img/       # Исходные
├── models/            # Веса моделей
└── notebooks/         # Ноутбуки
    ├── prepare_dataset.ipynb   # Подготовка данных
    ├── train_model.ipynb       # Обучение CNN
    └── test_model.ipynb        # Оценка модели
```
## Структура проекта: веб
```
animal-proj/
├── main.py
├── handlers.py        # FastAPI бэкенд
├── index.html         # Frontend
└── requirements.txt   # Зависимости Python

## Реализация

### ML-часть:

 --- Подготовка данных: скачивание датасета, предобработка изображений (resize до 224x224, нормализация), разделение на train/val/test (70/15/15), сохранение в data/processed

 --- Обучение модели : CNN с 3 сверточными слоями (16→32→64 каналов), параметры обучения: Adam (lr=0.0007), CrossEntropyLoss, 7 эпох

 --- Тестирование модели: загрузка лучшей модели и её последующая оценка на тестовых данных при помощи accuracy

### Backend:
Framework: FastAPI + Uvicorn

#### API ендпоинты:

 --- GET / → Главная страница с HTML формой

 --- POST /predict → приём изображения и возврат предикшена

#### Frontend: его писала не я , взяла из открытого источника, но получилось все по красоте :)

## Результаты
- Accuracy: 63.42%

## Использовала:
- Python, PyTorch, Torchvision, Pandas, NumPy, Matplotlib
- FastAPI, uvicorn