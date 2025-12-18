# Описание

Это проект по распознаванию животных с использованием CNN ; модель была обучена классифицировать изображения 10 видов животных

## Использованный датасета

- https://www.kaggle.com/alessiocorrado99/animals10

## Текущая реализация

### Подготовка данных (`prepare_dataset.ipynb`)
- Скачивание датасета, предобработка изображений (resize до 224x224, нормализация), разделение на train/val/test (70/15/15), сохранение в data/processed

### Обучение модели (`train_model.ipynb`)
- CNN с 3 сверточными слоями (16→32→64 каналов)
- параметры обучения: Adam (lr=0.0007), CrossEntropyLoss, 7 эпох

### Тестирование модели (`test_model.ipynb`)
- Загрузка лучшей модели
- Оценка на тестовых данных с помощью accuracy

## Структура проекта

ml-dev
├───data
│   ├───processed # обработанные фотки, загружены из prepare_dataset
│   └───raw-img # 10 классов животных
├───models
│       best_model.pt
│       best_model_acc_55.4.pt
│       best_model_acc_59.3.pt
│       final_model.pt
│
└───notebooks
    │   notebook_exp.ipynb
    │   prepare_dataset.ipynb - подготовка данных
    │   test_model.ipynb - тестирование
    │   train_model.ipynb - обучение 


## Результаты
- Accuracy: 61.26%

## Использовала:
- Python, PyTorch, Torchvision, Pandas, NumPy, Matplotlib

## TBD: backend, frontend