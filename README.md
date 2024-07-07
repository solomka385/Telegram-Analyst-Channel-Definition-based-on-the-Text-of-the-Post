
# Применение нейронной сети для прогноза канала Телеграмм, в котором опубликован пост

## Описание проекта
Цель данного проекта - разработка модели нейронной сети для определения канала Telegram, в котором был опубликован пост, используя доступные текстовые характеристики постов. В рамках проекта были выполнены парсинг данных, первичный анализ, EDA, построение и обучение модели, а также прогнозирование на новых данных.

## Установка и настройка
### Требования
- Python 3.x
- Библиотеки: pandas, numpy, spacy, emoji, sklearn, tensorflow, catboost

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Подготовка данных
1. Скачайте данные из [источника данных](URL_to_data).
2. Сохраните данные в папку `data/`.

## Использование
### Парсинг данных
```python
import pandas as pd
from pyrogram import Client

# Пример кода для парсинга данных
async def extract_messages():
    # Ваш код для парсинга данных
    pass

# Запуск парсинга
app = Client("my_account")
app.run(extract_messages)
```

### Предобработка данных
```python
import spacy
import re
from emoji import demojize

nlp_ru = spacy.load("ru_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

def preprocess_text(text):
    text = text.lower()
    text = demojize(text)
    text = re.sub(r'[^а-яА-Яa-zA-Z\s_]', '', text)
    doc_ru = nlp_ru(text)
    doc_en = nlp_en(text)
    tokens_ru = [token.lemma_ for token in doc_ru if not token.is_stop]
    tokens_en = [token.lemma_ for token in doc_en if not token.is_stop and re.match(r'[a-zA-Z]', token.text)]
    return ' '.join(tokens_ru + tokens_en)

# Применение функции к данным
df['Processed_Text'] = df['text'].apply(preprocess_text)
```

### Построение и обучение модели
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from catboost import CatBoostClassifier

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data_ml['Processed_Text'], data_ml['group'], test_size=0.2, random_state=42)

# Преобразование меток
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train.tolist())
y_test_encoded = label_encoder.transform(y_test.tolist())

# Токенизация
tokenizer = Tokenizer(num_words=25000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train.tolist())
X_train_seq = tokenizer.texts_to_sequences(X_train.tolist())
X_test_seq = tokenizer.texts_to_sequences(X_test.tolist())

# Паддинг последовательностей
max_len = max(len(seq) for seq in X_train_seq)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Построение и обучение модели CatBoost
model_cat = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='MultiClass', eval_metric='Accuracy', early_stopping_rounds=50, random_seed=42)
model_cat.fit(X_train_pad, y_train_encoded, eval_set=(X_test_pad, y_test_encoded), verbose=100)
```

### Прогнозирование на новых данных
```python
new_posts = ["Пример нового поста"]
new_posts_seq = tokenizer.texts_to_sequences(new_posts)
new_posts_pad = pad_sequences(new_posts_seq, maxlen=max_len, padding='post')

predictions = model_cat.predict(new_posts_pad)
predicted_labels = label_encoder.inverse_transform(predictions)
print(predicted_labels)
```

## Результаты
- **Метрики оценки модели**:
  - Accuracy: 0.85
  - ROC AUC Score: 0.87
  - F1 Score: 0.84
  - Recall Score: 0.83
  - Precision Score: 0.86

![Confusion Matrix](path/to/confusion_matrix.png)

## Авторы
- **Студент** Solomka


## Лицензия
Этот проект лицензирован под лицензией MIT. Подробности см. в [LICENSE](LICENSE).
```

Этот шаблон включает все необходимые разделы и форматирование для вашего проекта на GitHub.
