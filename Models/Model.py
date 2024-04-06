#pip install --upgrade

import pymorphy2
import nltk
from nltk.corpus import stopwords

import gensim.models
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from positive import positive_texts
from negative import negative_texts

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('russian'))


def preprocess_russian_text(text):
    # Инициализация анализатора pymorphy2
    morph = pymorphy2.MorphAnalyzer()

    text = text.lower()

    # Токенизация текста и удаление пунктуации
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]

    # Удаление стоп-слов

    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Лемматизация токенов
    lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens]

    return ' '.join(lemmatized_tokens)


def text_vectorization(texts, model_type='word2vec', vector_size=1000):
    """
    Векторизует текстовые данные с использованием Gensim.

    Параметры:
    texts (list): Список текстов для векторизации.
    model_type (str): Тип модели для векторизации. Возможные значения: 'word2vec' (по умолчанию) или 'tfidf'.
    vector_size (int): Размер вектора. Применим только для 'word2vec'.

    Возвращает:
    numpy.array: Матрица векторов текста.
    """

    if model_type == 'word2vec':
        # Предварительная обработка текстов
        # processed_texts = [simple_preprocess(text) for text in texts]
        processed_texts = [preprocess_russian_text(text) for text in texts]
        # Обучение модели Word2Vec
        model = Word2Vec(sentences=processed_texts, vector_size=vector_size, min_count=1)
        # Получение векторов для каждого текста
        vectors = np.array([np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(vector_size)], axis=0) for words in processed_texts])

    elif model_type == 'tfidf':
        # Предварительная обработка текстов и создание словаря
        dictionary = Dictionary([simple_preprocess(text) for text in texts])
        corpus = [dictionary.doc2bow(simple_preprocess(text)) for text in texts]
        # Обучение модели TF-IDF
        tfidf_model = TfidfModel(corpus)
        # Преобразование корпуса в матрицу TF-IDF
        tfidf_vectors = tfidf_model[corpus]
        # Преобразование TF-IDF векторов в numpy массив
        vectors = np.vstack([gensim.matutils.sparse2full(vector, len(dictionary)) for vector in tfidf_vectors])

    else:
        raise ValueError("Неподдерживаемый тип модели. Допустимые значения: 'word2vec' или 'tfidf'.")

    return vectors


X_positive = text_vectorization(positive_texts)
X_negative = text_vectorization(negative_texts)

# Создание меток для положительных и отрицательных текстов
y_positive = np.ones(len(positive_texts))
y_negative = np.zeros(len(negative_texts))

# Объединение текстов и меток
X = np.concatenate((X_positive, X_negative), axis=0)
y = np.concatenate((y_positive, y_negative), axis=0)

# Перемешивание данных
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Разделение на обучающую и тестовую выборки (например, 80% обучающих данных и 20% тестовых данных)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Создание модели
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')
# Введи текст
text = ["Все было просто отвратительно"]
X_new = text_vectorization(text)

# Предсказание тональности
predictions = model.predict(X_new)

# Вывод предсказаний
for text, prediction in zip(text, predictions):
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f'Text: {text} - Sentiment: {sentiment} (Probability: {prediction[0]})')