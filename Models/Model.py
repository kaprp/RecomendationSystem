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

from keras.models import Sequential, load_model
from keras.layers import Dense
import csv
import pickle


from positivepy import *
from negative import *

# file_pos = "pos.csv"
# file_neg = "neg.csv"
#
# positive_texts = []
# negative_texts = []
#
# with open(file_pos, 'r') as file:
#     csv_reader = csv.reader(file)
#
#     # Проход по каждой строке CSV файла и добавление ее в список данных
#     for row in csv_reader:
#         positive_texts.append(row)
#
# print(positive_texts)
#
# with open(file_neg, 'r') as file:
#     csv_reader = csv.reader(file)
#
#     # Проход по каждой строке CSV файла и добавление ее в список данных
#     for row in csv_reader:
#         negative_texts.append(row)



file = "kartaslovsent.csv"

positive_texts = []
negative_texts = []
words = []

with open(file, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file, delimiter=";")
    i = 0
    # Проход по каждой строке CSV файла и добавление ее в список данных
    for row in csv_reader:
        if i != 0 :
            words.append((row[0],row[2]))
        i += 1

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
        vectors = np.array([np.mean([model.wv[word] for word in text.split() if word in model.wv] or [np.zeros(vector_size)], axis=0) for text in processed_texts])

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

def predict_sentiment(text):
    # Векторизация нового текста
    vectorized_text = text_vectorization(text)
    # Предсказание тональности с помощью модели
    prediction = model.predict(vectorized_text)
    # Определение тональности на основе предсказания
    sentiment = "Positive" if prediction <= 0.5 else "Negative"
    return sentiment, prediction[0]

# X_positive = text_vectorization(positive_texts)
# X_negative = text_vectorization(negative_texts)
#
# # Создание меток для положительных и отрицательных текстов
# y_positive = np.ones(len(positive_texts))
# y_negative = np.zeros(len(negative_texts))
#
# # Объединение текстов и меток
# X = np.concatenate((X_positive, X_negative), axis=0)
# y = np.concatenate((y_positive, y_negative), axis=0)

X = text_vectorization([i[0] for i in words])
Y = [float(i[1]) for i in words]
print(X)
# Перемешивание данных
indices = np.arange(len(X))
np.random.shuffle(indices)
X = [X[i] for i in indices]
Y = [Y[i] for i in indices]
#
#
# Разделение на обучающую и тестовую выборки (например, 80% обучающих данных и 20% тестовых данных)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = Y[:split_index], Y[split_index:]

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)



# Создание модели
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))



# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')


# Введи текст
text = ["Плохо"]
X_new = text_vectorization(text)

# Предсказание тональности
predictions = model.predict(X_new)


model.save("analytics4.keras")

with open("word_dictionary.pkl", "wb") as f:
    pickle.dump(model.wv, f)

# Вывод предсказаний
for text, prediction in zip(text, predictions):
    sentiment = "Positive" if prediction <= 0.5 else "Negative"
    print(f'Text: {text} - Sentiment: {sentiment} (Probability: {prediction[0]})')

# loaded_model = load_model("analytics4.keras")
#
#     # Введенные тексты для анализа тональности
# texts = ["Хорошо"]
#
#     # Векторизация введенных текстов
# X_new = text_vectorization(texts)
#
#     # Предсказание тональности
# predictions = loaded_model.predict(X_new)
#
#     # Вывод предсказаний
# for text, prediction in zip(texts, predictions):
#         sentiment = "Positive" if prediction <= 0.5 else "Negative"
#         print(f'Text: {text} - Sentiment: {sentiment} (Probability: {prediction[0]})')

model = load_model("analytics4.keras")

# Загрузка словаря слов
with open("word_dictionary.pkl", "rb") as f:
    word_dictionary = pickle.load(f)

text = "Ваш новый текст"
sentiment, confidence = predict_sentiment(text)
print(f"Тональность текста: {sentiment}, Уверенность: {confidence}")