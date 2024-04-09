import csv
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2
import gensim
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

file = "kartaslovsent.csv"

words = []

with open(file, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file, delimiter=";")
    i = 0
    # Проход по каждой строке CSV файла и добавление ее в список данных
    for row in csv_reader:
        if i != 0:
            words.append((row[0], row[2]))
        i += 1

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('russian'))


def preprocess_russian_text(text):
    # Инициализация анализатора pymorphy2
    morph = pymorphy2.MorphAnalyzer()

    text = text.lower()

    # Токенизация текста и удаление пунктуации
    tokens = word_tokenize(text)
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
    print(model_type)
    if model_type == 'word2vec':
        # Предварительная обработка текстов
        processed_texts = [preprocess_russian_text(text) for text in texts]
        # Обучение модели Word2Vec
        model = Word2Vec(sentences=processed_texts, vector_size=vector_size, min_count=1)
        model.save("1.bin")
        # Получение векторов для каждого текста
        vectors = np.array(
            [np.mean([model.wv[word] for word in text.split() if word in model.wv] or [np.zeros(vector_size)], axis=0)
             for text in processed_texts])

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
    elif model_type == 'word2vecsave':
        processed_texts = [preprocess_russian_text(text) for text in texts]
        model = Word2Vec.load("1.bin")
        vectors = np.array(
            [np.mean([model.wv[word] for word in text.split() if word in model.wv] or [np.zeros(vector_size)], axis=0)
             for text in processed_texts])
    else:
        raise ValueError("Неподдерживаемый тип модели. Допустимые значения: 'word2vec' или 'tfidf'.")

    return vectors
#
# #
# # Разделение данных на обучающий и тестовый наборы
# X = np.array([word[0] for word in words])
# y =  np.array([float(word[1]) for word in words])
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Векторизация текста с использованием модели Word2Vec
# X_train_vectors = text_vectorization(X_train, model_type='word2vec')
# X_test_vectors = text_vectorization(X_test, model_type='word2vec')
#
# # Создание модели нейронной сети
# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train_vectors.shape[1],)),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])
#
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# # Обучение модели
# early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
# model.fit(X_train_vectors, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[early_stopping])
#
#
# model.save("sentiment_analysis_model2.h5")
#
# # Оценка модели
# loss, accuracy = model.evaluate(X_test_vectors, y_test)
# print("Test Accuracy:", accuracy)
#
model = load_model("sentiment_analysis_model.h5")

# Примеры текстов для классификации
texts = [
    "Этот фильм был просто потрясающим, я наслаждался каждой минутой!",
    "Ужасный опыт, не рекомендую этот продукт никому.",
    "Отличное обслуживание в этом ресторане, всегда рад возвращаться.",
    "Качество товара оставляет желать лучшего, я очень разочарован.",
    "Этот книжный магазин - мое новое любимое место, так много интересных книг!",
    "Кофе в этом кафе просто ужасный, никогда больше сюда не пойду."
]

# Предобработка текстов
# Векторизация текстов
vector_size = 1000  # Размер вектора, должен соответствовать тому, который использовался при обучении модели
X_vectors = text_vectorization(texts, model_type='word2vecsave', vector_size=vector_size)



print(X_vectors)
# Классификация текстов
predictions = model.predict(X_vectors)

# Преобразование вероятностей в метки классов
labels = ["Отрицательный", "Положительный"]
predicted_classes = [labels[int(round(pred[0]))] for pred in predictions]

# Вывод результатов
for text, label in zip(texts, predicted_classes):
    print("Текст:", text)
    print("Прогноз:", label)
    print()


