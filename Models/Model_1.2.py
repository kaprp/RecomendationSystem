import csv
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3
import gensim
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import random



nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
positive = "positive.csv"
negative = "negative.csv"


def preprocess_russian_text(text):
    # Инициализация анализатора pymorphy2
    morph = pymorphy3.MorphAnalyzer()

    text = text.lower()

    # Токенизация текста и удаление пунктуации
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]

    # Удаление стоп-слов
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Лемматизация токенов
    lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens]

    return lemmatized_tokens


def test_preprocess(text):
    print(preprocess_russian_text(text))


def read_csv(file):
    words = []
    with open(file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter=";")
        i = 0
        # Проход по каждой строке CSV файла и добавление ее в список данных
        for row in csv_reader:
            if i != 0:
                words.append(str(row[3]))
            i += 1
    return words


# n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']

class ModelTonal:
    def __init__(self):
        self.model = None
        self.wordvec = None

    def load_model_main(self, model):
        self.model = load_model(model)

    def load_word2vec(self, model):
        self.wordvec = Word2Vec.load(model)

    def vectorization(self, text):
        # Разбиваем текст на слова
        words = text.split()
        word_vectors = []

        # Проходимся по каждому слову
        for word in words:
            # Проверяем, есть ли вектор для данного слова в модели Word2Vec
            if word in self.wordvec.wv:
                # Если вектор есть, добавляем его к списку
                word_vector = self.wordvec.wv[word]
                word_vectors.append(word_vector)

        # Преобразуем список в массив numpy
        if word_vectors:
            return np.array(word_vectors)
        else:
            # Если нет известных слов, возвращаем нулевой вектор такой же длины, как в модели Word2Vec
            return np.zeros_like(self.wordvec.wv['текст'])

    def learning_wordvec(self, data, vector_size=100, window=5, min_count=1, workers=4, epochs=10, savename="testing.bin"):
        preprocessed_texts = [preprocess_russian_text(text) for text in data]
        # Обучение модели Word2Vec на предварительно обработанных данных
        self.wordvec = Word2Vec(sentences=preprocessed_texts, vector_size=vector_size, window=window, min_count=min_count,
                         workers=workers)
        self.wordvec.train(preprocessed_texts, total_examples=len(preprocessed_texts), epochs=epochs)
        self.wordvec.save(savename)

    def vector_wordvec(self, texts):
        vector_data = []
        for text in texts:
            preprocessed_text = preprocess_russian_text(text)
            vector = []
            for token in preprocessed_text:
                # Проверка наличия токена в словаре модели Word2Vec
                if token in self.wordvec.wv.key_to_index:
                    vector.append(self.wordvec.wv[token])
                else:
                    # Если токен отсутствует в словаре, добавляем нулевой вектор
                    vector.append([0] * len(self.wordvec.wv))
            vector_data.append(vector)
        print(vector_data)
        return vector_data


pos = read_csv(positive)
neg = read_csv(negative)

all = [(i , 1) for i in pos] + [(i , 0) for i in neg]

data = [a for a,b in all]

model_tonal = ModelTonal()

model_tonal.learning_wordvec(data)

text = "Любимая"

model_tonal.vector_wordvec([text])

# # Загружаем предварительно обученную модель Word2Vec
# model_tonal.load_word2vec("3.bin")
#
# # Загружаем или обучаем модель для анализа тональности
# # Например, мы загружаем предварительно обученную модель
# model_tonal.load_model_main("1.keras")
#
# # Пример текста для анализа
# text = "Ужасный"
#
# # Получаем общую тональность текста
# tone = model_tonal.text_tone(text)
#
# # Выводим результат
# print("Тональность текста:", tone)