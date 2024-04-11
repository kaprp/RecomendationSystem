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

    def load_wor2vec(self, model):
        self.wordvec = Word2Vec.load(model)

    def vectorization(self, text):
        vector_sum = 0
        count = 0
        for word in text.split():
            if word in self.wordvec.wv:
                vector_sum += self.wordvec.wv[word]
                count += 1
        return vector_sum / count

    def test_vectorization(self, text):
        print(self.vectorization(text))

    def wordvec_learn(self, texts, vector_size=1000):
        processed_texts = [preprocess_russian_text(text) for text in texts]
        print(processed_texts)
        self.wordvec = Word2Vec(sentences=processed_texts, vector_size=vector_size, min_count=1, workers=4)
        self.wordvec.save("1.bin")
# Md = ModelTonal()

# Md.test_vectorization("стоять")

ps = read_csv(positive)
ns = read_csv(negative)
elems = [(i, 1) for i in ps] + [(i, 0) for i in ns]
random.shuffle(elems)
piks = elems
strs = np.array([p for p, k in piks])
t = ModelTonal()
t.wordvec_learn(strs)
print(t.vectorization("русский"))