# pip install spacy
# python -m spacy download ru_core_news_md
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3
import csv
import spacy
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
positive = "positive.csv"
negative = "negative.csv"


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

class LossAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: Loss - {logs['loss']}, Accuracy - {logs['accuracy']}")

class TextClassifier:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.nlp_ru = spacy.load("ru_core_news_md")

    def preprocess_text(self, texts):
        tokens_arr = []
        for text in texts:
            # Инициализация анализатора pymorphy3
            morph = pymorphy3.MorphAnalyzer()
            text = text.lower()
            # Токенизация текста и удаление пунктуации
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token.isalnum()]
            # Удаление стоп-слов
            tokens = [token for token in tokens if token.lower() not in stop_words]
            # Лемматизация токенов
            lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens]
            tokens_arr.append(lemmatized_tokens)
        return tokens_arr

    def vectorize_text(self, elems):
        texts = [self.preprocess_text(i) for i in elems]
        array_text = []
        for i in texts:
            arr = []
            for j in i:
                arr.append(self.nlp_ru(j).vector)
            array_text.append(arr)
        print(array_text)
        return array_text

    def train_model(self, X_train, y_train):
        model = Sequential()
        model.add(Embedding(input_dim=X_train.shape[0], output_dim=300, weights=[X_train], trainable=False))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        loss_accuracy_callback = LossAccuracyCallback()
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[loss_accuracy_callback])
        return model


neg = read_csv(negative)
pos = read_csv(positive)

classifier = TextClassifier()
word_vectors_pos = classifier.vectorize_text(pos)
word_vectors_neg = classifier.vectorize_text(pos)

all = [(i,1) for i in word_vectors_pos] + [(i,0) for i in word_vectors_neg]

word_vectors = [i for i, cl in all][:100]
labels = [cl for i, cl in all][:100]

X_train, X_test, y_train, y_test = train_test_split(word_vectors, labels, test_size=0.2, random_state=42)
trained_model = classifier.train_model(X_train, y_train)

