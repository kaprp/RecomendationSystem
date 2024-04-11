import csv
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3
from gensim.models import Word2Vec
from keras.layers import Dense, Input
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
positive = "positive.csv"
negative = "negative.csv"

def read_csv(file):
    words = []
    with open(file, 'r') as file:
        csv_reader = csv.reader(file, delimiter=",")
        i = 0
        # Проход по каждой строке CSV файла и добавление ее в список данных
        for row in csv_reader:
            if i != 0:
                words.append(str(row[0]))
            i += 1
    return words


def preprocess_text(texts):
    morph = pymorphy3.MorphAnalyzer()
    text = texts.lower()
    # Токенизация текста и удаление пунктуации
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]
    # Удаление стоп-слов
    tokens = [token for token in tokens if token.lower() not in stop_words]
    # Лемматизация токенов
    lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return lemmatized_tokens


class Data:
    def __init__(self):
        self.wordvec = None
        self.keras_model = None

    def training_wordvec(self, data, vector_size=100, window=5, min_count=1, workers=4, epochs=10, savename="testing.bin"):
        preprocessed_texts = [preprocess_text(text) for text in data]
        self.wordvec = Word2Vec(sentences=preprocessed_texts, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        self.wordvec.train(preprocessed_texts, total_examples=len(preprocessed_texts), epochs=epochs)
        self.wordvec.save(savename)

    def load_wordvec_model(self, model_path):
        self.wordvec = Word2Vec.load(model_path)

    def get_word_vector(self, word):
        if self.wordvec:
            try:
                return self.wordvec.wv[word]
            except Exception:
                # Возвращаем вектор, близкий к нулевому
                return np.zeros_like(self.wordvec.wv.vectors[0])
        else:
            print("Word2Vec model has not been loaded yet.")

    def get_common_vectors(self, sentences):
        common_vectors = []
        for sentence in sentences:
            preprocessed_text = preprocess_text(sentence)
            vector_sum = np.zeros_like(self.get_word_vector(preprocessed_text[0]))
            for word in preprocessed_text:
                vector_sum += self.get_word_vector(word)
            common_vectors.append(vector_sum / len(preprocessed_text))  # Усреднение векторов слов
        return np.array(common_vectors)

    def build_keras_model(self, input_dim):
        self.keras_model = Sequential([
            Input(shape=(input_dim,)),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

    def train_keras_model(self, X_train, y_train, epochs=10, batch_size=100, validation_data=None):
        if not self.keras_model:
            print("Keras model has not been built yet.")
            return
        self.keras_model.compile(optimizer='adam',
                                 loss='binary_crossentropy',
                                 metrics=['accuracy'])
        self.keras_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate_keras_model(self, X_test, y_test):
        if not self.keras_model:
            print("Keras model has not been built yet.")
            return
        loss, accuracy = self.keras_model.evaluate(X_test, y_test)
        print("Test Accuracy:", accuracy)

    def save_keras_model(self, model_path="2.keras"):
        if not self.keras_model:
            print("Keras модель еще не создана.")
            return
        self.keras_model.save(model_path)
        print("Keras модель успешно сохранена.")

    def load_keras_model(self, model_path):
        self.keras_model = load_model(model_path)
#
# # #
# pos = read_csv(positive)
# neg = read_csv(negative)
#
# all_data = [(i, 1) for i in pos] + [(i, 0) for i in neg]
#
# # Создаем экземпляр класса Data
# data_handler = Data()
#
# # Обучение модели Word2Vec
# # data_handler.training_wordvec(all_data, vector_size=100, window=5, min_count=1, workers=4, epochs=10, savename="word2vec_model.bin")
#
# data_handler.load_wordvec_model("testing.bin")
# #
# # Представление текстов в виде векторов
# X = np.array([data_handler.get_common_vectors([text]) for text, label in all_data])
# y = np.array([label for text, label in all_data])
#
# # Разделение данных на обучающий и тестовый наборы
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Построение Keras модели
# input_dim = X_train.shape[1]
# data_handler.build_keras_model(input_dim)
#
# # Обучение Keras модели
# data_handler.train_keras_model(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
#
# # Оценка модели
# data_handler.evaluate_keras_model(X_test, y_test)
#
# # Сохранение обученной Keras модели
# data_handler.save_keras_model("keras_model2.keras")



#  Пример применения 1 - неудачный

# data_handler.load_keras_model("keras_model2.keras")
#
# sentence = "Хороший звук хорошая цена"
#
# # Получение общего вектора предложения из модели Word2Vec (если это необходимо для вашей модели Keras)
# sentence_vector = data_handler.get_common_vectors([sentence])
#
# # Прогноз тональности с использованием обученной Keras модели
# sentiment_prediction = data_handler.keras_model.predict(sentence_vector)
#
# print("Прогноз тональности с использованием Keras модели:", sentiment_prediction)


 # пример применения 2
# Предположим, у вас есть предложение для анализа тональности
sentence_to_analyze = "Я считаю что данный продукт является одним з лучших в своем сегменте "

# Создаем экземпляр класса Data
data_handler = Data()

# Загружаем обученную модель Word2Vec
data_handler.load_wordvec_model("testing.bin")

# Получаем общий вектор предложения
common_vector = data_handler.get_common_vectors([sentence_to_analyze])

# Загружаем обученную модель Keras
data_handler.load_keras_model("keras_model2.keras")

# Предсказываем тональность предложения с помощью модели Keras
prediction = data_handler.keras_model.predict(common_vector)

print(prediction)

# Выводим результат
if prediction >= 0.5:
    print("Позитивный отзыв")
else:
    print("Негативный отзыв")