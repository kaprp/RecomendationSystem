# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
#
#
# nltk.download('popular')
#
# sia = SentimentIntensityAnalyzer()
# text = "Fuck"
#
# print(sia.polarity_scores(text))
#
#
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import pymorphy2
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import gensim
# import numpy as np
#
# from negative import negative_texts
# from positive import positive_texts
#
# # Загрузим список стоп-слов для русского языка из NLTK
# nltk.download('stopwords')
#
# # Создадим объект для морфологического анализа
# morph_analyzer = pymorphy2.MorphAnalyzer()
#
# common = [positive_texts, negative_texts]
#
#
# # Функция для предобработки русского текста
# def preprocess_russian_text(text):
#     # Инициализация анализатора pymorphy2
#     morph = pymorphy2.MorphAnalyzer()
#
#     text = text.lower()
#
#     # Токенизация текста и удаление пунктуации
#     tokens = nltk.word_tokenize(text)
#     tokens = [token for token in tokens if token.isalnum()]
#
#     # Удаление стоп-слов
#     stop_words = set(stopwords.words('russian'))
#     tokens = [token for token in tokens if token.lower() not in stop_words]
#
#     # Лемматизация токенов
#     lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens]
#
#     return ' '.join(lemmatized_tokens)
#
#
# def vectorize_text_with_rusentilex(text, rusentilex):
#     # Предположим, что rusentilex - это словарь RuSentiLex, где ключи - слова, значения - их тональность
#
#     # Разделение текста на отдельные слова
#     words = text.split()
#
#     # Инициализация счетчиков для позитивных и негативных слов
#     positive_count = 0
#     negative_count = 0
#
#     # Подсчет количества позитивных и негативных слов в тексте
#     for word in words:
#         if word in rusentilex:
#             if rusentilex[word] == 'positive':
#                 positive_count += 1
#             elif rusentilex[word] == 'negative':
#                 negative_count += 1
#
#     # Возвращение вектора с количеством позитивных и негативных слов
#     return [positive_count, negative_count]
#
#
# import numpy as np
#
# # Предположим, что rusentilex - это словарь RuSentiLex, где ключи - слова, значения - их тональность
# rusentilex = {'хороший': 'positive', 'плохой': 'negative', 'замечательный': 'positive', 'ужасный': 'negative'}
#
#
# # Преобразование текста в векторное представление
# def vectorize_text_with_rusentilex(text, rusentilex):
#     # Разделение текста на отдельные слова
#     words = text.split()
#
#     # Инициализация счетчиков для позитивных и негативных слов
#     positive_count = 0
#     negative_count = 0
#
#     # Подсчет количества позитивных и негативных слов в тексте
#     for word in words:
#         if word in rusentilex:
#             if rusentilex[word] == 'positive':
#                 positive_count += 1
#             elif rusentilex[word] == 'negative':
#                 negative_count += 1
#
#     # Возвращение вектора с количеством позитивных и негативных слов
#     return np.array([positive_count, negative_count])
#
#
# # Пример текстов для обучения
# training_texts = [
#     "Этот фильм был просто замечательным!",
#     "Эта книга мне не понравилась, слишком скучная.",
#     "Моя поездка была ужасной из-за плохой погоды.",
#     "Я рад, что наконец-то нашел хороший ресторан в этом районе."
# ]
#
# # Метки классов для обучения (положительные - 1, отрицательные - 0)
# training_labels = np.array([1, 0, 0, 1])
#
# # Обучение нейросети
# from keras.models import Sequential
# from keras.layers import Dense
#
# # Создание модели нейросети
# model = Sequential()
# model.add(Dense(8, input_dim=2, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# # Компиляция модели
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Преобразование текстов в векторы и обучение модели
# X_train = np.array([vectorize_text_with_rusentilex(text, rusentilex) for text in training_texts])
# y_train = training_labels
# model.fit(X_train, y_train, epochs=100, batch_size=2)
#
# # Пример текста для оценки тональности
# test_text = "Этот день был просто ужасным!"
#
# # Преобразование текста в вектор и оценка тональности с помощью модели
# X_test = np.array([vectorize_text_with_rusentilex(test_text, rusentilex)])
# prediction = model.predict(X_test)
#
# # Вывод результата
# if prediction > 0.5:
#     print("Текст имеет положительную тональность.")
# else:
#     print("Текст имеет отрицательную тональность.")