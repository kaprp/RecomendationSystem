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

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузим список стоп-слов для русского языка из NLTK
nltk.download('stopwords')

# Создадим объект для морфологического анализа
morph_analyzer = pymorphy2.MorphAnalyzer()


# Функция для предобработки русского текста
def preprocess_russian_text(text):
    # Инициализация анализатора pymorphy2
    morph = pymorphy2.MorphAnalyzer()

    # Токенизация текста и удаление пунктуации
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]

    # Удаление стоп-слов
    stop_words = set(stopwords.words('russian'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Лемматизация токенов
    lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens]

    return ' '.join(lemmatized_tokens)

# Пример данных для обучения модели
positive_texts  = [
    "Сегодня прекрасный день, полный радости и удивительных возможностей.",
    "Я рад проснуться и начать новый день с улыбкой на лице.",
    "Всё вокруг так красиво и ярко, словно природа сама радуется вместе со мной.",
    "Я благодарен за все возможности, которые дарит мне жизнь, и готов использовать их на полную мощность.",
    "У меня есть замечательные друзья и близкие, которые всегда поддержат меня в трудную минуту.",
    "Мир так прекрасен, и я рад, что я часть этого удивительного мира.",
    "Я полон энергии и готов принять все вызовы, которые принесет этот день.",
    "У меня есть мечты, и я знаю, что они сбудутся, потому что я верю в себя и свои силы.",
    "Я благодарен за каждый момент жизни и готов делать его еще более ярким и насыщенным.",
    "Я люблю себя и ценю свою жизнь, и это делает меня счастливым.",
    "Сегодняшний день будет прекрасным, потому что я сам создаю свою судьбу и наполняю ее радостью и любовью."
]
negative_texts =  [
    "Сегодня ужасный день, полный горя и разочарований.",
    "Я расстроен просыпаться и начинать новый день с тяжёлым сердцем.",
    "Всё вокруг такое унылое и серое, словно природа печалится вместе со мной.",
    "Я разочарован во всех возможностях, которые дарит мне жизнь, и не вижу смысла использовать их.",
    "У меня нет настоящих друзей и близких, которые могли бы поддержать меня в трудную минуту.",
    "Мир так безысходен, и я печален, что я часть этого унылого мира.",
    "Я без энергии и не готов принять даже малейший вызов, который принесет этот день.",
    "У меня есть мечты, но я сомневаюсь, что они когда-либо сбудутся, потому что я не верю в себя и свои силы.",
    "Я разочарован в каждом моменте жизни и не готов делать его даже немного лучше.",
    "Я ненавижу себя и не ценю свою жизнь, и это делает меня несчастным.",
    "Сегодняшний день будет ужасным, потому что я не в состоянии создать свою судьбу и наполнять её радостью и любовью."
]

# Создадим списки для текстов и их меток
texts = [preprocess_russian_text(text) for text in positive_texts + negative_texts]
labels = [1] * len(positive_texts) + [0] * len(negative_texts)

# Разделим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Создадим конвейер для обработки текста и классификации
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear'))
])

# Обучим модель на обучающем наборе
text_clf.fit(X_train, y_train)

# Оценим качество модели на тестовом наборе
predicted = text_clf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print("Accuracy:", accuracy)

input_text = preprocess_russian_text("Это был ужасный день!")

# Классификация текста с использованием обученной модели
predicted_label = text_clf.predict([input_text])[0]

# Вывод результата
if predicted_label == 1:
    print("Модель считает текст положительным.")
else:
    print("Модель считает текст отрицательным.")