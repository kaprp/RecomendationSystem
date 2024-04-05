import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


nltk.download('popular')

sia = SentimentIntensityAnalyzer()
text = "Fuck"

print(sia.polarity_scores(text))



