import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from pattern.en import suggest

def tokenise(string):
    return nltk.word_tokenize(string)


def punkt_tokenise(string):
    tokeniser = WordPunctTokenizer()
    return tokeniser.tokenize(string)


def filter_stopwords(words):
    stops = set(stopwords.words('english'))
    #stops.add('per')
    return {word for word in words if word not in stops}

# this finds synonymns of the first synset, but more sysnsets (other contexts of the word) can also be cycled through
def find_synonymns(word):
    syn = wordnet.synsets(word)[0]
    lemmas = syn.lemmas()
    return [lemma.name() for lemma in lemmas]


def find_stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)


def find_lemma(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)


def fix_spelling(word):
    return suggest(word)
