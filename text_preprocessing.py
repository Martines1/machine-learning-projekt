import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import unicodedata
from nltk import pos_tag
from nltk.corpus import wordnet
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")
nltk.download("stopwords")


lemmatizer = WordNetLemmatizer()
stopword = stopwords.words('english')
stemmer = PorterStemmer()

def remove_interpunction(text: str) -> str:
    text = text.lower()
    text =  re.sub(r'[^\w\s]',' ', text)
    return " ".join(text.split())

def keep_alphabet_only(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return " ".join(text.split())

def remove_non_ascii(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return " ".join(text.split())

def remove_stopWords(text: str) -> str:
    new_text = []
    for word in text.split():
        if word not in stopword:
            new_text.append(word)
    return " ".join(new_text)

def apply_stemmer(text: str) -> str:
    new_text = [stemmer.stem(word) for word in text.split()]
    return " ".join(new_text)

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def apply_lemmantizer(text: str) -> str:
    return " ".join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text.split()])

def remove_tags(text):
    html_tag = '<.*?>'
    text = re.sub(html_tag, ' ',  text)
    return " ".join(text.split())
