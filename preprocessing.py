import os

# import keras_nlp
import tensorflow as tf
from tensorflow import keras
import json
import numpy as np

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# # download the required resources
# nltk.download('punkt')
# nltk.download('stopwords')

# # define the tokenizer, stop words, and stemmer
# tokenizer = nltk.RegexpTokenizer(r'\w+')
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()

# def preprocess(text):
#     # tokenize the text
#     tokens = tokenizer.tokenize(text)
    
#     # lowercase the tokens
#     tokens = [token.lower() for token in tokens]
    
#     # remove stop words
#     tokens = [token for token in tokens if token not in stop_words]
    
#     # apply stemming
#     tokens = [stemmer.stem(token) for token in tokens]
    
#     # join the tokens back into a string
#     processed_text = ' '.join(tokens)
    
    return processed_text

