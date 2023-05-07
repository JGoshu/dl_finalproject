import os
import hyperparameters as hp

# import keras_nlp
import tensorflow as tf
from tensorflow import keras
import json
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# download the required resources
nltk.download('punkt')
nltk.download('stopwords')

# define the tokenizer, stop words, and stemmer
tokenizer = nltk.RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def load_embedding():
        data_dict = {}
        # Load the embedding file
        with open('data/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Extract the embedding matrix
        embedding_matrix = np.zeros((len(lines), 300))
        for i, line in enumerate(lines):
            parts = line.strip().split(' ')
            embedding_matrix[i] = np.array(parts[1:], dtype=np.float32)
            data_dict[parts[0]] = i

        # Return the embedding matrix
        data_dict['<pad>'] = 0 #may need to change
        return embedding_matrix, data_dict
def preprocess_post(text):
    # tokenize the text
    tokens = tokenizer.tokenize(text)
    
    # lowercase the tokens
    tokens = [token.lower() for token in tokens]
    
    # remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    
    # join the tokens back into a string
    processed_text = ' '.join(tokens)
    
    return processed_text

def get_data():

    train_data_filepath = "data/train_anonymized_with_posts.json"
    val_data_filepath = "data/test_anonymized_with_posts.json"
    test_data_filepath = "data/val_anonymized_with_posts.json"

    labels_idx = {"anger" : 0, 
                  "anticipation" : 1, 
                  "joy" : 2, 
                  "trust" : 3,
                  "fear" : 4,
                  "sadness" : 5,
                  "disgust" : 6}


    with open(train_data_filepath) as json_file:
        train_data = json.load(json_file)
        train_posts = []
        train_emotions = np.zeros((len(train_data), 7))
        for index, i in enumerate(train_data):
            train_posts.append(preprocess_post((train_data[i]["Reddit Post"])))
            
            emotion_per_post = []
            keys = list(train_data[i]["Annotations"].keys())
            for j in range(len(keys)):
                key = keys[j]
                emotions_list = train_data[i]["Annotations"][key] # this is a list
                for k in range(len(emotions_list)):
                    emotion = train_data[i]["Annotations"][keys[j]][k]["Emotion"]
                    emotion_per_post.append(emotion)
            emotion_set = set(emotion_per_post)
            emotion_vector = np.array([1 if x in emotion_set else 0 for x in labels_idx.keys()])
            train_emotions[index] = emotion_vector
  
    with open(val_data_filepath) as json_file:
        val_data = json.load(json_file)
        val_posts = []
        val_emotions = np.zeros((len(val_data), 7))
        for index, i in enumerate(val_data):
            val_posts.append(preprocess_post(val_data[i]["Reddit Post"]))

            emotion_per_post = []
            keys = list(val_data[i]["Annotations"].keys())
            for j in range(len(keys)):
                key = keys[j]
                emotions_list = val_data[i]["Annotations"][key] # this is a list
                for k in range(len(emotions_list)):
                    emotion = val_data[i]["Annotations"][keys[j]][k]["Emotion"]
                    emotion_per_post.append(emotion)
            emotion_set = set(emotion_per_post)
            emotion_vector = np.array([1 if x in emotion_set else 0 for x in labels_idx.keys()])
            val_emotions[index] = emotion_vector
        
    with open(test_data_filepath) as json_file:
        test_data = json.load(json_file)
        test_posts = []
        test_emotions = np.zeros((len(test_data), 7))
        for index, i in enumerate(test_data):
            test_posts.append(preprocess_post(test_data[i]["Reddit Post"]))

            emotion_per_post = []
            keys = list(test_data[i]["Annotations"].keys())
            for j in range(len(keys)):
                key = keys[j]
                emotions_list = test_data[i]["Annotations"][key] # this is a list
                for k in range(len(emotions_list)):
                    emotion = test_data[i]["Annotations"][keys[j]][k]["Emotion"]
                    emotion_per_post.append(emotion)
            emotion_set = set(emotion_per_post)
            emotion_vector = np.array([1 if x in emotion_set else 0 for x in labels_idx.keys()])
            test_emotions[index] = emotion_vector
        
    
    train_posts = np.reshape(np.array(train_posts), (-1, 1))
    val_posts = np.reshape(np.array(val_posts), (-1, 1))
    test_posts = np.reshape(np.array(test_posts), (-1, 1))

    
    train_emotions = np.array(train_emotions)
    val_emotions = np.array(val_emotions)
    test_emotions = np.array(test_emotions)
    embedding, word2idx = load_embedding()
    vocab_size = hp.vocab_size
    train_tokenized = np.zeros((train_posts.shape[0], hp.window_size + 1))
    val_tokenized = np.zeros((train_posts.shape[0], hp.window_size + 1))
    test_tokenized = np.zeros((train_posts.shape[0], hp.window_size + 1))
    for index, post in enumerate(train_posts):
        words = post[0].split()
        words += (hp.window_size + 1 - len(words)) * ['<pad>']
        new_words = []
        for i, word in enumerate(words):
            if word in word2idx:
                new_words.append(word2idx[word])
            else:
                word2idx[word] = vocab_size
                new_words.append(vocab_size)
                vocab_size += 1
        train_tokenized[index] = new_words

    for index, post in enumerate(val_posts):
        words = post[0].split()
        words += (hp.window_size + 1 - len(words)) * ['<pad>']
        new_words = []
        for i, word in enumerate(words):
            if word in word2idx:
                new_words.append(word2idx[word])
            else:
                word2idx[word] = vocab_size
                new_words.append(vocab_size)
                vocab_size += 1
        val_tokenized[index] = new_words

    for index, post in enumerate(test_posts):
        words = post[0].split()
        words += (hp.window_size + 1 - len(words)) * ['<pad>']
        new_words = []
        for i, word in enumerate(words):
            if word in word2idx:
                new_words.append(word2idx[word])
            else:
                word2idx[word] = vocab_size
                new_words.append(vocab_size)
                vocab_size += 1   ####### TODO: REMOVE LATER OR TRY TRAINING EMBEDDING ########
        test_tokenized[index] = new_words

    return train_tokenized, val_tokenized, test_tokenized, train_emotions, val_emotions, test_emotions, embedding, word2idx


