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
    
#     return processed_text

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
        train_emotions = []
        for i in train_data:
            train_posts.append(train_data[i]["Reddit Post"])

            emotion_per_post = []
            keys = list(train_data[i]["Annotations"].keys())
            for j in range(len(keys)):
                key = keys[j]
                emotions_list = train_data[i]["Annotations"][key] # this is a list
                for k in range(len(emotions_list)):
                    emotion = train_data[i]["Annotations"][keys[j]][k]["Emotion"]
                    emotion_per_post.append(emotion)
            # train_emotions.append(set(emotion_per_post))
            emotion_set = set(emotion_per_post)
            emotion_vector = np.array([1 if x in emotion_set else 0 for x in labels_idx.keys()])
            print(emotion_vector)
            # emotion_vector = np.zeros((7))
            # for x in emotion_set:
            #     emotion_vector[labels_idx[x]] = 1
        
    with open(val_data_filepath) as json_file:
        val_data = json.load(json_file)
        val_posts = []
        val_emotions = []
        for i in val_data:
            val_posts.append(val_data[i]["Reddit Post"])

            emotion_per_post = []
            keys = list(val_data[i]["Annotations"].keys())
            for j in range(len(keys)):
                key = keys[j]
                emotions_list = val_data[i]["Annotations"][key] # this is a list
                for k in range(len(emotions_list)):
                    emotion = val_data[i]["Annotations"][keys[j]][k]["Emotion"]
                    emotion_per_post.append(emotion)
            val_emotions.append(set(emotion_per_post))
        
    with open(test_data_filepath) as json_file:
        test_data = json.load(json_file)
        test_posts = []
        test_emotions = []
        for i in test_data:
            test_posts.append(test_data[i]["Reddit Post"])

            emotion_per_post = []
            keys = list(test_data[i]["Annotations"].keys())
            for j in range(len(keys)):
                key = keys[j]
                emotions_list = test_data[i]["Annotations"][key] # this is a list
                for k in range(len(emotions_list)):
                    emotion = test_data[i]["Annotations"][keys[j]][k]["Emotion"]
                    emotion_per_post.append(emotion)
            test_emotions.append(set(emotion_per_post))
        

    train_posts = np.reshape(np.array(train_posts), (-1, 1))
    val_posts = np.reshape(np.array(val_posts), (-1, 1))
    test_posts = np.reshape(np.array(test_posts), (-1, 1))
    
    train_emotions = np.array(train_emotions)
    val_emotions = np.array(val_emotions)
    test_emotions = np.array(test_emotions)

    return train_posts, val_posts, test_posts, train_emotions, val_emotions, test_emotions