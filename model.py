import numpy as np
import tensorflow as tf

# Model call function 
def call(self, inputs):
    """Call function (forward pass)"""
    return self.model(inputs)

# Model loss function 
def sentiment_loss(self, labels, predictions):
    """Loss function"""
    return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

def summary_loss(self, labels, predictions, threshold, perplexity):
    """Loss function"""
    other_emotions = 0
    target_emotion = 0
    for i in predictions:
        if i >= threshold:
            target_emotion += i
        else:
            other_emotions += i
    loss = other_emotions + perplexity - target_emotion
    return 

def accuracy(self, logits, labels):
    """Computes accuracy and returns a float representing the average accuracy"""
    correct_predictions = np.argmax(logits, axis = 1)
    num_correct = 0

    for i, j in zip(list(correct_predictions), list(labels)):
        if i == j: 
            num_correct += 1

    AVERAGE = num_correct / len(labels)
    return AVERAGE

