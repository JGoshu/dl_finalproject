import numpy as np
import tensorflow as tf

# Model call function 

class EmotionDetectionModel(tf.keras.Model):
    def __init__(self, decoder, **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder

    @tf.function
    def call(self, encoded_text):
        return self.decoder(encoded_text)  

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]

    # Model loss function 
    def sentiment_loss(self, labels, predictions):
        """Loss function for sentiment analysis"""
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

        correct_predictions = np.argmax(logits, axis=1)
        num_correct = 0

        for prediction, label in zip(list(correct_predictions), list(labels)):
            if prediction == label: 
                num_correct += 1

        avg = num_correct / len(labels)
        return avg
