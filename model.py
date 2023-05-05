import numpy as np
import tensorflow as tf

# Model call function 

class EmotionDetectionModel(tf.keras.Model):
    def __init__(self, decoder, hidden_size, window_size, embed_size, **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder(hidden_size, window_size, embed_size)

        self.loss_list = []
        self.accuracy_list = []
        self.final_loss = []

    def call(self, encoded_text):
        return self.decoder(encoded_text)  

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.sentiment_loss = loss 
        self.accuracy = metrics[0]
    
        
    def optimizer(learning_rate):
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Model loss function 
    def sentiment_loss(labels, predictions):
        """Loss function for sentiment analysis"""
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

    def summary_loss(labels, predictions, threshold, perplexity):
        """Loss function"""
        other_emotions = 0
        target_emotion = 0
        for i in predictions:
            if i >= threshold:
                target_emotion += i
            else:
                other_emotions += i
        loss = other_emotions + perplexity - target_emotion
        return loss

    def accuracy(logits, labels):
        """Computes accuracy and returns a float representing the average accuracy"""

        correct_predictions = np.argmax(logits, axis=1)
        num_correct = 0

        for prediction, label in zip(list(correct_predictions), list(labels)):
            if prediction == label: 
                num_correct += 1

        avg = num_correct / len(labels)
        return avg
