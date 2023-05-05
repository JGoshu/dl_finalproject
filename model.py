import numpy as np
import tensorflow as tf
from decoder import TransformerDecoder

# Model call function 

class EmotionDetectionModel(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size, window_size, embed_size, embedding, word2idx, **kwargs):
        super().__init__(**kwargs)
        self.decoder = TransformerDecoder(vocab_size, hidden_size, window_size, embed_size, embedding)
        self.word2idx= word2idx
        self.loss_list = []
        self.accuracy_list = []
        self.final_loss = []

    def call(self, encoded_text):
        print("encoded_text: ", encoded_text)
        return self.decoder(encoded_text)  

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.sentiment_loss = loss 
        self.accuracy = metrics[0]
    
        
    def optimizer(self, learning_rate):
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Model loss function 
    def sentiment_loss(self, labels, predictions, mask):
        """Loss function for sentiment analysis"""
        masked_labs = tf.boolean_mask(labels, mask)
        masked_prbs = tf.boolean_mask(predictions, mask)
        return tf.keras.losses.sparse_categorical_crossentropy(masked_labs, masked_prbs)

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
        return loss

    def accuracy(self, logits, labels, mask):
        """Computes accuracy and returns a float representing the average accuracy"""
        
        # correct_predictions = np.argmax(logits, axis=1)
        # num_correct = 0

        # for prediction, label in zip(list(correct_predictions), list(labels)):
        #     if prediction == label: 
        #         num_correct += 1

        # avg = num_correct / len(labels)
        # return avg

        correct_classes = tf.argmax(logits, axis=-1) == labels
        acc = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask))
        return acc