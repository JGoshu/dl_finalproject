import numpy as np
import tensorflow as tf
from encoder import TransformerEncoder
from decoder import TransformerDecoder

# Model call function 

class EmotionDetectionModel(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size, window_size, embed_size, embedding, word2idx, **kwargs):
        super( EmotionDetectionModel, self).__init__(**kwargs)
        self.decoder = TransformerDecoder(vocab_size, hidden_size, window_size, embed_size, embedding)
        self.encoder = TransformerEncoder(vocab_size=vocab_size, hidden_size=hidden_size, window_size=window_size, embedding=embedding, embed_size=embed_size)
        # self.trainable_variables = self.decoder.trainable_variables + self.encoder.trainable_variables
        self.word2idx= word2idx
        self.loss_list = []
        self.accuracy_list = []
        self.final_loss = []

    def call(self, inputs, labels):
        print("encoded_text: ", inputs)
        encoded_text, embedded_inputs = self.encoder(inputs)
        print("ENCODING: " , encoded_text)
        probs = self.decoder(encoded_text, embedded_inputs)
        return probs

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.sentiment_loss = loss 
        self.accuracy = metrics
        
    def optimizer(self, learning_rate):
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Model loss function 
    def sentiment_loss(self, labels, predictions):
        """Loss function for sentiment analysis"""
        
        return tf.keras.losses.categorical_crossentropy(labels, predictions)

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

    def accuracy(self, logits, labels):
        """Computes accuracy and returns a float representing the average accuracy"""
        
        # correct_predictions = np.argmax(logits, axis=1)
        # num_correct = 0

        # for prediction, label in zip(list(correct_predictions), list(labels)):
        #     if prediction == label: 
        #         num_correct += 1

        # avg = num_correct / len(labels)
        # return avg

        correct_classes = tf.argmax(logits, axis=-1) == labels
        acc = tf.reduce_mean(tf.cast(correct_classes, tf.float32))
        return acc