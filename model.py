import numpy as np
import tensorflow as tf
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from transformer import *

# Model call function 

class EmotionDetectionModel(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size, window_size, embed_size, embedding, word2idx, **kwargs):
        super( EmotionDetectionModel, self).__init__(**kwargs)
        # self.decoder = TransformerDecoder(vocab_size, hidden_size, window_size, embed_size, embedding)
        self.embedding_matrix = embedding
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.embedding_matrix.shape[0],
            output_dim=self.embedding_matrix.shape[1],
            weights=[self.embedding_matrix],
            trainable=True
        )
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation="relu" )
        self.dense2 = tf.keras.layers.Dense(7, activation= "sigmoid")
        # self.encoder = TransformerEncoder(vocab_size=vocab_size, hidden_size=hidden_size, window_size=window_size, embedding=embedding, embed_size=embed_size)
        # self.trainable_variables = self.decoder.trainable_variables + self.encoder.trainable_variables
        
        self.enc_positional_embedding = TokenAndPositionEmbedding(self.embedding_layer, hp.maxlen, vocab_size, hp.embed_size)
        self.enc_transformerblock = TransformerBlock(embed_size=embed_size,  is_decoder=False)
        self.word2idx= word2idx
        self.loss_list = []
        self.accuracy_list = []
        self.final_loss = []
        self.dec_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size*2, activation="leaky_relu"),
            tf.keras.layers.Dense(hidden_size, activation="leaky_relu"),
        ])
        #use pre-trained embedding!?
        self.dec_transformer_block = TransformerBlock(embed_size=embed_size, is_decoder=True)
        self.dec_positional_embedding = TokenAndPositionEmbedding(self.dec_embedding, hp.maxlen, vocab_size, hp.embed_size)
        self.dec_pooling = tf.keras.layers.GlobalMaxPooling2D(input_shape=(400, 400, 300)) #or alternative for dimensional reduction
        self.dec_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(7, "sigmoid"),
        ]) 
        self.dropout= tf.keras.layers.Dropout(0.5)

    def call(self, x_in, labels):
        x = x_in
        # x = self.embedding_layer(inputs)
        # x = self.encoder(x)
        x_emb= self.enc_positional_embedding(x)
        x = self.enc_transformerblock(x_emb)
        x = self.dropout(x)
        # x = self.dense1(x)
        # x = self.dense2(x)
        # x = tf.reduce_mean(x, axis=0)
        x = self.dec_transformer_block(x, context_sequence=tf.cast(x_emb, np.float32), is_decoder=True)
        x = np.expand_dims(x, axis=0)
        x = self.dec_pooling(x)
        x = self.dec_classifier(x)

        # print("encoded_text: ", inputs)
        # encoded_text, embedded_inputs = self.encoder(inputs)
        # print("ENCODING: " , encoded_text)
        # x = self.decoder(x, xe)
        return x

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