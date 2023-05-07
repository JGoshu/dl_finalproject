import tensorflow as tf
from tensorflow import keras
import numpy as np
import hyperparameters as hp
import warnings

class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, is_decoder=False, is_self_attention,**kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask
        self.is_decoder = is_decoder
        self.is_self_attention = is_self_attention

    def call(self, inputs):
        """
        STUDENT MUST WRITE:

        Computes attention given key and query matrices.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        K, Q = inputs
        window_size_queries = Q.get_shape()[0]  # window size of queries
        window_size_keys    = K.get_shape()[0]  # window size of keys
        print("window_size_queries", window_size_queries)
        ## Fill triangle below diagonal of matrix with negative infinity and top part with 0.
        ## This helps to avoid over-contribution, since adjacency matrix is symmetric across diagonal. 
        ## Tile this upward to be compatible with addition against computed attention scores.
        mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        # TODO:
        # 1) compute attention weights using queries and key matrices 
        #       - if use_mask==True, then make sure to add the attention mask before softmax
        # 2) return the attention matrix
        attention_weights = None
        if self.is_decoder:
            attention_weights = tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])) / tf.math.sqrt(tf.cast(window_size_keys, tf.float32))
        else:
            attention_weights = tf.matmul(Q, tf.transpose(K, perm=[1,0])) / tf.math.sqrt(tf.cast(window_size_keys, tf.float32))
        if self.use_mask:
            attention_weights = attention_weights + atten_mask
        return tf.nn.softmax(attention_weights)
        # Check lecture slides for how to compute self-attention
        # Remember:
        # - Q is [batch_size x window_size_queries x embedding_size]
        # - K is [batch_size x window_size_keys x embedding_size]
        # - Mask is [batch_size x window_size_queries x window_size_keys]

        # Here, queries are matrix multiplied with the transpose of keys to produce for every query vector, weights per key vector.
        # This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
        # Those weights are then used to create linear combinations of the corresponding values for each query.
        # Those queries will become the new embeddings. Return attention score as per lecture slides.



class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, is_decoder, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        # TODO:
        # Initialize the weight matrices for K, V, and Q.
        self.K = tf.Variable(tf.random.truncated_normal([input_size, output_size], stddev=0.1))
        self.V = tf.Variable(tf.random.truncated_normal([input_size, output_size], stddev=0.1))
        self.Q = tf.Variable(tf.random.truncated_normal([input_size, output_size], stddev=0.1))

        # if is_decoder and not is_self_attention:
        #     self.Q = tf.Variable(tf.random.truncated_normal([7, 300], stddev=0.1))
        # else:
        #     self.Q = tf.Variable(tf.random.truncated_normal([input_size, output_size], stddev=0.1))
        
     #    
        self.attention_matrix = AttentionMatrix(use_mask=self.use_mask, is_decoder=is_decoder, is_self_attention=is_self_attention)
        # They should be able to multiply an input_size vector to produce an output_size vector


    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        STUDENT MUST WRITE:

        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        # TODO:
        # - Apply 3 matrix products to turn inputs into keys, values, and queries. 
        # - You will need to use tf.tensordot for this.
        # - Call your AttentionMatrix layer with the keys and queries.
        # - Apply the attention matrix to the values.

        K = tf.tensordot(inputs_for_keys, self.K, axes=1)
        V = tf.tensordot(inputs_for_values, self.V, axes=1)
        Q = tf.tensordot(inputs_for_queries, self.Q, axes=1)
     
        attention_weights = self.attention_matrix((K, Q))
       
        attention = tf.matmul(attention_weights, V)
        return attention

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, is_decoder=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        # TODO:
        # 1) Define the Feed Forward, self-attention, encoder-decoder-attention, and layer normalization layers
        # 2) For 2470 students, use multiheaded attention

        self.ff_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_size, activation='leaky_relu'),
            tf.keras.layers.Dense(embed_size)
        ])
        self.self_atten         = AttentionHead(embed_size, embed_size, True, is_decoder) 
        self.self_context_atten = AttentionHead(embed_size, embed_size, False, is_decoder)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    @tf.function
    def call(self, inputs, context_sequence=None, is_decoder=False):
        """
        This functions calls a transformer block.

        TODO:
        1) compute MASKED attention on the inputs
        2) residual connection and layer normalization
        3) computed UNMASKED attention using context
        4) residual connection and layer normalization
        5) feed forward layer
        6) residual layer and layer normalization
        7) return relu of tensor

        NOTES: This article may be of great use:
        https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """
        in_attention = self.self_atten(inputs, inputs, inputs)
        in_attention = in_attention + inputs
        in_attention_norm = self.layer_norm(in_attention)
        if is_decoder:
            context_attention = self.self_context_atten(in_attention_norm, in_attention_norm, context_sequence)
            context_attention = context_attention + in_attention_norm
            context_attention_norm = self.layer_norm(context_attention)
            ff_out = self.ff_layer(context_attention_norm)
            ff_out = ff_out + context_attention_norm
        else:
            ff_out = self.ff_layer(in_attention_norm)
            ff_out = ff_out + in_attention_norm
        ff_out_norm = self.layer_norm(ff_out)
        return ff_out_norm

    

    #NOTE: not clas made, consider switching back
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, token_embedding, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = token_embedding
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        # maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=hp.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions