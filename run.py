import tensorflow as tf
import hyperparameters as hp
import numpy as np
import argparse
from preprocessing import get_data
from model import EmotionDetectionModel
# from decoder import TransformerDecoder
# from encoder import TransformerEncoder
from plot import plot, plot_all_sentiments

def train(model, train_inputs, train_labels, padding_index):
    total_loss= 0
    total_accuracy = 0
    num_seen = 0
    print("train_inputs: ", train_inputs.shape)
    num_batches = int(len(train_inputs) / hp.batch_size)
    # for i in range(train_inputs.shape[0]):
    for index, end in enumerate(range(hp.batch_size, len(train_inputs)+1, hp.batch_size)):
        start = end - hp.batch_size
        inputs = train_inputs[start:end]
        labels = train_labels[start:end]
        with tf.GradientTape () as tape:
            # mask = input != padding_index
            # input = tf.boolean_mask(input, mask)
        
            probs = model(inputs)
            
            loss = model.sentiment_loss(probs, labels)
       
            num_seen += 1
        accuracy = model.accuracy(probs, labels)
        total_accuracy+= accuracy
       
        print("total accuracy: ", total_accuracy/num_seen)
        # temp = model.encoder.weights + model.decoder. 
        gradients = tape.gradient(loss, model.trainable_weights) 
        model.optimizer.apply_gradients(zip(gradients,model.trainable_weights))
    total_loss += loss
    ## Compute and report on aggregated statistics
    

    
    print("LOSS: ", total_loss)


def test(model, test_inputs, test_labels, padding_index):
    for index, end in enumerate(range(hp.batch_size, len(test_inputs)+1, hp.batch_size)):
        start = end - hp.batch_size
        inputs = test_inputs[start:end]
        labels = test_labels[start:end]
        probs = model(inputs)
       
        loss = model.sentiment_loss(probs, labels)
        print("LOSS: ", loss)
        accuracy = model.accuracy(probs, labels)
        ## Compute and report on aggregated statistics
        model.accuracy_list.append(accuracy)
        model.loss_list = loss
        print("TEST ACCURACY: ", accuracy)
    return np.mean(model.accuracy_list)

# Main for running the training and testing along with specifying arguments for the user
def main():
    train_posts, val_posts, test_posts, train_emotions, val_emotions, test_emotions, embedding, word2idx = get_data()
    # make sure batches is right
    
    train_posts = tf.keras.preprocessing.sequence.pad_sequences(train_posts, maxlen=hp.maxlen)
    
    test_posts = tf.keras.preprocessing.sequence.pad_sequences(test_posts, maxlen=hp.maxlen)
    model = EmotionDetectionModel(vocab_size=len(word2idx), hidden_size=hp.hidden_size, window_size=hp.window_size, embed_size=hp.embed_size, embedding=embedding, word2idx=word2idx)
    # print('Trainable variables:')
    # for var in model.trainable_weights:
    #     print(var.name)
    
    parser = argparse.ArgumentParser(
    description="Let's analyze some sentiments!",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs",required=False)
    parser.add_argument("--learning_rate",required=False)
    args = parser.parse_args()

    if args.epochs: 
        hp.num_epochs = tf.cast(args.epochs, int)
    if args.learning_rate: 
        hp.learning_rate = tf.cast(args.learning_rate, tf.float32)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), 
        loss=model.sentiment_loss,
        metrics=model.accuracy)
    for i in range(hp.num_epochs): 
        print(f"EPOCH: {i}/{hp.num_epochs}")
        train(model, train_posts, train_emotions, model.word2idx['<pad>'])
        
    plot_all_sentiments(model.loss_list)
    plot(model.accuracy_list, "ACCURACY")
    print(f"FINAL TESTING SCORE: {test(model, test_posts, test_emotions, model.word2idx['<pad>'])}")
    
    model.save('saved_model/my_model')
    model.save('my_model') 
main()
