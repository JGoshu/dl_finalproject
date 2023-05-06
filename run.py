import tensorflow as tf
import hyperparameters as hp
import numpy as np
import argparse
from preprocessing import get_data
from model import EmotionDetectionModel
from decoder import TransformerDecoder
from encoder import TransformerEncoder
# from plot import plot, plot_all_sentiments

def train(model, train_inputs, train_labels, padding_index):
    total_loss = total_seen = total_correct = 0  
    for i in range(train_inputs.shape[0]):
        input = train_inputs[i]
        #should be encoder output.

        print("DECODER :", input)
        labels = train_labels[i]
        print("decoder_labels: ", labels)
        ## TODO
        with tf.GradientTape () as tape:
            mask = input != padding_index
            input = tf.boolean_mask(input, mask)
            probs = model(input)
            loss = model.sentiment_loss(probs, labels)
        gradients = tape.gradient(loss, model.trainable_variables) 
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        accuracy = model.accuracy(probs, labels)
        ## Compute and report on aggregated statistics
        total_loss += loss

        



def test(model, test_inputs, test_labels, padding_index):
    total_loss = total_seen = total_correct = 0
    input = test_inputs[i]
    print("DECODER :", input)
    labels = test_labels[i]
    print("decoder_labels: ", labels)
    ## TODO
    mask = input != padding_index
    input = tf.boolean_mask(input, mask)
    probs = model(input)
    loss = model.sentiment_loss(probs, labels)
    accuracy = model.accuracy(probs, labels)
    ## Compute and report on aggregated statistics
    total_loss += loss

# Main for running the training and testing along with specifying arguments for the user
def main():
    train_posts, val_posts, test_posts, train_emotions, val_emotions, test_emotions, embedding, word2idx = get_data()
    model = EmotionDetectionModel(vocab_size=hp.vocab_size, hidden_size=hp.hidden_size, window_size=hp.window_size, embed_size=hp.embed_size, embedding=embedding, word2idx=word2idx)
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


    # test_inputs, test_labels = get_test_data()

    for i in range(hp.num_epochs):
        print(f"EPOCH: {i}/{hp.num_epochs}")
        train(model, train_posts, train_emotions, model.word2idx['<pad>'])
        
    # plot_all_sentiments(model.loss_list, "LOSS", f"{args.model}_LOSS")
    # plot(model.accuracy_list, "ACCURACY", f"{args.model}_ACCURACY")
    print(f"FINAL TESTING SCORE: {test(model, test_posts, test_emotions, model.word2idx['<pad>'])}")
    
main()
