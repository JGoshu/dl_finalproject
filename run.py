import tensorflow as tf
import hyperparameters as hp
import numpy as np
import argparse
from preprocessing import get_data
from model import EmotionDetectionModel
# from plot import plot, plot_all_sentiments

def train(model, train_inputs, train_labels):
    indices = tf.range(start=0, limit=tf.shape(train_inputs)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_inputs = tf.gather(train_inputs, shuffled_indices)
    shuffled_labels = tf.gather(train_labels, shuffled_indices)
    
    for i in range(0,len(shuffled_inputs), hp.batch_size):
        print(f"Progress for training: {i}/{len(shuffled_inputs)}")
        inputs = shuffled_inputs[i: i+ hp.batch_size]
        labels = shuffled_labels[i: i + hp.batch_size]

        with tf.GradientTape() as tape:
            forward_pass = model(inputs) 
            loss = model.sentiment_loss(labels, forward_pass)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer(hp.learning_rate).apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    avg_list = []
    num_batches = np.floor(len(test_inputs) / hp.batch_size)
    model.loss_list = np.zeros((num_batches, 8))
    j = 0
    for i in range(0, len(test_inputs), hp.batch_size):
        print(f"Function is now testing: {i}/{len(test_inputs)}")
        inputs = tf.convert_to_tensor(test_inputs[i : i + hp.batch_size])
        labels = test_labels[i : i + hp.batch_size]
        forward_pass = model.call(inputs)
        loss = model.loss_function(labels, forward_pass)
        model.loss_list[j] = loss
        avg_list.append(model.accuracy(forward_pass, labels))
        j += 1

    return sum(avg_list)/len(avg_list)

# Main for running the training and testing along with specifying arguments for the user
def main():
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

    train_posts, val_posts, test_posts, train_emotions, val_emotions, test_emotions = get_data()
    # test_inputs, test_labels = get_test_data()

    for i in range(hp.num_epochs):
        print(f"EPOCH: {i}/{hp.num_epochs}")
        train(EmotionDetectionModel, train_posts, train_emotions)
        
    # plot_all_sentiments(model.loss_list, "LOSS", f"{args.model}_LOSS")
    # plot(model.accuracy_list, "ACCURACY", f"{args.model}_ACCURACY")
    print(f"FINAL TESTING SCORE: {test(EmotionDetectionModel, test_posts, test_emotions)}")
    
main()
