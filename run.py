import tensorflow as tf
import hyperparameters as hp

def train(model, train_inputs, train_labels):
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
    indices = tf.range(start=0, limit=tf.shape(train_inputs)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_inputs = tf.gather(train_inputs, shuffled_indices)
    shuffled_labels = tf.gather(train_labels, shuffled_indices)
    for i in range(0,len(shuffled_inputs), hp.batch_size):
        print(f"Progress for training: {i}/{len(shuffled_inputs)}")
        inputs = shuffled_inputs[i: i+ hp.batch_size]
        labels = shuffled_labels[i: i + hp.batch_size]
        with tf.GradientTape() as tape:

            forward_pass = model.call(inputs) 
            loss = model.loss_function(labels, forward_pass)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    avg_list = []
    for i in range(0,len(test_inputs), hp.batch_size):
        print(f"Function is now testing: {i}/{len(test_inputs)}")
        images = test_inputs[i : i + hp.batch_size]
        labels = test_labels[i : i + hp.batch_size]
        images = tf.convert_to_tensor(images)
        forward_pass = model.call(images)
        loss = model.loss_function(labels, forward_pass)
        model.loss_list.append(loss)
        avg_list.append(model.accuracy(forward_pass, labels))

    return sum(avg_list)/len(avg_list)
