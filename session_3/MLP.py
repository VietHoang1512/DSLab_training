import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

NUM_CLASSES = 20


class MLP:
    def __init__(self, vocab_size, hidden_size):
        self.real_Y = tf.placeholder(tf.int32, shape=[None,])
        self.X = tf.placeholder(tf.float32, shape=[None, vocab_size])
        self.vocab_size = vocab_size
        self.hidden_state = hidden_size

    def build_graph(self):
        weights_1 = tf.get_variable(
            name="weights_input_hidden",
            shape=(self.vocab_size, self.hidden_state),
            initializer=tf.random_normal_initializer(seed=42),
        )
        biases_1 = tf.get_variable(
            name="biases_input_hidden",
            shape=self.hidden_state,
            initializer=tf.random_normal_initializer(seed=42),
        )
        weights_2 = tf.get_variable(
            name="weights_hidden_output",
            shape=(self.hidden_state, NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=42),
        )
        biases_2 = tf.get_variable(
            name="biases_hidden_output",
            shape=NUM_CLASSES,
            initializer=tf.random_normal_initializer(seed=42),
        )

        # hidden = self.X * weights_1 + biases_1
        hidden = tf.matmul(self.X, weights_1) + biases_1
        hidden = tf.sigmoid(hidden)
        logits = tf.matmul(hidden, weights_2) + biases_2

        labels_one_hot = tf.one_hot(
            indices=self.real_Y, depth=NUM_CLASSES, dtype=tf.float32
        )
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot, logits=logits
        )
        loss = tf.reduce_mean(loss)

        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis=1)
        predicted_labels = tf.squeeze(predicted_labels)

        return predicted_labels, loss

    @staticmethod
    def trainer(loss, learning_rate):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op


class DataReader:
    def __init__(self, data_path, batch_size, vocab_size):
        self.batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.readlines()

        self.data = []
        self.labels = []
        for data_id, line in enumerate(d_lines):
            vector = [0.0 for _ in range(vocab_size)]
            features = line.split("<fff>")
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split()
            for token in tokens:
                index = int(token.split(":")[0])
                value = float(token.split(":")[1])
                vector[index] = value
            self.data.append(vector)
            self.labels.append(label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.num_epoch = 0
        self.batch_id = 0

    def next_batch(self):
        start = self.batch_id * self.batch_size
        end = start + self.batch_size
        self.batch_id += 1

        if end + self.batch_size > len(self.data):
            end = len(self.data)
            self.num_epoch += 1
            self.batch_id = 0
            indices = list(range(len(self.data)))
            random.seed(2018)
            random.shuffle(indices)
            self.data, self.labels = self.data[indices], self.labels[indices]

        return self.data[start:end], self.labels[start:end]


def save_parameters(name, value, epoch):
    filename = name.replace(":", "-colon-") + "-epoch-{}.txt".format(epoch)
    if len(value.shape) == 1:  # is a list
        string_form = ",".join([str(number) for number in value])
    else:
        string_form = "\n".join(
            [
                ",".join([str(number) for number in value[row]])
                for row in range(value.shape[0])
            ]
        )

    with open("./model/" + filename, "w") as f:
        f.write(string_form)


def restore_parameters(name, epoch):
    filename = name.replace(":", "-colon-") + "-epoch-{}.txt".format(epoch)
    with open("./model/" + filename) as f:
        lines = f.readlines()
    if len(lines) == 1:  # is a vector
        value = [float(number) for number in lines[0].split(",")]
    else:  # is a matrix
        value = [
            [float(number) for number in lines[row].split(",")]
            for row in range(len(lines))
        ]
    return value


def train():
    def load_dataset():
        train_reader = DataReader(
            data_path="./20news-bydate/train_tf_idf.txt",
            batch_size=50,
            vocab_size=vocab_size,
        )
        return train_reader

    # create a computational graph
    with open("./20news-bydate/words_idfs.txt") as f:
        vocab_size = len(f.readlines())
    train_loss = []
    steps = []

    mlp = MLP(vocab_size=vocab_size, hidden_size=50)
    predicted_labels, loss = mlp.build_graph()
    train_op = mlp.trainer(loss=loss, learning_rate=0.01)

    # open a session to run
    with tf.Session() as sess:
        train_data_reader = load_dataset()
        step, MAX_STEP = 0, 5000

        sess.run(tf.global_variables_initializer())
        while step < MAX_STEP:
            train_data, train_labels = train_data_reader.next_batch()
            plabels_eval, loss_eval, _ = sess.run(
                [predicted_labels, loss, train_op],
                feed_dict={mlp.X: train_data, mlp.real_Y: train_labels},
            )
            step += 1
            train_loss.append(loss_eval)
            steps.append(step)
            print("Step: {}, loss: {}".format(step, loss_eval))
            if loss_eval < 1e-5:
                break
        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            save_parameters(
                name=variable.name,
                value=variable.eval(),
                epoch=train_data_reader.num_epoch,
            )

        plt.plot(steps[:1000], train_loss[:1000], "g")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()


def test():
    with open("./20news-bydate/words_idfs.txt") as f:
        vocab_size = len(f.readlines())

    test_data_reader = DataReader(
        data_path="./20news-bydate/test_tf_idf.txt",
        batch_size=50,
        vocab_size=vocab_size,
    )
    mlp = MLP(vocab_size=vocab_size, hidden_size=50)
    predicted_labels, loss = mlp.build_graph()
    with tf.Session() as sess:
        epoch = 44

        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            saved_value = restore_parameters(variable.name, epoch)
            assign_op = variable.assign(saved_value)
            sess.run(assign_op)
        num_true_preds = 0
        while True:
            test_data, test_labels = test_data_reader.next_batch()
            test_plabels_eval = sess.run(
                predicted_labels, feed_dict={mlp.X: test_data, mlp.real_Y: test_labels}
            )
            matches = np.equal(test_plabels_eval, test_labels)
            num_true_preds += np.sum(matches.astype(float))

            if test_data_reader.batch_id == 0:
                break
        print("Epoch: ", epoch)
        print("Accuracy on test data: ", num_true_preds / len(test_data_reader.data))


train()
tf.reset_default_graph()
test()
