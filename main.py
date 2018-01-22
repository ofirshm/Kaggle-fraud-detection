import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,f1_score
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
import time


def show_confusion_matrix(C, class_labels=['0', '1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert C.shape == (2, 2), "Confusion matrix should be from binary classification only."

    # true negative, false positive, etc...
    tn = C[0, 0];
    fp = C[0, 1];
    fn = C[1, 0];
    tp = C[1, 1];

    NP = fn + tp  # Num positive examples
    NN = tn + fp  # Num negative examples
    N = NP + NN

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.plot([-0.5, 2.5], [0.5, 0.5], '-k', lw=2)
    ax.plot([-0.5, 2.5], [1.5, 1.5], '-k', lw=2)
    ax.plot([0.5, 0.5], [-0.5, 2.5], '-k', lw=2)
    ax.plot([1.5, 1.5], [-0.5, 2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34, 1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''], rotation=90)
    ax.set_yticks([0, 1, 2])
    ax.yaxis.set_label_coords(-0.09, 0.65)

    # Fill in initial metrics: tp, tn, etc...
    ax.text(0, 0,
            'True Neg: %d\n(Num Neg: %d)' % (tn, NN),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 1,
            'False Neg: %d' % fn,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 0,
            'False Pos: %d' % fp,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 1,
            'True Pos: %d\n(Num Pos: %d)' % (tp, NP),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2, 0,
            'False Pos Rate: %.2f' % (fp / (fp + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 1,
            'True Pos Rate: %.2f' % (tp / (tp + fn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 2,
            'Accuracy: %.2f' % ((tp + tn + 0.) / N),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 2,
            'Neg Pre Val: %.2f' % (1 - fn / (fn + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 2,
            'Pos Pred Val: %.2f' % (tp / (tp + fp + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    plt.tight_layout()
    plt.show()
def explore_features(feature_vector):


    plt.figure(figsize=(12, 28 * 4))
   # gs = gridspec.GridSpec(28, 1)
    for i, cn in enumerate(df[v_features]):
        ax = plt.figure(i)
        sns.distplot(df[cn][df.Class == 1], bins=50)
        sns.distplot(df[cn][df.Class == 0], bins=50)
        #ax.set_xlabel('')
        plt.title('histogram of feature: ' + str(cn))
        plt.show()
        #Credit :http://notmatthancock.github.io/2015/10/28/confusion-matrix.html
def PlotHistogram(df, norm,title):
    bins = np.arange(df['hour'].min(), df['hour'].max() + 2)
    plt.figure(figsize=(15, 4))
    plt.title(title)
    sns.distplot(df[df['Class'] == 0.0]['hour'],
                 norm_hist=norm,
                 bins=bins,
                 kde=False,
                 color='b',
                 hist_kws={'alpha': .5},
                 label='Legit')
    sns.distplot(df[df['Class'] == 1.0]['hour'],
                 norm_hist=norm,
                 bins=bins,
                 kde=False,
                 color='r',
                 label='Fraud',
                 hist_kws={'alpha': .5})
    plt.xticks(range(0, 24))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Data loading
    df = pd.read_csv("creditcard.csv")
    '''
    Anonymized credit card transactions labeled as fraudulent or genuine
    FILE Information:
    Amount - Transaction Amount
    Class - The actual classification classes 
              0  Legit transactions;  1  Fruadulent transactions
    Time - the seconds elapsed between each transaction and the first transaction in the dataset.
    V1-v28 PCA componenets
    '''
    data_exploration=True
    vis=False
    # Data exploration
    if data_exploration:

        #From the plot,
        # Check for errors in DB:
        # 1. missing values
        df.isnull().sum()
        print("Fraud transactions description")
        print(df.Time[df.Class == 1].describe())
        print('')
        print("Normal transactions description")
        print(df.Time[df.Class == 0].describe())
        #Create new colum for hour for more logical transaction quantization for both classes
        df['hour'] = df['Time'].apply(lambda x: np.ceil(float(x) / 3600) % 24)
        df.pivot_table(values='Amount', index='hour', columns='Class', aggfunc='count')
        if vis:
            start = time.time()
            print()
            PlotHistogram(df, True,title='Normalized histogram of Legit/Fraud over hour of the day')
            print(time.time() - start)

            v_features = df.ix[:, 1:29].columns
            #Exploring features
            start = time.time()
            explore_features(v_features)
            print(time.time() - start)

        print('Fraud is {}% of our data.'.format(df['Class'].value_counts()[1] / float(df['Class'].value_counts()[0]) * 100))

        '''
        feature engineering 
            Using the feature maps we plotted, 
            we should drop the features that have similar distributions
        '''

    df = df.drop(['V28', 'V27', 'V26', 'V25', 'V24', 'V23', 'V22', 'V20', 'V15', 'V13', 'V8'], axis=1)
    # Create a new feature for normal (non-fraudulent) transactions.

    df.loc[df.Class == 0, 'Normal'] = 1
    df.loc[df.Class == 1, 'Normal'] = 0
    # Rename 'Class' to 'Fraud'.
    df = df.rename(columns={'Class': 'Fraud'})

    # Create dataframes of only Fraud and Normal transactions.
    Fraud = df[df.Fraud == 1]
    Normal = df[df.Normal == 1]
    # Set X_train equal to 80% of the fraudulent transactions.
    X_train = Fraud.sample(frac=0.8)
    count_Frauds = len(X_train)

    # Add 80% of the normal transactions to X_train.
    X_train = pd.concat([X_train, Normal.sample(frac=0.8)], axis=0)

    # X_test contains all the transaction not in X_train.
    X_test = df.loc[~df.index.isin(X_train.index)]
    # Shuffle the dataframes so that the training is done in a random order.
    X_train = shuffle(X_train)
    X_test = shuffle(X_test)
    # Add our target features to y_train and y_test.
    y_train = X_train.Fraud
    y_train = pd.concat([y_train, X_train.Normal], axis=1)

    y_test = X_test.Fraud
    y_test = pd.concat([y_test, X_test.Normal], axis=1)
    # Drop target features from X_train and X_test.
    X_train = X_train.drop(['Fraud', 'Normal'], axis=1)
    X_test = X_test.drop(['Fraud', 'Normal'], axis=1)
    # Check to ensure all of the training/testing dataframes are of the correct length
    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))
    '''
    Due to the imbalance in the data, ratio will act as an equal weighting system for our model. 
    By dividing the number of transactions by those that are fraudulent, ratio will equal the value that when multiplied
    by the number of fraudulent transactions will equal the number of normal transaction. 
    Simply put: # of fraud * ratio = # of normal
    '''
    ratio = len(X_train) / count_Frauds
    y_train.Fraud *= ratio
    y_test.Fraud *= ratio

    # Names of all of the features in X_train.
    features = X_train.columns.values
    # Transform each feature in features so that it has a mean of 0 and standard deviation of 1;
    # this helps with training the neural network.
    for feature in features:
        mean, std = df[feature].mean(), df[feature].std()
        X_train.loc[:, feature] = (X_train[feature] - mean) / std
        X_test.loc[:, feature] = (X_test[feature] - mean) / std


    #Train the Neural Net
    # Split the testing data into validation and testing sets
    split = int(len(y_test) / 2)

    inputX = X_train.as_matrix()
    inputY = y_train.as_matrix()
    inputX_valid = X_test.as_matrix()[:split]
    inputY_valid = y_test.as_matrix()[:split]
    inputX_test = X_test.as_matrix()[split:]
    inputY_test = y_test.as_matrix()[split:]

    # Number of input nodes.
    input_nodes = 20

    # Multiplier maintains a fixed ratio of nodes between each layer.
    mulitplier = 1.5

    # Number of nodes in each hidden layer
    hidden_nodes1 = 18
    hidden_nodes2 = round(hidden_nodes1 * mulitplier)
    hidden_nodes3 = round(hidden_nodes2 * mulitplier)

    # Percent of nodes to keep during dropout.
    pkeep = tf.placeholder(tf.float32)
    # input
    x = tf.placeholder(tf.float32, [None, input_nodes])

    # layer 1
    W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev=0.15))
    b1 = tf.Variable(tf.zeros([hidden_nodes1]))
    y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    # layer 2
    W2 = tf.Variable(tf.truncated_normal([hidden_nodes1, hidden_nodes2], stddev=0.15))
    b2 = tf.Variable(tf.zeros([hidden_nodes2]))
    y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

    # layer 3
    W3 = tf.Variable(tf.truncated_normal([hidden_nodes2, hidden_nodes3], stddev=0.15))
    b3 = tf.Variable(tf.zeros([hidden_nodes3]))
    y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)
    y3 = tf.nn.dropout(y3, pkeep)

    # layer 4
    W4 = tf.Variable(tf.truncated_normal([hidden_nodes3, 2], stddev=0.15))
    b4 = tf.Variable(tf.zeros([2]))
    y4 = tf.nn.softmax(tf.matmul(y3, W4) + b4)

    # output
    y = y4
    y_ = tf.placeholder(tf.float32, [None, 2])

    # Parameters
    training_epochs = 2000
    training_dropout = 0.9
    display_step = 10
    n_samples = y_train.shape[0]
    batch_size = 1024
    learning_rate = 0.005

    # Cost function: Cross Entropy
    cost = -tf.reduce_sum(y_ * tf.log(y))

    # We will optimize our model via AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Correct prediction if the most likely value (Fraud or Normal) from softmax equals the target value.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy_summary = []  # Record accuracy values for plot
    cost_summary = []  # Record cost values for plot
    valid_accuracy_summary = []
    valid_cost_summary = []
    stop_early = 0  # To keep track of the number of epochs before early stopping

    # Save the best weights so that they can be used to make the final predictions
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    checkpoint = os.path.join(ROOT_DIR, "best_model.ckpt")
    saver = tf.train.Saver(max_to_keep=1)

    # Initialize variables and tensorflow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            for batch in range(int(n_samples / batch_size)):
                batch_x = inputX[batch * batch_size: (1 + batch) * batch_size]
                batch_y = inputY[batch * batch_size: (1 + batch) * batch_size]

                sess.run([optimizer], feed_dict={x: batch_x,
                                                 y_: batch_y,
                                                 pkeep: training_dropout})

            # Display logs after every 10 epochs
            if (epoch) % display_step == 0:
                train_accuracy, newCost = sess.run([accuracy, cost], feed_dict={x: inputX,
                                                                                y_: inputY,
                                                                                pkeep: training_dropout})

                valid_accuracy, valid_newCost = sess.run([accuracy, cost], feed_dict={x: inputX_valid,
                                                                                      y_: inputY_valid,
                                                                                      pkeep: 1})

                print("Epoch:", epoch,
                      "Acc =", "{:.5f}".format(train_accuracy),
                      "Cost =", "{:.5f}".format(newCost),
                      "Valid_Acc =", "{:.5f}".format(valid_accuracy),
                      "Valid_Cost = ", "{:.5f}".format(valid_newCost))

                # Save the weights if these conditions are met.
                if epoch > 0 and valid_accuracy > max(valid_accuracy_summary) and valid_accuracy > 0.9:
                    saver.save(sess, checkpoint)

                # Record the results of the model
                accuracy_summary.append(train_accuracy)
                cost_summary.append(newCost)
                valid_accuracy_summary.append(valid_accuracy)
                valid_cost_summary.append(valid_newCost)

                # If the model does not improve after 15 logs, stop the training.
                if valid_accuracy < max(valid_accuracy_summary) and epoch > 100:
                    stop_early += 1
                    if stop_early == 15:
                        break
                else:
                    stop_early = 0

    print()
    print("Optimization Finished!")
    print()
    # Plot the accuracy and cost summaries
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 4))

    ax1.plot(accuracy_summary)  # blue
    ax1.plot(valid_accuracy_summary)  # green
    ax1.set_title('Accuracy')

    ax2.plot(cost_summary)
    ax2.plot(valid_cost_summary)
    ax2.set_title('Cost')

    plt.xlabel('Epochs (x10)')
    plt.show()
    # Find the predicted values, then use them to build a confusion matrix
    predicted = tf.argmax(y, 1)
    with tf.Session() as sess:
    # Load the best weights and show its results
        saver.restore(sess, checkpoint)
        training_accuracy = sess.run(accuracy, feed_dict={x: inputX, y_: inputY, pkeep: training_dropout})
        validation_accuracy = sess.run(accuracy, feed_dict={x: inputX_valid, y_: inputY_valid, pkeep: 1})

    print("Results using the best Valid_Acc:")
    print()
    print("Training Accuracy =", training_accuracy)
    print("Validation Accuracy =", validation_accuracy)


    # Find the predicted values, then use them to build a confusion matrix
    predicted = tf.argmax(y, 1)
    with tf.Session() as sess:
    #    # Load the best weights
        saver.restore(sess, checkpoint)
        testing_predictions, testing_accuracy = sess.run([predicted, accuracy],feed_dict={x: inputX_test, y_:inputY_test, pkeep: 1})
    #
        print("F1-Score =", f1_score(inputY_test[:,1], testing_predictions))
        print("Testing Accuracy =", testing_accuracy)
        print()
    c = confusion_matrix(inputY_test[:,1], testing_predictions)
    show_confusion_matrix(c, ['Fraud', 'Normal'])

