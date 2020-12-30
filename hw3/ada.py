from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import data
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow.keras as keras

def main():
    #initialize the data and the ada model
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    rng = np.random.RandomState(1)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME", n_estimators=300, random_state=rng, learning_rate=0.8)
    bdt.fit(train_data, train_labels)

    # get the predict result and the predict probablity
    predict_result = bdt.predict(test_data)
    predict_prob = bdt.predict_proba(test_data)
    # translate them into one-hot encoding
    predict = keras.utils.to_categorical(predict_result, 10)
    test = keras.utils.to_categorical(test_labels, 10)

    plt.figure()

    for i in range(10):
        # get roc curve
        plt_roc_curve(plt, predict_prob, test, i)
        # get its average precision and average recall
        precision, recall = cal_precision_recall(predict_prob, test, i)
        print("\nprecision for " + str(i) + ": ", precision)
        print("\nrecall for " + str(i) + ": ", recall)
        # get confusion matrix
        print("\nconfusion matrix for " + str(i), confusion_matrix(test[:, i], predict[:, i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    # get its accuracy
    predict_result = bdt.predict(test_data).tolist()
    print("\naccuracy:", cal_accuracy(predict_result, test_labels))


def plt_roc_curve(plt, predict_prob, test_labels, i):
    fpr, tpr, _ = roc_curve(test_labels[:, i], predict_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, marker='.', label='MLP AUC = %0.2f for %d' % (roc_auc, i))


def cal_precision_recall(predict_prob, test_labels, i):
    precision, recall, _ = precision_recall_curve(test_labels[:, i], predict_prob[:, i])
    return precision.mean(), recall.mean()


def cal_accuracy(predict_result, test_labels):
    sum = 0
    for i in range(len(predict_result)):
        if predict_result[i] == test_labels[i]:
            sum += 1
    accuracy = sum / len(test_labels)
    return accuracy




if __name__ == '__main__':
    main()