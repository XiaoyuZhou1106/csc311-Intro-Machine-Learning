from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    real_news = open("clean_real.txt", "r")
    fake_news = open("clean_fake.txt", "r")
    data = []
    target = []
    for line in real_news.readlines():
        data.append(line.strip())
        target.append(1)
    for line in fake_news.readlines():
        data.append(line.strip())
        target.append(0)
    cv = CountVectorizer()
    cv_trans = cv.fit_transform(data).toarray()
    names = cv.get_feature_names()
    data_temp, data_training, target_temp, target_training = train_test_split(cv_trans, target, test_size=0.7)
    data_valid, data_test, target_valid, target_test = train_test_split(data_temp, target_temp, test_size=0.5)
    training = [data_training, target_training]
    validation = [data_valid, target_valid]
    test = [data_test, target_test]
    return training, validation, test, names

def select_tree_model(training, validation, depth):
    final = []
    for i in range(7):
        max_depth = depth * (i + 1)
        gini_tree = DecisionTreeClassifier(criterion="gini", max_depth=max_depth)
        gini_tree = gini_tree.fit(training[0], training[1])
        gini_score = gini_tree.score(validation[0], validation[1])
        gain_tree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
        gain_tree = gain_tree.fit(training[0], training[1])
        gain_score = gain_tree.score(validation[0], validation[1])
        print([gini_score, gain_score])
        result = [max_depth, gini_tree, gain_tree, gini_score, gain_score]
        final.append(result)
    return final

def select_best_tree(temp, test):
    score = temp[0][3]
    tree = temp[0][1]
    for i in range(len(temp)):
        if temp[i][3] >= temp[i][4]:
            if temp[i][3] > score:
                score = temp[i][3]
                tree = temp[i][1]
        else:
            if temp[i][4] > score:
                score = temp[i][4]
                tree = temp[i][2]
    test_score = tree.score(test[0], test[1])
    print(test_score)
    return tree, score

def compute_information_gain(training, names, keyword):
    #Calculate the real or fake news in the whole training set.
    real_num_total = 0
    fake_num_total = 0
    for i in range(len(training[1])):
        if training[1][i] == 0:
            fake_num_total += 1
        else:
            real_num_total += 1
    total_num = real_num_total + fake_num_total

    #Get the index of the keyword in the name_features.
    key_index = names.index(keyword)
    left_real = 0
    left_fake = 0
    right_real = 0
    right_fake = 0
    for j in range(len(training[0])):
        if training[0][j][key_index] != 0:
            if training[1][j] == 0:
                left_fake += 1
            else:
                left_real += 1
        else:
            if training[1][j] == 0:
                right_fake += 1
            else:
                right_real += 1
    right_total = right_fake + right_real
    left_total = left_real + left_fake

    p_real_left = left_real / left_total
    p_fake_left = left_fake / left_total
    p_real_right = right_real / right_total
    p_fake_right = right_fake / right_total
    p_real = real_num_total / total_num
    p_fake = fake_num_total / total_num
    p_left = left_total / total_num
    p_right = right_total / total_num

    final = -p_real*(np.log2(p_real)) -p_fake*(np.log2(p_fake)) + \
            (p_real_left*(np.log2(p_real_left)) + p_fake_left*(np.log2(p_fake_left)))*p_left + \
            (p_real_right * (np.log2(p_real_right)) + p_fake_right*(np.log2(p_fake_right)))*p_right

    return final


def select_knn_model(training, validation, test):
    k_val = range(1, 21)
    training_error = []
    valid_error = []
    knn_lst = []
    for i in k_val:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(training[0], training[1])
        train_pred = knn.predict(training[0])
        tra_score = metrics.accuracy_score(training[1], train_pred)
        training_error.append(1 - tra_score)
        valid_pred = knn.predict(validation[0])
        val_score = metrics.accuracy_score(validation[1], valid_pred)
        valid_error.append(1 - val_score)
        knn_lst.append([i, knn, tra_score, val_score])

    plt.figure()
    plt.plot(k_val, training_error, marker = "o", label="Train")
    plt.plot(k_val, valid_error, marker = "o", label="Validation")
    plt.xlabel("k - Number of Nearest Neighbours")
    plt.ylabel("TestError")
    plt.show()

    best_score = knn_lst[0][3]
    best_knn = knn_lst[0][1]
    for j in range(len(knn_lst)):
        if knn_lst[j][3] > best_score:
            best_knn = knn_lst[j][1]
    test_pred = best_knn.predict(test[0])
    test_score = metrics.accuracy_score(test[1], test_pred)
    return best_knn, best_score, test_score




if __name__ == "__main__":
    training, validation, test, names = load_data()
    # print(training, validation, test)
    # trees = select_tree_model(training, validation, 5)
    # print(trees)
    # best_tree, best_score = select_best_tree(trees, test)
    # print(best_tree, best_score)
    # dot_data = tree.export_graphviz(best_tree, out_file="best_tree.dot", rounded=True, proportion=False, \
    #                                 feature_names=names, max_depth=3)
    num = compute_information_gain(training, names, "is")
    print(num)
    # knn, knn_score, knn_test_score = select_knn_model(training, validation, test)
    # print(knn_test_score)
