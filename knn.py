import random
import math

def main():
    dataset = get_data_set()  # get data set
    euclid_test = [0, 0, 0, 0, 0]  # accuracy for test euclidean distance for k=1 index 0,k = 3 index 1 and so on
    manhat_test = [0, 0, 0, 0, 0]
    freche_test = [0, 0, 0, 0, 0]
    euclid_train = [0, 0, 0, 0, 0]  # accuracy for train euclidean distance
    manhat_train = [0, 0, 0, 0, 0]
    freche_train = [0, 0, 0, 0, 0]
    for i in range(0, 500):
        random.shuffle(dataset)  # shuffle the dataset
        trainset = dataset[:65]  # train is first 65 points
        testset = dataset[65:130]  # test is the last 65 points
        for k in range(1, 10, 2):
            sum_euclid_test = 0  # counter of euclidean distance accuracy in test set for current k
            sum_manh_test = 0
            sum_frech_test = 0
            sum_euclid_train = 0  # counter of euclidean distance accuracy in train set for current k
            sum_manh_train = 0
            sum_frech_train = 0
            # run knn algho on test set
            sum_euclid_test, sum_frech_test, sum_manh_test = knn_classify(k, sum_euclid_test, sum_frech_test,
                                                                          sum_manh_test, testset, trainset)
            # run knn algho on train set
            sum_euclid_train, sum_frech_train, sum_manh_train = knn_classify(k, sum_euclid_train, sum_frech_train,
                                                                             sum_manh_train, trainset, trainset)
            # sum the results of train and test
            calculate_train_acc(euclid_train, freche_train, k, manhat_train, sum_euclid_train, sum_frech_train, sum_manh_train)
            calculate_test_acc(euclid_test, freche_test, k, manhat_test, sum_euclid_test,sum_frech_test, sum_manh_test)
    print_results(euclid_test, euclid_train, freche_test, freche_train, manhat_test, manhat_train)


# sum the result of the iteration of the test
def calculate_test_acc(euclid_test, freche_test, k, manhat_test, sum_euclid_test, sum_frech_test, sum_manh_test):
    euclid_test[int(k / 2)] += sum_euclid_test / 65
    manhat_test[int(k / 2)] += sum_manh_test / 65
    freche_test[int(k / 2)] += sum_frech_test / 65


# sum the result of the iteration on the train
def calculate_train_acc(euclid_train, freche_train, k, manhat_train, sum_euclid_train, sum_frech_train, sum_manh_train):
    euclid_train[int(k / 2)] += sum_euclid_train / 65
    manhat_train[int(k / 2)] += sum_manh_train / 65
    freche_train[int(k / 2)] += sum_frech_train / 65


def print_results(euclid_test, euclid_train, freche_test, freche_train, manhat_test, manhat_train):
    print_eucledean_accuracy(euclid_test, euclid_train)
    print_manhatten_accuracy(manhat_test, manhat_train)
    print_freche_accuracy(freche_test, freche_train)


def knn_classify(k, sum_euclid_acc, sum_frech_acc, sum_manh_acc, test, train):
    for point in test:  # for point in test set
        euclidean_neighbors = get_k_naighbors(point, train, k, euclidean_dist)  # get k nearest neighbors with euclid
        manhattan_neighbors = get_k_naighbors(point, train, k, manhattan_distance)
        frechet_neighbors = get_k_naighbors(point, train, k, frechet_distance)
        sum_euclid_acc += classify(point, euclidean_neighbors) # classify the point
        sum_manh_acc += classify(point, manhattan_neighbors)
        sum_frech_acc += classify(point, frechet_neighbors)
    return sum_euclid_acc, sum_frech_acc, sum_manh_acc


def print_freche_accuracy(freche_test_accuracy, freche_train_accuracy):
    print()
    print('----------freche accuracy:')
    k = 1
    for i in range(0, 5):
        print('train accuracy for k = ' + str(k) + ': ' + str(freche_train_accuracy[i] / 500))
        print('test accuracy for  k = ' + str(k) + ': ' + str(freche_test_accuracy[i] / 500))
        print('')
        k += 2


def print_manhatten_accuracy(manhat_test_accuracy, manhat_train_accuracy):
    print()
    print('---------manhattan accuracy:')
    k = 1
    for i in range(0, 5):
        print('train accuracy for k = ' + str(k) + ': ' + str(manhat_train_accuracy[i] / 500))
        print('test accuracy for  k = ' + str(k) + ': ' + str(manhat_test_accuracy[i] / 500))
        print('')
        k += 2


def print_eucledean_accuracy(euclid_test_accuracy, euclid_train_accuracy):
    print('----------eucledean accuracy:--------')
    k = 1
    for i in range(0, 5):
        print('train accuracy for k = ' + str(k) + ': ' + str(euclid_train_accuracy[i] / 500))
        print('test accuracy for  k = ' + str(k) + ': ' + str(euclid_test_accuracy[i] / 500))
        print('')
        k += 2


def get_data_set():
    f = open('HC_Body_Temperature.txt', "r")
    temp_points = list(f.read().splitlines())  # get the points from the file
    points = list()  # list of lists
    get_listof_points(points, temp_points)  # each point in points is a list that contain [x, y, label]
    return points


def get_listof_points(points, temp_points):
    for point in temp_points:
        temp = point.split()  # get the point x y and label without spaces
        x = float(temp[0])  # turn the x to float
        y = int(temp[2])  # turn y to int
        label = int(temp[1])  # turn label to int
        number_point = [x, y, label]
        tuple_point = list(number_point)  # build a tuple of (x, y, label)
        points.append(tuple_point)  # add the tuple to the tuple list


def euclidean_dist(test_point, train_point):  # calculate the euclidean distance between two points
    x_test, x_train = test_point[0], train_point[0]
    y_test, y_train = test_point[1], train_point[1]
    dist = math.sqrt((x_train - x_test)**2 + (y_train - y_test)**2)
    return dist


def get_k_naighbors(test_point, train, k, dist_func):  # this function return the k nearest neghibors
    neighbors = list()  # list of all the neighbors
    for train_point in train:
        dist = dist_func(test_point, train_point)
        neighbors.append((train_point, dist))
    nearest_neighbors = get_k_nearest_neighbors(neighbors, k)  # get the k nearest neighbors
    return nearest_neighbors


def get_k_nearest_neighbors(neighbors, k):  # return the nearest neighbors
    neighbors.sort(key=lambda tup: tup[1])  # sort the neighbors according to their distance from the test point
    nearest_neighbors = list()
    for i in range(0, k):
        nearest_neighbors.append(neighbors[i][0])
    return nearest_neighbors


def manhattan_distance(test_point, train_point):
    x_test, x_train = test_point[0], train_point[0]
    y_test, y_train = test_point[1], train_point[1]
    dist = abs(x_test - x_train) + abs(y_test - y_train)
    return dist


def frechet_distance(test_point, train_point):
    x_test, x_train = test_point[0], train_point[0]
    y_test, y_train = test_point[1], train_point[1]
    dist = max(abs(x_test - x_train), abs(y_test - y_train))
    return dist


def classify(test_point, neighbors):
    sum = 0
    predict = 0
    for point in neighbors:
        sum += point[2]
    if sum > 0:
        predict = 1
    else:
        predict = -1
    if predict == test_point[2]:
        return 1
    else:
        return 0


main()
