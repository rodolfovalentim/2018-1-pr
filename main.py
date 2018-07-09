from classifier import Classifier
import matplotlib.pyplot as plt

if __name__ == '__main__':

    for i in range(1, 19):
        classifier = Classifier()
        train_data, train_labels = classifier.load_data('/home/rodolfo/Documents/te/data/fault_0' + str(i) + '/train'
                                                                                                             '.csv')
        test_data, test_labels = classifier.load_data('/home/rodolfo/Documents/te/data/fault_0' + str(i) + '/test.csv')

        classifier.train(train_data, train_labels)
        precision, data, labels_hat = classifier.test(test_data, test_labels)
        project_train_data = classifier.get_data()

        for idx, val in enumerate(project_train_data):
            color = 'red'
            if train_labels[idx] != 0:
                if len(val) == 2:
                    plt.scatter(val[0], val[1], c='green')
                else:
                    plt.scatter(2.5, val, c='green')
            else:
                if len(val) == 2:
                    plt.scatter(val[0], val[1], c='yellow')
                else:
                    plt.scatter(2, val, c='yellow')

        for idx, val in enumerate(data):
            color = 'red'
            if labels_hat[idx] != 0:
                if len(val) == 2:
                    plt.scatter(val[0], val[1], c='blue')
                else:
                    plt.scatter(1.5, val, c='blue')
            else:
                if len(val) == 2:
                    plt.scatter(val[0], val[1], c='red')
                else:
                    plt.scatter(1, val, c='red')

        plt.show()

        print('--> Fault {} | Train: {} samples | Train: {} samples | Precision {} %'.format(i,
                                                                                             len(train_labels),
                                                                                             len(test_labels),
                                                                                             precision*100))
