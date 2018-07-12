from classifier import Classifier
import numpy as np

if __name__ == '__main__':

    for i in range(1, 19):
        classifier = Classifier()
        data, labels = classifier.load_data('./data/fault_0' + str(i) + '/train.csv')

        acc = classifier.cross_validation(data, labels, k=5)

        print('--> Fault {} |  Accuracy per fold {} | Accuracy {} %'
              .format(i, acc, np.mean(acc)* 100))
