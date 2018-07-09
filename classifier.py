import numpy as np
import pywt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

from utils import Utils


class Classifier(object):
    cols_names = ['D feed flow (stream 2)',
                  'E feed flow (stream 3)',
                  'A feed flow (stream 1)',
                  'A and C feed flow (stream 4)',
                  'Compressor recycle valve',
                  'Purge valve (stream 9)',
                  'Separator pot liquid flow (stream 10)',
                  'Stripper liquid product flow (stream 11)',
                  'Stripper steam valve',
                  'Reactor cooling water flow',
                  'Condenser cooling water flow',
                  'Input Feed - A feed (stream 1)',
                  'Input Feed - D feed (stream 2)',
                  'Input Feed - E feed (stream 3)',
                  'Input Feed - A and C feed (stream 4)',
                  'Reactor feed rate (stream 6)',
                  'Reactor pressure',
                  'Reactor level',
                  'Reactor temperature',
                  'Separator - Product separator temperature',
                  'Separator - Product separator level',
                  'Separator - Product separator pressure',
                  'Separator - Product separator underflow (stream 10)',
                  'Stripper level',
                  'Stripper pressure',
                  'Stripper underflow (stream 11)',
                  'Stripper temperature',
                  'Stripper steam flow',
                  'Miscellaneous - Recycle flow (stream 8)',
                  'Miscellaneous - Purge rate (stream 9)',
                  'Miscellaneous - Compressor work',
                  'Miscellaneous - Reactor cooling water outlet temperature',
                  'Miscellaneous - Separator cooling water outlet temperature',
                  'Reactor Feed Analysis - Component A',
                  'Reactor Feed Analysis - Component B',
                  'Reactor Feed Analysis - Component C',
                  'Reactor Feed Analysis - Component D',
                  'Reactor Feed Analysis - Component E',
                  'Reactor Feed Analysis - Component F',
                  'Purge gas analysis - Component A',
                  'Purge gas analysis - Component B',
                  'Purge gas analysis - Component C',
                  'Purge gas analysis - Component D',
                  'Purge gas analysis - Component E',
                  'Purge gas analysis - Component F',
                  'Purge gas analysis - Component G',
                  'Purge gas analysis - Component H',
                  'Product analysis -  Component D',
                  'Product analysis - Component E',
                  'Product analysis - Component F',
                  'Product analysis - Component G',
                  'Product analysis - Component H',
                  'Label']
    model = None
    mlda = None
    projected_data = None

    def load_data(self, config_file):
        raw_data = Utils.read_data(config_file, self.cols_names)
        data = raw_data.ix[:, :-1]
        data = np.array(data)

        labels = raw_data.ix[:, -1].str[1:]
        labels = '0' + labels

        new_labels = []
        for label in labels:
            new_labels.append(list(label))

        new_labels = Utils.to_label(new_labels)

        return data, new_labels

    def __remove_noise(self, data):
        denoised_data = []

        for feature in data.T:
            var = np.var(feature)
            n = float(len(feature))
            threshold = np.sqrt(2 * var * np.log10(n) / n)

            ca, cd = pywt.dwt(feature, 'db1')

            for i in range(len(ca)):
                if np.abs(ca[i]) >= threshold:
                    ca[i] = np.sign(ca[i]) * (np.absolute(ca[i]) - threshold)
                else:
                    ca[i] = 0

            for i in range(len(cd)):
                if np.abs(cd[i]) >= threshold:
                    cd[i] = np.sign(cd[i]) * (np.absolute(cd[i]) - threshold)
                else:
                    cd[i] = 0

            new_data = pywt.idwt(ca, cd, 'db1')
            denoised_data.append(new_data)

        denoised_data = np.array(denoised_data).T
        denoised_data = np.delete(denoised_data, -1, axis=0)

        return denoised_data

    def __mlda(self, data, labels):
        clf = LinearDiscriminantAnalysis()
        clf.fit(data, labels)
        self.mlda = clf
        self.projected_data = clf.transform(data)

        return self.projected_data

    def __svm(self, data, labels):
        clf = svm.SVC()
        clf.fit(data, labels)
        return clf

    def __process_data(self, data):
        return self.mlda.transform(data)

    def __precision(self, true_cls, pred_cls):
        test_result = []

        for idx, val in enumerate(true_cls):
            test_result.append(1) if val == pred_cls[idx] else test_result.append(0)

        return np.mean(test_result)

    def get_data(self):
        return self.projected_data

    def train(self, data, labels):
        denoised_data = self.__remove_noise(data)
        project_data = self.__mlda(denoised_data, labels)
        self.model = self.__svm(project_data, labels)
        return None

    def test(self, data, labels):
        assert self.model is not None, "Error: Train the model first!"

        test_data = self.__process_data(data)
        predicted_label = self.model.predict(test_data)

        return self.__precision(labels, predicted_label), test_data, predicted_label
