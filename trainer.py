import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from config.config import data_path, data_vectors_path
from sklearn.model_selection import train_test_split
from libs.models import get_model1


class Trainer(object):
    def __init__(self):
        self.data_dir = './data/'
        self.data_vectors_dir = './data_vectors/'
        self.labels = os.listdir(self.data_dir)
        self.one_hot_encoder = OneHotEncoder(sparse=False)

    def load_process_data(self):
        x_list = []
        y_list = []

        for i, label in enumerate(self.labels):
            x = np.load(self.data_vectors_dir + label + '.npy')
            x_list.append(x)
            y = np.full(x.shape[0], fill_value=(i))
            y_list.append(y)
        X = np.expand_dims(np.vstack(x_list), axis=3)
        y = self.one_hot_encoder.fit_transform(np.hstack(y_list).reshape(-1,1), )

        assert X.shape[0] == len(y), "len of X incompatible with len of y"
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, shuffle=True)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.load_process_data()
    model = get_model1(in_shape=tuple(trainer.x_train.shape[1:]),
                       num_classes=trainer.y_train.shape.__len__())
    model.fit(trainer.x_train,
              trainer.y_train,
              batch_size=16,
              epochs=200,
              validation_split=0.2)
    y_predict = np.argmax(model.predict(trainer.x_test), axis=1)
    y_true = np.argmax(trainer.y_test, axis=1)

    print(classification_report(y_true, y_predict))

