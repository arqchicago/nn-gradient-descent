# -*- coding: utf-8 -*-
"""

"""
import tensorflow as tf
import random
import numpy as np
np.random.seed(1234)

# mnist class to load image data set
class mnist:
    def __init__(self):
        # - training_set is a tuple (60000 images, 60000 labels)
        # - testing_set is a tuple (10000 images, 10000 labels)
        self.training_set, self.testing_set = tf.keras.datasets.mnist.load_data(path="mnist.npz")

        #print(f'training set (x): {len(self.training_set[0])}  training set (y): {len(self.training_set[1])}')
        #print(f'testing set (x): {len(self.testing_set[0])}    testing set (y): {len(self.testing_set[1])}')

    def get_raw_datasets(self):
        return self.training_set, self.testing_set
        
    def get_vectorized_datasets(self, validation_data_pct=0.0):
        training_x = [np.reshape(i, (784,1)) for i in self.training_set[0]]
        testing_x = [np.reshape(i, (784,1)) for i in self.testing_set[0]]
        
        if validation_data_pct>0.0:
            random.shuffle(testing_x)
            n_validation_x = int(round(len(testing_x) * validation_data_pct,0))
            n_testing_x = len(testing_x) - n_validation_x
            
            validation_x = testing_x[0:n_validation_x]
            testing_x = testing_x[n_validation_x:]          
            
        training_y = []
        for i in self.training_set[1]:
            y_array = np.zeros((10,1))
            y_array[i] = 1.0
            training_y.append(y_array)
        training_data = list(zip(training_x, training_y))  #python2 returned a list but python3 returns an iterable object
        
        testing_y = []
        for i in self.testing_set[1]:
            y_array = np.zeros((10,1))
            y_array[i] = 1.0
            testing_y.append(y_array)
        
        testing_data = list(zip(testing_x, testing_y))  #python2 returned a list but python3 returns an iterable object        
        
        return training_data, testing_data

'''
if __name__ == "__main__":
    get_val_data = 0.20
    mnist_data = mnist()
    training_data, testing_set = mnist_data.get_vectorized_datasets(get_val_data)
'''