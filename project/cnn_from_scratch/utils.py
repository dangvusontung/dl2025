import tensorflow as tf
import numpy as np # I'M JUST USING IT TO CREATE DATA FOR TESTING YOU CAN SEE NO NUMPY IN THE CODE EXECEPT FOR THIS FUNCTION

def load_mnist_data(num_samples, kind='train'):
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    if kind == 'train':
        images_np = x_train[:num_samples]
        labels_np = y_train[:num_samples]
    elif kind == 'test':
        images_np = x_test[:num_samples]
        labels_np = y_test[:num_samples]
    else:
        raise ValueError("unknow kind")

    images_np = np.expand_dims(images_np, axis=1)
    images_np = images_np.astype('float32') / 255.0
    
    num_actual_samples = images_np.shape[0]
    labels_one_hot_np = np.zeros((num_actual_samples, 10))
    labels_one_hot_np[np.arange(num_actual_samples), labels_np] = 1
    
    images_list = images_np.tolist()
    labels_list = labels_one_hot_np.tolist()
    
    return list(zip(images_list, labels_list))