import pickle
from pathlib import Path
import polars as pl
import numpy as np
import cv2
import tensorflow as tf

CIFAR_DIR = Path('data/cifar-10-batches-py')

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_pickled_data(path):
    data_dict = unpickle(path)
    data_dict = {k.decode('utf-8'):data_dict[k] for k in data_dict}
    batch_label = data_dict.pop('batch_label')
    data_df = pl.from_dict(data_dict).with_columns(pl.lit(batch_label.decode('utf-8')).alias('batch_label'))
    return data_df


def get_train_test(cifar_dir: Path):
    all_files = [file for file in cifar_dir.iterdir() if file.is_file()]
    train_paths = [x for x in all_files if 'data_batch' in str(x)]
    test_path  = [x for x in all_files if 'test_batch' in str(x)][0]
    test_df = get_pickled_data(test_path)
    train_df = pl.concat([get_pickled_data(x) for x in train_paths])
    return train_df, test_df

def convert_image(element: list) -> np.array:
    new_data = np.zeros((32, 32, 3), dtype=np.uint8)
    new_data[:,:,2] = np.reshape(element[:1024], (32,32))
    new_data[:,:,1] = np.reshape(element[1024:2048], (32,32))
    new_data[:,:,0] = np.reshape(element[2048:], (32,32))
    return new_data

def convert_image_list_to_nd_array(data_df: pl.DataFrame):
    return data_df.with_columns(pl.col('data').map_elements(convert_image, return_dtype=pl.Object))

def main():
    train, test = get_train_test(CIFAR_DIR)
    train = convert_image_list_to_nd_array(train)
    test = convert_image_list_to_nd_array(test)

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



    cv2.imshow('t', train)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print()


if __name__=="__main__":
    main()

