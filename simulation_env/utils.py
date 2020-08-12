import _pickle as cPickle
import os


def dump(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    cPickle.dump(data, open(path, 'wb'))


def reload_data(path):
    with open(path, 'rb') as f:
        return cPickle.load(f, encoding="bytes")


if __name__ == "__main__":
    print(1)