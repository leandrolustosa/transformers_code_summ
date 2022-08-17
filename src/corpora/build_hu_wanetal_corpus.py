import os

from src.utils import utils


if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus = 'wanetal'
    corpus = 'huetal'

    original_path = f'../../resources/corpora/{lang}/{corpus}/original'
    processed_path = f'../../resources/corpora/{lang}/{corpus}/processed'
    splitted_path = f'../../resources/corpora/{lang}/{corpus}/original'

    if not os.path.exists(processed_path):
        os.mkdir(processed_path)

    if not os.path.exists(splitted_path):
        os.mkdir(splitted_path)

    train_portion = 0.7
    valid_portion = 0.5

    print('\nProcessing Python corpus:')

    utils.split_python_dataset(original_path, splitted_path, train_portion, valid_portion)
