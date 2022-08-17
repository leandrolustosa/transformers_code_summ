import spacy
import numpy as np
import sys

from src.utils.utils import read_corpus_csv
from src.preprocessing.code_parser import process_code_java, process_code_python
from tqdm import tqdm


def generate_data_stats(codes, descs, process_code_, nlp_):
    codes_sizes = []
    descs_sizes = []
    with tqdm(total=len(codes), file=sys.stdout,
              desc='    Generating Statistics') as pbar:
        for code, desc in zip(codes, descs):
            tokens = process_code_(code).split(' ')
            codes_sizes.append(len(tokens))
            doc = nlp_(desc)
            words = [t.text for t in doc]
            descs_sizes.append(len(words))
            pbar.update(1)
    print('\n\n    Codes')
    print('      Min:', min(codes_sizes))
    print('      Max:', max(codes_sizes))
    print('      Mean:', np.mean(codes_sizes), '~', np.std(codes_sizes))
    print('\n    Descriptions')
    print('      Min:', min(descs_sizes))
    print('      Max:', max(descs_sizes))
    print('      Mean:', np.mean(descs_sizes), '~', np.std(descs_sizes))


if __name__ == '__main__':

    lang = 'java'
    # lang = 'python'

    # corpus_name = 'huetal'
    corpus_name = 'codexglue'
    # corpus_name = 'wanetal'

    preproc_config_name = 'none'
    # preproc_config_name = 'camelsnakecase'

    train_file_path = f'../../resources/corpora/{lang}/{corpus_name}/csv/train_{preproc_config_name}.csv'
    valid_file_path = f'../../resources/corpora/{lang}/{corpus_name}/csv/valid_{preproc_config_name}.csv'
    test_file_path = f'../../resources/corpora/{lang}/{corpus_name}/csv/test_{preproc_config_name}.csv'

    train_data, valid_data, test_data = read_corpus_csv(
        train_file_path, valid_file_path, test_file_path, sample_size=-1)

    process_code = None

    if lang == 'java':
        process_code = process_code_java
    elif lang == 'python':
        process_code = process_code_python
    else:
        print('\nLanguage Invalid!')
        exit()

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

    print('\nCorpus:', corpus_name, '-', preproc_config_name)

    train_codes = train_data[0]
    train_descs = train_data[1]

    valid_codes = valid_data[0]
    valid_descs = valid_data[1]

    test_codes = test_data[0]
    test_descs = test_data[1]

    print('\n  Test set:', len(test_codes), '-', len(test_descs), '\n')

    generate_data_stats(test_codes, test_descs, process_code, nlp)

    print('\n  Valid set:', len(valid_codes), '-', len(valid_descs), '\n')

    generate_data_stats(valid_codes, valid_descs, process_code, nlp)

    print('\n  Train set:', len(train_codes), '-', len(train_descs), '\n')

    generate_data_stats(train_codes, train_descs, process_code, nlp)
