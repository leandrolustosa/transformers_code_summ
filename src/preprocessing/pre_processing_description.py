import numpy as np
import os
import sys

from src.utils import utils
from src.evaluation.evaluation_measures import compute_rouge, \
    compute_bleu, compute_meteor
from tqdm import tqdm
from src.evaluation.bleu import compute_maps, bleu_from_maps


if __name__ == '__main__':

    lang = 'java'
    # lang = 'python'

    # corpus_name = 'huetal'
    corpus_name = 'codexglue'
    # corpus_name = 'wanetal'

    size_threshold = -1

    max_desc_len = 30

    test_file_path = f'../../resources/corpora/{lang}/{corpus_name}/csv/test_none_none.csv'
    new_test_file_path = f'../../resources/corpora/{lang}/{corpus_name}/csv/new_test_none_none.csv'

    _, _, test_data = utils.read_corpus_csv(test_file_path=test_file_path,
                                            sample_size=size_threshold)

    test_descs = test_data[1]
    for i, test_desc in enumerate(test_descs):
        test_descs[i] = test_desc.split('@')[0]
        test_descs[i] = test_descs[i].split('. ')[0]
        if test_descs[i][-1] != '.':
            test_descs[i] += '.'


    with open(new_test_file_path, 'w') as file:
        file.write('\n'.join(test_descs))