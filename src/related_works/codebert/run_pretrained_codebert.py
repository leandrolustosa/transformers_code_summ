import numpy as np
import torch
import os
import code_bert_utils
import sys
import time

from src.utils import utils
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from tqdm import tqdm

"""
    https://huggingface.co/microsoft/codebert-base
"""

if __name__ == '__main__':

    lang = 'java'
    # lang = 'python'

    corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    # corpus_name = 'wanetal'

    preproc_config_name = 'none'
    # preproc_config_name = 'camelsnakecase'

    test_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/test_{preproc_config_name}.csv'

    size_threshold = 20

    max_code_len = 300
    max_desc_len = 20

    beam_size = 5

    model_file = '../../../resources/related_works/models/codebert/pytorch_model.bin'

    generated_desc_dir = f'../../../resources/related_works/descriptions/{lang}/{corpus_name}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nUsing device:', device)

    _, _, test_data = utils.read_corpus_csv(test_file_path=test_file_path, sample_size=size_threshold)

    test_codes = test_data[0]

    os.makedirs(generated_desc_dir, exist_ok=True)

    model_path = 'microsoft/codebert-base'

    config = RobertaConfig.from_pretrained(model_path)

    tokenizer = RobertaTokenizer.from_pretrained(model_path, do_lower_case=False)

    model = code_bert_utils.build_model(model_class=RobertaModel, model_file=model_file, config=config,
                                        tokenizer=tokenizer, beam_size=beam_size, max_len=max_desc_len,
                                        device=device)

    model = model.to(device)

    print('\nLanguage:', lang)

    print('\nCorpus:', corpus_name, '-', preproc_config_name)

    print('\n  Test set:', len(test_codes))
    print('    Code:', test_codes[0], '\n')

    total_examples = len(test_codes)

    generated_descriptions = []

    execution_times = []

    with tqdm(total=total_examples, file=sys.stdout, desc='  Generating summaries') as pbar:

        for i in range(total_examples):

            code = test_codes[i]

            start = time.time()

            example = [code_bert_utils.Example(source=code, target=None)]

            features_code = code_bert_utils.get_features(example, tokenizer, max_code_len)

            generated_desc, length = code_bert_utils.inference(features_code, model, tokenizer, device=device)

            end = time.time()

            execution_time = end - start

            execution_times.append(execution_time)

            generated_descriptions.append(generated_desc[0])

            pbar.update(1)

    generated_desc_file = os.path.join(generated_desc_dir, f'pre_codebert_{corpus_name}_{preproc_config_name}.txt')

    with open(generated_desc_file, 'w') as file:
        file.write('\n'.join(generated_descriptions))

    execution_stats = f'time: {np.mean(execution_times)} -- std: {np.std(execution_times)}'

    execution_stats_file = os.path.join(generated_desc_dir,
                                        f'pre_codebert_{corpus_name}_{preproc_config_name}_time.txt')

    with open(execution_stats_file, 'w') as file:
        file.write(execution_stats)
