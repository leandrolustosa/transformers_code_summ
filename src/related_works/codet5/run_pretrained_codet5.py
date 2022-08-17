import torch
import os
import sys
import time
import numpy as np

from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from src.utils import utils


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

    num_beams = 5

    generated_desc_dir = f'../../../resources/related_works/descriptions/{lang}/{corpus_name}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nDevice:', device)

    _, _, test_data = utils.read_corpus_csv(test_file_path=test_file_path, sample_size=size_threshold)

    test_codes = test_data[0]

    os.makedirs(generated_desc_dir, exist_ok=True)

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

    model = model.to(device)

    print('\nCorpus:', corpus_name, '-', preproc_config_name)

    print('\n  Test set:', len(test_codes))
    print('    Code:', test_codes[0], '\n')

    total_examples = len(test_codes)

    print('\nGenerating descriptions\n')

    generated_descriptions = []

    execution_times = []

    with tqdm(total=total_examples, file=sys.stdout, desc='  Generating summaries') as pbar:

        for code in test_codes:

            start = time.time()

            input_ids = tokenizer.encode(code, return_tensors='pt', max_length=max_code_len, truncation=True)

            input_ids = input_ids.to(device)

            desc_ids = model.generate(input_ids=input_ids, bos_token_id=model.config.bos_token_id,
                                      eos_token_id=model.config.eos_token_id, length_penalty=2.0,
                                      max_length=max_desc_len, num_beams=num_beams)

            desc = tokenizer.decode(desc_ids[0], skip_special_tokens=True)

            end = time.time()

            execution_time = end - start

            execution_times.append(execution_time)

            generated_descriptions.append(desc)

            pbar.update(1)

    generated_desc_file = os.path.join(generated_desc_dir, f'pre_codet5_{corpus_name}_{preproc_config_name}.txt')

    with open(generated_desc_file, 'w') as file:
        file.write('\n'.join(generated_descriptions))

    execution_stats = f'time: {np.mean(execution_times)} -- std: {np.std(execution_times)}'

    execution_stats_file = os.path.join(generated_desc_dir, f'pre_codet5_{corpus_name}_{preproc_config_name}_time.txt')

    with open(execution_stats_file, 'w') as file:
        file.write(execution_stats)
