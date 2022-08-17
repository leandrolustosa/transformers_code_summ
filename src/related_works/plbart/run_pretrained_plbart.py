import torch
import os
import sys
import time
import numpy as np

from transformers import PLBartTokenizer, PLBartForConditionalGeneration
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

    generated_desc_dir = f'../../../resources/related_works/descriptions/{lang}/{corpus_name}'

    size_threshold = 20

    max_code_len = 300
    max_desc_len = 20

    num_beams = 5

    model_name = None

    if lang == 'java':
        model_name = 'uclanlp/plbart-java-en_XX'
    elif lang == 'python':
        model_name = 'uclanlp/plbart-python-en_XX'
    else:
        print('\nError lang')
        exit(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nDevice:', device)

    _, _, test_data = utils.read_corpus_csv(test_file_path=test_file_path, sample_size=size_threshold)

    test_codes = test_data[0]

    os.makedirs(generated_desc_dir, exist_ok=True)

    tokenizer = PLBartTokenizer.from_pretrained(model_name, src_lang=lang, tgt_lang='en_XX')

    model = PLBartForConditionalGeneration.from_pretrained(model_name)

    model = model.to(device)

    print('\nLanguage:', lang)

    print('\nCorpus:', corpus_name, '-', preproc_config_name)

    print('\n  Test set:', len(test_codes))
    print('    Code:', test_codes[0], '\n')

    total_examples = len(test_codes)

    print('\nGenerating descriptions\n')

    generated_descriptions = []

    execution_times = []

    start_token_id = tokenizer.lang_code_to_id['en_XX']

    with tqdm(total=total_examples, file=sys.stdout, desc='  Generating summaries') as pbar:

        for code in test_codes:

            start = time.time()

            input_ids = tokenizer(code, return_tensors='pt', max_length=max_code_len, truncation=True)

            input_ids = input_ids.to(device)

            translated_tokens = model.generate(**input_ids, max_length=max_desc_len, num_beams=num_beams,
                                               decoder_start_token_id=start_token_id)

            desc = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

            end = time.time()

            execution_time = end - start

            execution_times.append(execution_time)

            generated_descriptions.append(desc)

            pbar.update(1)

    generated_desc_file = os.path.join(generated_desc_dir, f'pre_plbart_{corpus_name}_{preproc_config_name}.txt')

    with open(generated_desc_file, 'w') as file:
        file.write('\n'.join(generated_descriptions))

    execution_stats = f'time: {np.mean(execution_times)} -- std: {np.std(execution_times)}'

    execution_stats_file = os.path.join(generated_desc_dir, f'pre_plbart_{corpus_name}_{preproc_config_name}_time.txt')

    with open(execution_stats_file, 'w') as file:
        file.write(execution_stats)
