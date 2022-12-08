import torch
import os
import sys

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from src.utils.utils import read_corpus_csv


if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_name = 'wanetal'

    # model_name = 'mt_tf_small'
    model_name = 'mt_tf_base'
    # model_name = 'mt_tf_large'

    # eval_measure_opt = 'loss'
    # eval_measure_opt = 'bleu'
    eval_measure_opt = 'rougel'
    # eval_measure_opt = 'meteor'

    preproc_config = 'none'

    if lang == 'java':
        preproc_config = 'camelsnakecase'

    ft_model_path = f'../../../resources/fine_tuning/models/codetrans/{model_name}/{eval_measure_opt}/' \
                    f'{lang}/{corpus_name}/best_model/{model_name}.bin'

    if not os.path.exists(ft_model_path):
        print(f'\nERROR: {ft_model_path} not found')
        exit(-1)

    gen_desc_dir = f'../../../resources/fine_tuning/descriptions/{lang}/{corpus_name}'

    os.makedirs(gen_desc_dir, exist_ok=True)

    desc_file_name = f'codetrans_{model_name}_{preproc_config}_{eval_measure_opt}_ft.txt'

    num_beams = 5

    max_code_len = 300

    min_desc_len = 4
    max_desc_len = 20

    corpus_size_threshold = -1

    test_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/test_{preproc_config}.csv'

    _, _, test_data = read_corpus_csv(test_file_path=test_file_path, sample_size=corpus_size_threshold)

    pre_model_path = None

    if lang == 'python':
        if model_name == 'mt_tf_base':
            pre_model_path = 'SEBIS/code_trans_t5_base_code_documentation_generation_python_multitask_finetune'
        elif model_name == 'mt_tf_small':
            pre_model_path = 'SEBIS/code_trans_t5_small_code_documentation_generation_python_multitask_finetune'
        elif model_name == 'mt_tf_large':
            pre_model_path = 'SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask_finetune'
    elif lang == 'java':
        if model_name == 'mt_tf_base':
            pre_model_path = 'SEBIS/code_trans_t5_base_code_documentation_generation_java_multitask_finetune'
        elif model_name == 'mt_tf_small':
            pre_model_path = 'SEBIS/code_trans_t5_small_code_documentation_generation_java_multitask_finetune'
        elif model_name == 'mt_tf_large':
            pre_model_path = 'SEBIS/code_trans_t5_large_code_documentation_generation_java_multitask_finetune'
    else:
        print(f'ERROR. {model_name} not found.')
        exit(-1)

    test_codes = test_data[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nDevice:', device)

    tokenizer = AutoTokenizer.from_pretrained(pre_model_path, skip_special_tokens=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(pre_model_path)

    model = model.to(device)

    model.load_state_dict(torch.load(ft_model_path))

    print(f'\nCorpus: {lang} - {corpus_name} - {preproc_config} - {eval_measure_opt}')

    print('\nModel:', model_name)

    print('\nTest set:', len(test_codes))
    print('  Code:', test_codes[0], '\n')

    total_examples = len(test_codes)

    print('\nGenerating descriptions\n')

    generated_descriptions = []

    with tqdm(total=total_examples, file=sys.stdout, colour='green', desc='  Generating summaries') as pbar:

        for code in test_codes:

            code_seq = tokenizer.encode(code, return_tensors='pt', truncation=True, max_length=max_code_len)

            code_seq = code_seq.to(device)

            desc_ids = model.generate(code_seq, min_length=min_desc_len, max_length=max_desc_len,
                                      num_beams=num_beams, early_stopping=True)

            description = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for g in desc_ids]

            description = description[0].strip()

            generated_descriptions.append(description)

            pbar.update(1)

    generated_desc_file = os.path.join(gen_desc_dir, desc_file_name)

    with open(generated_desc_file, 'w') as file:
        file.write('\n'.join(generated_descriptions))
