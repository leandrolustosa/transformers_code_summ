import torch
import os
import sys

from transformers import RobertaTokenizer, T5Config, T5ForConditionalGeneration
from tqdm import tqdm
from src.utils.utils import read_corpus_csv


if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_name = 'wanetal'

    # model_name = 'small'
    model_name = 'base'

    # eval_measure_opt = 'loss'
    # eval_measure_opt = 'bleu'
    eval_measure_opt = 'rougel'

    preproc_config = 'none'

    if lang == 'java':
        preproc_config = 'camelsnakecase'

    model_path = f'../../../resources/fine_tuning/models/codet5/{model_name}/{eval_measure_opt}/' \
                 f'{lang}/{corpus_name}/best_model/codet5_{model_name}.bin'

    gen_desc_dir = f'../../../resources/fine_tuning/descriptions/{lang}/{corpus_name}'

    os.makedirs(gen_desc_dir, exist_ok=True)

    desc_file_name = f'codet5_{model_name}_{preproc_config}_{eval_measure_opt}_ft.txt'

    beam_size = 5

    max_code_len = 300

    min_desc_len = 4
    max_desc_len = 20

    size_threshold = -1

    test_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/test_{preproc_config}.csv'

    _, _, test_data = read_corpus_csv(test_file_path=test_file_path, sample_size=size_threshold)

    tokenizer_path = None
    pre_model_path = None

    if model_name == 'small':
        tokenizer_path = 'Salesforce/codet5-small'
        pre_model_path = 'Salesforce/codet5-small'
    elif model_name == 'base':
        tokenizer_path = 'Salesforce/codet5-base'
        pre_model_path = 'Salesforce/codet5-base-multi-sum'
    else:
        print('\n\nError Model Name')
        exit(-1)

    test_codes = test_data[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device}')

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

    model_config = T5Config.from_pretrained(pre_model_path)

    model = T5ForConditionalGeneration(config=model_config)

    model = model.to(device)

    model.load_state_dict(torch.load(model_path))

    print(f'\nModel: {model_name} - {preproc_config} - {eval_measure_opt}')
    print(f'\nCorpus: {corpus_name}')

    print(f'\n  Test set: {len(test_codes)}')
    print(f'    Code: {test_codes[0]}\n')

    total_examples = len(test_codes)

    print('\nGenerating descriptions\n')

    generated_descriptions = []

    with tqdm(total=total_examples, file=sys.stdout, colour='green', desc='  Generating summaries') as pbar:

        for code in test_codes:

            input_ids = tokenizer.encode(code, return_tensors='pt', max_length=max_code_len, truncation=True)

            input_ids = input_ids.to(device)

            desc_ids = model.generate(input_ids=input_ids, bos_token_id=model.config.bos_token_id,
                                      eos_token_id=model.config.eos_token_id, length_penalty=2.0,
                                      min_length=min_desc_len, max_length=max_desc_len, num_beams=beam_size)

            desc = tokenizer.decode(desc_ids[0], skip_special_tokens=True)

            generated_descriptions.append(desc)

            pbar.update(1)

    generated_desc_file = os.path.join(gen_desc_dir, desc_file_name)

    with open(generated_desc_file, 'w') as file:
        file.write('\n'.join(generated_descriptions))
