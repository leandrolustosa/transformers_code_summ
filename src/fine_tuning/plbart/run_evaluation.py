import torch
import os
import sys

from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from tqdm import tqdm
from src.utils.utils import read_corpus_csv


if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_name = 'wanetal'

    # eval_measure_opt = 'loss'
    # eval_measure_opt = 'bleu'
    # eval_measure_opt = 'rougel'
    eval_measure_opt = 'meteor'

    preproc_config = 'none'

    if lang == 'java':
        preproc_config = 'camelsnakecase'

    desc_file_name = f'plbart_{preproc_config}_{eval_measure_opt}_ft.txt'

    model_path = f'../../../resources/fine_tuning/models/plbart/{eval_measure_opt}/{lang}/{corpus_name}' \
                 f'/best_model/plbart.bin'

    gen_desc_dir = f'../../../resources/fine_tuning/descriptions/{lang}/{corpus_name}'

    test_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/test_{preproc_config}.csv'

    model_name = None

    if lang == 'java':
        model_name = 'uclanlp/plbart-java-en_XX'
    elif lang == 'python':
        model_name = 'uclanlp/plbart-python-en_XX'
    else:
        print('\nError lang')
        exit(-1)

    os.makedirs(gen_desc_dir, exist_ok=True)

    beam_size = 5

    max_code_len = 300

    min_desc_len = 4
    max_desc_len = 20

    size_threshold = -1

    _, _, test_data = read_corpus_csv(test_file_path=test_file_path, sample_size=size_threshold)

    test_codes = test_data[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device}')

    tokenizer = PLBartTokenizer.from_pretrained(model_name, src_lang=lang, tgt_lang='en_XX')

    model = PLBartForConditionalGeneration.from_pretrained(model_name)

    model = model.to(device)

    model.load_state_dict(torch.load(model_path))

    print(f'\nModel: {model_name} - {eval_measure_opt}')
    print(f'\nCorpus: {corpus_name}')

    print(f'\n  Test set: {len(test_codes)}')
    print(f'    Code: {test_codes[0]} \n')

    total_examples = len(test_codes)

    print('\nGenerating descriptions\n')

    generated_descriptions = []

    with tqdm(total=total_examples, file=sys.stdout, colour='green', desc='  Generating summaries') as pbar:

        for code in test_codes:

            input_ids = tokenizer(code, return_tensors='pt', max_length=max_code_len, truncation=True)

            input_ids = input_ids.to(device)

            translated_tokens = model.generate(**input_ids, min_length=min_desc_len, max_length=max_desc_len,
                                               num_beams=beam_size,
                                               decoder_start_token_id=tokenizer.lang_code_to_id['en_XX'])

            desc = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

            generated_descriptions.append(desc)

            pbar.update(1)

    generated_desc_file = os.path.join(gen_desc_dir, desc_file_name)

    with open(generated_desc_file, 'w') as file:
        file.write('\n'.join(generated_descriptions))
