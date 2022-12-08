import torch
import os
import wandb

from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from src.utils import utils
from plbart_utils import generate_descriptions


if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_name = 'wanetal'

    # preproc_config_name = 'none'
    preproc_config_name = 'camelsnakecase'

    project_name = f'code_summ_pretrained_{lang}_{corpus_name}'

    test_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/test_{preproc_config_name}.csv'

    generated_desc_dir = f'../../../resources/related_work/descriptions/{lang}/{corpus_name}'

    size_threshold = -1

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

    start_token_id = tokenizer.lang_code_to_id['en_XX']

    wandb.login(key='2122de51cbbe8b9eeac749c5ccb5945dc9453b67')

    with wandb.init(project=project_name) as run:
        run.name = f'pre_plbart_{preproc_config_name}'
        generated_descriptions = generate_descriptions(test_codes, tokenizer, model, max_code_len, max_desc_len,
                                                       num_beams, start_token_id, device)

    generated_desc_file = os.path.join(generated_desc_dir,
                                       f'pre_plbart_{corpus_name}_{preproc_config_name}.txt')

    with open(generated_desc_file, 'w') as file:
        file.write('\n'.join(generated_descriptions))
