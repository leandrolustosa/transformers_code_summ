import torch
import os
import wandb

from src.utils import utils
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from code_bert_utils import build_model, generate_descriptions


"""
    https://huggingface.co/microsoft/codebert-base
"""

if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_name = 'wanetal'

    # preproc_config = 'none'
    preproc_config = 'camelsnakecase'

    project_name = f'code_summ_pretrained_{lang}_{corpus_name}'

    test_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/test_{preproc_config}.csv'

    size_threshold = -1

    max_code_len = 300
    max_desc_len = 20

    beam_size = 5

    model_file = '../../../resources/related_work/models/codebert/pytorch_model.bin'

    generated_desc_dir = f'../../../resources/related_work/descriptions/{lang}/{corpus_name}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nUsing device:', device)

    _, _, test_data = utils.read_corpus_csv(test_file_path=test_file_path, sample_size=size_threshold)

    test_codes = test_data[0]

    os.makedirs(generated_desc_dir, exist_ok=True)

    model_path = 'microsoft/codebert-base'

    config = RobertaConfig.from_pretrained(model_path)

    tokenizer = RobertaTokenizer.from_pretrained(model_path, do_lower_case=False)

    model = build_model(model_class=RobertaModel, model_file=model_file, config=config,
                        tokenizer=tokenizer, beam_size=beam_size, max_len=max_desc_len, device=device)

    model = model.to(device)

    print('\nLanguage:', lang)

    print('\nCorpus:', corpus_name, '-', preproc_config)

    print('\n  Test set:', len(test_codes))
    print('    Code:', test_codes[0], '\n')

    wandb.login(key='2122de51cbbe8b9eeac749c5ccb5945dc9453b67')

    with wandb.init(project=project_name) as run:

        run.name = f'pre_codebert_{preproc_config}'
        generated_descriptions = generate_descriptions(test_codes, tokenizer, model, max_code_len, device)

    generated_desc_file = os.path.join(generated_desc_dir, f'pre_codebert_{corpus_name}_{preproc_config}.txt')

    with open(generated_desc_file, 'w') as file:
        file.write('\n'.join(generated_descriptions))
