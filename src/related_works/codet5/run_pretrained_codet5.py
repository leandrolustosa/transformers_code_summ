import torch
import os
import wandb

from transformers import RobertaTokenizer, T5ForConditionalGeneration
from src.utils import utils
from codet5_utils import generate_descriptions


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

    size_threshold = -1

    max_code_len = 300
    max_desc_len = 20

    num_beams = 5

    generated_desc_dir = f'../../../resources/related_work/descriptions/{lang}/{corpus_name}'

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

    print('\nGenerating descriptions\n')

    wandb.login(key='2122de51cbbe8b9eeac749c5ccb5945dc9453b67')

    with wandb.init(project=project_name) as run:
        run.name = f'pre_codet5_{preproc_config_name}'
        generated_descriptions = generate_descriptions(test_codes, tokenizer, model, max_code_len,
                                                       max_desc_len, num_beams, device)

    generated_desc_file = os.path.join(generated_desc_dir,
                                       f'pre_codet5_{corpus_name}_{preproc_config_name}.txt')

    with open(generated_desc_file, 'w') as file:
        file.write('\n'.join(generated_descriptions))
