import os
import torch
import wandb

from src.utils import utils
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from codetrans_utils import generate_descriptions


"""
    https://github.com/agemagician/CodeTrans
"""

if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_name = 'wanetal'

    # preproc_config_name = 'none'
    preproc_config_name = 'camelsnakecase'

    # model_name = 'codetrans_mt_tf_small'
    # model_name = 'codetrans_mt_tf_base'
    model_name = 'codetrans_mt_tf_large'

    project_name = f'code_summ_pretrained_{lang}_{corpus_name}'

    test_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/test_{preproc_config_name}.csv'

    size_threshold = -1

    max_code_len = 300
    max_len_desc = 20

    num_beams = 5

    generated_desc_dir = f'../../../resources/related_work/descriptions/{lang}/{corpus_name}'

    model_path = None

    if lang == 'python':
        if model_name == 'codetrans_mt_tf_base':
            model_path = 'SEBIS/code_trans_t5_base_code_documentation_generation_python_multitask_finetune'
        elif model_name == 'codetrans_mt_tf_small':
            model_path = 'SEBIS/code_trans_t5_small_code_documentation_generation_python_multitask_finetune'
        elif model_name == 'codetrans_mt_tf_large':
            model_path = 'SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask_finetune'
    elif lang == 'java':
        if model_name == 'codetrans_mt_tf_base':
            model_path = 'SEBIS/code_trans_t5_base_code_documentation_generation_java_multitask_finetune'
        elif model_name == 'codetrans_mt_tf_small':
            model_path = 'SEBIS/code_trans_t5_small_code_documentation_generation_java_multitask_finetune'
        elif model_name == 'codetrans_mt_tf_large':
            model_path = 'SEBIS/code_trans_t5_large_code_documentation_generation_java_multitask_finetune'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nDevice:', device)

    _, _, test_data = utils.read_corpus_csv(test_file_path=test_file_path, sample_size=size_threshold)

    test_codes = test_data[0]

    os.makedirs(generated_desc_dir, exist_ok=True)

    print('\nLanguage:', lang)

    print('\nCorpus:', corpus_name, '-', preproc_config_name)

    print('\n  Test set:', len(test_codes))
    print('    Code:', test_codes[0])

    print('\nModel:', model_name, '\n')

    tokenizer = AutoTokenizer.from_pretrained(model_path, skip_special_tokens=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    wandb.login(key='2122de51cbbe8b9eeac749c5ccb5945dc9453b67')

    with wandb.init(project=project_name) as run:
        run.name = f'pre_{model_name}_{preproc_config_name}'
        generated_descriptions = generate_descriptions(test_codes, tokenizer, model, max_code_len,
                                                       max_len_desc, num_beams, device)

    generated_desc_file = os.path.join(generated_desc_dir,
                                       f'pre_{model_name}_{corpus_name}_{preproc_config_name}.txt')

    with open(generated_desc_file, 'w') as file:
        file.write('\n'.join(generated_descriptions))
