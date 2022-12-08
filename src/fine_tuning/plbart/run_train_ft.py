import torch
import os
import wandb

from src.utils.utils import read_corpus_csv
from plbart_utils import build_data
from plbart_helper import run_train, translate_code
from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW


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

    project_name = f'code_summ_ft_{lang}_{corpus_name}'

    model_name = None

    if lang == 'java':
        model_name = 'uclanlp/plbart-java-en_XX'
    elif lang == 'python':
        model_name = 'uclanlp/plbart-python-en_XX'
    else:
        print('\nError lang')
        exit(-1)

    model_dir = f'../../../resources/fine_tuning/models/plbart/{eval_measure_opt}/{lang}/{corpus_name}'

    size_threshold = 60000

    num_epochs = 15

    save_model_state = False

    train_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/train_{preproc_config}.csv'
    valid_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/valid_{preproc_config}.csv'

    train_data, valid_data, _ = read_corpus_csv(train_file_path=train_file_path, valid_file_path=valid_file_path,
                                                sample_size=size_threshold)

    print(f'\nCorpus: {lang} - {corpus_name} - {eval_measure_opt}')

    print(f'\nModel: {model_name}')

    print(f'\n  Train data: {len(train_data[0])}')
    print(f'    Example code: {train_data[0][0]}')
    print(f'    Example Desc: {train_data[1][0]}')

    print(f'\n  Valid data: {len(valid_data[0])}')
    print(f'    Example code: {valid_data[0][0]}')
    print(f'    Example Desc: {valid_data[1][0]}')

    os.makedirs(model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device}')

    num_beams = 5

    max_code_len = 300
    max_desc_len = 20

    batch_size = 32

    weight_decay = 0.01
    grad_accum_steps = 1
    adam_epsilon = 1e-8
    learning_rate = 5e-5
    warmup_steps = 100

    tokenizer = PLBartTokenizer.from_pretrained(model_name, src_lang=lang, tgt_lang='en_XX')

    model = PLBartForConditionalGeneration.from_pretrained(model_name)

    train_examples, train_features = build_data(train_data[0], train_data[1], tokenizer, max_code_len,
                                                max_desc_len, stage='Train')

    train_sampler = RandomSampler(train_features)

    train_dataloader = DataLoader(train_features, sampler=train_sampler, batch_size=batch_size)

    model = model.to(device)

    print(f'\nModel: {model} -- {batch_size}')

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    t_total = len(train_dataloader) // grad_accum_steps * num_epochs

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    num_train_optimization_steps = num_epochs * len(train_dataloader)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    print('\nTraining')

    val_code = valid_data[0][0]

    wandb.login(key='2122de51cbbe8b9eeac749c5ccb5945dc9453b67')

    with wandb.init(project=project_name) as run:

        run.name = f'plbart_{preproc_config}_{eval_measure_opt}'

        run_train(train_dataloader, val_code, valid_data, tokenizer, model, model_name, optimizer, scheduler,
                  num_epochs, max_code_len, max_desc_len, num_beams, batch_size, grad_accum_steps,
                  eval_measure_opt, model_dir, save_model_state, device)

    print('\n\nTraining completed')

    val_desc = translate_code(val_code, model, tokenizer, max_code_len, max_desc_len, num_beams, device)

    print(f'\n\nVal desc: {val_desc} \n')
