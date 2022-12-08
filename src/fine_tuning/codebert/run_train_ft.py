import codebert_utils
import torch
import torch.nn as nn
import os
import wandb

from src.utils.utils import read_corpus_csv
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from model import Seq2Seq
from torch.optim import AdamW
from codebert_helper import train_model


if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_name = 'wanetal'

    # eval_measure_opt = 'loss'
    # eval_measure_opt = 'bleu'
    eval_measure_opt = 'rougel'
    # eval_measure_opt = 'meteor'

    preproc_config = 'none'

    if lang == 'java':
        preproc_config = 'camelsnakecase'

    project_name = f'code_summ_ft_{lang}_{corpus_name}'

    size_threshold = 60000

    num_epochs = 15

    save_model_state = False

    pretrained_model_file = '../../../resources/related_work/models/codebert/pytorch_model.bin'

    model_dir = f'../../../resources/fine_tuning/models/codebert/{eval_measure_opt}/{lang}/{corpus_name}'

    train_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/train_{preproc_config}.csv'
    valid_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/valid_{preproc_config}.csv'

    train_data, valid_data, _ = read_corpus_csv(train_file_path=train_file_path,
                                                valid_file_path=valid_file_path,
                                                sample_size=size_threshold)

    print(f'\nCorpus: {lang} - {corpus_name} - {eval_measure_opt}')

    print(f'\nTrain data: {len(train_data[0])}')
    print(f'  Example code: {train_data[0][0]}')
    print(f'  Example Desc: {train_data[1][0]}')

    print(f'\nValid data: {len(valid_data[0])}')
    print(f'  Example code: {valid_data[0][0]}')
    print(f'  Example Desc: {valid_data[1][0]}')

    os.makedirs(model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\n\nDevice: {device}')

    max_code_len = 300
    max_desc_len = 20

    batch_size = 16
    num_beams = 5

    weight_decay = 0.01
    grad_accum_steps = 1
    adam_epsilon = 1e-8
    learning_rate = 5e-5
    warmup_steps = 100

    config = RobertaConfig.from_pretrained('microsoft/codebert-base')

    print('\n', config)

    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    encoder = RobertaModel.from_pretrained('microsoft/codebert-base', config=config)

    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)

    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config, beam_size=num_beams,
                    max_length=max_desc_len, sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    model.load_state_dict(torch.load(pretrained_model_file, map_location=torch.device(device), ), strict=False)

    model = model.to(device)

    train_dataloader = codebert_utils.build_dataloader(train_data[0], train_data[1], tokenizer, max_code_len,
                                                       max_desc_len, batch_size, stage='train')

    val_dataloader = codebert_utils.build_dataloader(valid_data[0], valid_data[1], tokenizer, max_code_len,
                                                     max_desc_len, batch_size, stage='dev')

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    t_total = len(train_dataloader) // grad_accum_steps * num_epochs

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * 0.1),
                                                num_training_steps=t_total)

    print(f'\nTraining -- {batch_size}')

    val_code = valid_data[0][0]

    wandb.login(key='2122de51cbbe8b9eeac749c5ccb5945dc9453b67')

    with wandb.init(project=project_name) as run:

        run.name = f'ft_codebert_{preproc_config}_{eval_measure_opt}'

        train_model(train_dataloader, val_code, valid_data, val_dataloader, tokenizer, model, optimizer,
                    scheduler, num_epochs, max_code_len, max_desc_len, grad_accum_steps, eval_measure_opt,
                    model_dir, save_model_state, device)

    print('\n\nTraining completed')

    val_desc = codebert_utils.translate_code(val_code, model, tokenizer, max_code_len, max_desc_len,
                                             device, stage='test')

    print(f'\n\nVal desc: {val_desc}\n')

    import time

    time.sleep(2 * 60)

    os.system('shutdown -h now')
