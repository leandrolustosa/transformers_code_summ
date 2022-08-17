import codebert_utils
import torch
import torch.nn as nn
import os

from src.utils.utils import read_corpus_csv
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from model import Seq2Seq
from torch.optim import AdamW
from codebert_helper import train, evaluate_loss, evaluate_bleu


if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_name = 'wanetal'

    # eval_measure_opt = 'loss'
    eval_measure_opt = 'bleu'

    preproc_config = 'none'
    # preproc_config = 'camelsnakecase'

    size_threshold = 100000

    num_epochs = 5
    # num_epochs = 20

    model_dir = f'../../../resources/fine_tuning/models/codebert/{eval_measure_opt}/{lang}/{corpus_name}'

    train_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/train_{preproc_config}.csv'
    valid_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/valid_{preproc_config}.csv'

    train_data, valid_data, _ = read_corpus_csv(train_file_path=train_file_path,
                                                valid_file_path=valid_file_path,
                                                sample_size=size_threshold)

    print(f'\nCorpus: {lang} - {corpus_name} - {eval_measure_opt}')

    print('\nTrain data:', len(train_data[0]))
    print('  Example code:', train_data[0][0])
    print('  Example Desc:', train_data[1][0])

    print('\nValid data:', len(valid_data[0]))
    print('  Example code:', valid_data[0][0])
    print('  Example Desc:', valid_data[1][0])

    os.makedirs(model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\n\nUsing', device)

    max_code_len = 300
    max_desc_len = 20

    batch_size = 32
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

    print('\nTraining')

    val_code = valid_data[0][0]

    best_eval_measure = None

    if eval_measure_opt == 'bleu':
        best_eval_measure = 0
    else:
        best_eval_measure = 1e10

    best_epoch = -1

    for epoch in range(num_epochs):

        print('\n  Epoch: {} / {}'.format(epoch + 1, num_epochs))

        val_desc = codebert_utils.translate_code(val_code, model, tokenizer, max_code_len, max_desc_len,
                                                 device, stage='test')

        print('\n    Val desc:', val_desc, '\n')

        train_loss = train(train_dataloader, model, optimizer, scheduler, grad_accum_steps, device)

        save_best_model = None
        eval_measure = None

        if eval_measure_opt == 'bleu':
            eval_bleu = evaluate_bleu(valid_data, model, tokenizer, max_code_len, max_desc_len,
                                      codebert_utils.translate_code, device)
            save_best_model = eval_bleu > best_eval_measure
            eval_measure = eval_bleu
        else:
            eval_loss = evaluate_loss(val_dataloader, model, device)
            save_best_model = eval_loss < best_eval_measure
            eval_measure = eval_loss

        # if (epoch + 1) % 2 == 0:
        #     last_ck_dir = os.path.join(model_dir, 'checkpoint-last')
        #     if not os.path.exists(last_ck_dir):
        #         os.makedirs(last_ck_dir)
        #     last_ck_file = os.path.join(last_ck_dir, 'model.pt')
        #     torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(), 'loss': train_loss}, last_ck_file)

        if save_best_model:
            print('\n    Saving best model ...')
            best_eval_measure = eval_measure
            best_epoch = epoch
            output_dir = os.path.join(model_dir, 'best_model')
            os.makedirs(output_dir, exist_ok=True)
            output_model_file = os.path.join(output_dir, 'codebert.bin')
            torch.save(model.state_dict(), output_model_file)

        if (epoch - best_epoch) >= 5:
            print('\n    Stopping training ...')
            break

    print('\n\nTraining completed')

    val_desc = codebert_utils.translate_code(val_code, model, tokenizer, max_code_len, max_desc_len,
                                             device, stage='test')

    print('\n\nVal desc:', val_desc, '\n')
