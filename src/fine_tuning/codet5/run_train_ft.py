import torch
import os

from src.utils.utils import read_corpus_csv
from codet5_utils import build_data
from codet5_helper import train, evaluate_loss, evaluate_bleu, translate_code
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW


if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_name = 'wanetal'

    # model_name = 'small'
    model_name = 'base'

    eval_measure_opt = 'loss'
    # eval_measure_opt = 'bleu'

    preproc_config = 'none'
    # preproc_config = 'camelsnakecase'

    size_threshold = 10
    num_epochs = 5

    model_dir = f'../../../resources/fine_tuning/models/codet5/{model_name}/{eval_measure_opt}/{lang}/' \
                f'{corpus_name}'

    train_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/train_{preproc_config}.csv'
    valid_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/valid_{preproc_config}.csv'

    train_data, valid_data, _ = read_corpus_csv(train_file_path=train_file_path,
                                                valid_file_path=valid_file_path, sample_size=size_threshold)

    max_code_len = 300
    max_desc_len = 20

    batch_size = 32
    num_beams = 5

    weight_decay = 0.01
    grad_accum_steps = 1
    adam_epsilon = 1e-8
    learning_rate = 5e-5
    warmup_steps = 100

    tokenizer_path = None
    model_path = None

    if model_name == 'small':
        tokenizer_path = 'Salesforce/codet5-small'
        model_path = 'Salesforce/codet5-small'
    elif model_name == 'base':
        tokenizer_path = 'Salesforce/codet5-base'
        model_path = 'Salesforce/codet5-base-multi-sum'
    else:
        print('\n\nError Model Name')
        exit(-1)

    os.makedirs(model_dir, exist_ok=True)

    print('\nModel:', model_name)

    print(f'\nCorpus: {lang} - {corpus_name} - {eval_measure_opt}')

    print('\n  Train data:', len(train_data[0]))
    print('    Example code:', train_data[0][0])
    print('    Example Desc:', train_data[1][0])

    print('\n  Valid data:', len(valid_data[0]))
    print('    Example code:', valid_data[0][0])
    print('    Example Desc:', valid_data[1][0])

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

    model = T5ForConditionalGeneration.from_pretrained(model_path)

    train_examples, train_features = build_data(train_data[0], train_data[1], tokenizer, max_code_len,
                                                max_desc_len)

    train_sampler = RandomSampler(train_features)

    train_dataloader = DataLoader(train_features, sampler=train_sampler, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nUsing', device)

    model = model.to(device)

    print('\nModel:', model)

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    num_train_optimization_steps = num_epochs * len(train_dataloader)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    print('\nTraining')

    val_code = valid_data[0][0]

    best_eval_measure = None

    if eval_measure_opt == 'bleu':
        best_eval_measure = 0
    else:
        best_eval_measure = 1e10

    best_epoch = -1

    for epoch in range(int(num_epochs)):

        print('\n  Epoch: {} / {}\n'.format(epoch+1, num_epochs))

        val_desc = translate_code(val_code, model, tokenizer, max_code_len, max_desc_len, num_beams, device)

        print('    Val desc:', val_desc, '\n')

        train_loss = train(train_dataloader, model, optimizer, scheduler, tokenizer, grad_accum_steps, device)

        save_best_model = None
        eval_measure = None

        if eval_measure_opt == 'bleu':
            eval_bleu = evaluate_bleu(valid_data, model, tokenizer, max_code_len, max_desc_len, num_beams, device)
            save_best_model = eval_bleu > best_eval_measure
            eval_measure = eval_bleu
        else:
            eval_loss = evaluate_loss(valid_data, model, tokenizer, max_code_len, max_desc_len, batch_size,
                                      device)
            save_best_model = eval_loss < best_eval_measure
            eval_measure = eval_loss

        # if (epoch + 1) % 2 == 0:
        #     last_ck_dir = os.path.join(model_dir, 'checkpoint-last')
        #     if not os.path.exists(last_ck_dir):
        #         os.makedirs(last_ck_dir)
        #     last_ck_file = os.path.join(last_ck_dir, model_name + '.pt')
        #     torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(), 'loss': train_loss}, last_ck_file)

        if save_best_model:
            print('\n    Saving best model')
            best_eval_measure = eval_measure
            best_epoch = epoch
            output_dir = os.path.join(model_dir, 'best_model')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_model_file = os.path.join(output_dir, 'codet5_' + model_name + '.bin')
            torch.save(model.state_dict(), output_model_file)

        if (epoch - best_epoch) >= 5:
            print('\n    Stopping training ...')
            break

    print('\n\nTraining completed')

    val_desc = translate_code(val_code, model, tokenizer, max_code_len, max_desc_len, num_beams, device)

    print('\n\nVal desc:', val_desc, '\n')