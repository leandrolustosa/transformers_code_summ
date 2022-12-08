import sys
import torch
import os

from tqdm import tqdm
from codet5_utils import build_data
from torch.utils.data import DataLoader, SequentialSampler
from bleu import compute_maps, bleu_from_maps
from src.evaluation.evaluation_measures import compute_rouge
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used // 1024**2


def run_train(train_dataloader, val_code, valid_data, tokenizer, model, model_name, optimizer, scheduler,
              num_epochs, max_code_len, max_desc_len, num_beams, batch_size, grad_accum_steps,
              eval_measure_opt, model_dir, save_model_state, device):

    best_eval_measure = 1e10 if eval_measure_opt == 'loss' else 0

    best_epoch = -1

    for epoch in range(int(num_epochs)):

        print('\n  Epoch: {} / {}\n'.format(epoch+1, num_epochs))

        val_desc = translate_code(val_code, model, tokenizer, max_code_len, max_desc_len, num_beams, device)

        print('    Val desc:', val_desc, '\n')

        train_loss = train(train_dataloader, model, optimizer, scheduler, tokenizer, grad_accum_steps, device)

        if eval_measure_opt == 'bleu':
            eval_bleu = evaluate_bleu(valid_data, model, tokenizer, max_code_len, max_desc_len, num_beams, device)
            save_best_model = eval_bleu > best_eval_measure
            eval_measure = eval_bleu
        elif eval_measure_opt == 'rougel':
            eval_rougel = evaluate_rougel(valid_data, model, tokenizer, max_code_len, max_desc_len, num_beams, device)
            save_best_model = eval_rougel > best_eval_measure
            eval_measure = eval_rougel
        else:
            eval_loss = evaluate_loss(valid_data, model, tokenizer, max_code_len, max_desc_len, batch_size,
                                      device)
            save_best_model = eval_loss < best_eval_measure
            eval_measure = eval_loss

        if save_model_state and (epoch + 1) % 2 == 0:
            last_ck_dir = os.path.join(model_dir, 'checkpoint-last')
            if not os.path.exists(last_ck_dir):
                os.makedirs(last_ck_dir)
            last_ck_file = os.path.join(last_ck_dir, model_name + '.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': train_loss}, last_ck_file)

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


def train(train_dataloader, model, optimizer, scheduler, tokenizer, grad_accum_steps, device):
    model.train()
    train_loss = 0
    with tqdm(total=len(train_dataloader), file=sys.stdout, colour='red', desc='    Training') as pbar:
        for train_step, batch in enumerate(train_dataloader, start=1):
            batch = tuple(t.to(device) for t in batch)
            source_ids, target_ids = batch
            source_mask = source_ids.ne(tokenizer.pad_token_id)
            target_mask = target_ids.ne(tokenizer.pad_token_id)
            outputs = model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids,
                            decoder_attention_mask=target_mask)
            loss = outputs.loss
            loss = loss / grad_accum_steps
            train_loss += loss.item()
            loss.backward()
            if train_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad(set_to_none=True)
            gpu_utilization = get_gpu_utilization()
            pbar.set_description(f'    Training loss: {train_loss / train_step:.4f} - GPU: {gpu_utilization} MB')
            pbar.update(1)
    return train_loss / train_step


def evaluate_loss(valid_data, model, tokenizer, max_code_len, max_desc_len, batch_size, device):
    model.eval()
    valid_examples, valid_features = build_data(valid_data[0], valid_data[1], tokenizer, max_code_len,
                                                max_desc_len)
    valid_sampler = SequentialSampler(valid_features)
    valid_dataloader = DataLoader(valid_features, sampler=valid_sampler, batch_size=batch_size)
    eval_loss = 0
    with tqdm(total=len(valid_dataloader), file=sys.stdout, colour='blue', desc='    Validation') as pbar_valid:
        for eval_step, batch in enumerate(valid_dataloader, start=1):
            batch = tuple(t.to(device) for t in batch)
            source_ids, target_ids = batch
            source_mask = source_ids.ne(tokenizer.pad_token_id)
            target_mask = target_ids.ne(tokenizer.pad_token_id)
            with torch.no_grad():
                outputs = model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids,
                                decoder_attention_mask=target_mask)
            loss = outputs.loss
            eval_loss += loss.item()
            gpu_utilization = get_gpu_utilization()
            pbar_valid.set_description(f'    Validation loss: {eval_loss / eval_step:.4f} - GPU: {gpu_utilization} MB')
            pbar_valid.update(1)
    return eval_loss / eval_step


def evaluate_bleu(valid_data, model, tokenizer, max_code_len, max_desc_len, num_beams, device):
    model.eval()
    mean_bleu_score = 0
    with tqdm(total=len(valid_data[0]), file=sys.stdout, colour='blue', desc='    Validation') as pbar:
        for eval_step, (code, ref_desc) in enumerate(zip(valid_data[0], valid_data[1]), start=1):
            pred_desc = translate_code(code, model, tokenizer, max_code_len, max_desc_len, num_beams, device)
            gold_map, prediction_map = compute_maps([pred_desc], [ref_desc])
            bleu_scores = bleu_from_maps(gold_map, prediction_map)
            bleu_score = bleu_scores[0] / 100
            mean_bleu_score += bleu_score
            pbar.set_description('    Validation bleu {:.4f}'.format(mean_bleu_score / eval_step))
            pbar.update(1)
    return mean_bleu_score / len(valid_data[0])


def evaluate_rougel(valid_data, model, tokenizer, max_code_len, max_desc_len, num_beams, device):
    model.eval()
    mean_rougel = 0
    with tqdm(total=len(valid_data[0]), file=sys.stdout, colour='blue', desc='    Validation') as pbar:
        for eval_step, (code, ref_desc) in enumerate(zip(valid_data[0], valid_data[1]), start=1):
            pred_desc = translate_code(code, model, tokenizer, max_code_len, max_desc_len, num_beams, device)
            rouge_scores = compute_rouge(ref_desc, pred_desc, max_desc_len)
            mean_rougel += rouge_scores['rouge-l']['f']
            pbar.set_description(f'    Validation rouge-l {mean_rougel/ eval_step:.4f}')
            pbar.update(1)
    return mean_rougel / len(valid_data[0])


def translate_code(code, model, tokenizer, max_code_len, max_desc_len, num_beams, device):
    input_ids = tokenizer.encode(code, return_tensors='pt', max_length=max_code_len, truncation=True)
    input_ids = input_ids.to(device)
    desc_ids = model.generate(input_ids=input_ids, bos_token_id=model.config.bos_token_id,
                              eos_token_id=model.config.eos_token_id, length_penalty=2.0,
                              min_length=4, max_length=max_desc_len, num_beams=num_beams)
    desc = tokenizer.decode(desc_ids[0], skip_special_tokens=True)
    return desc
