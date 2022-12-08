import sys
import torch
import codebert_utils
import os

from tqdm import tqdm
from bleu import compute_maps, bleu_from_maps
from src.evaluation.evaluation_measures import compute_rouge, compute_meteor
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used // 1024**2


def train_model(train_dataloader, val_code, valid_data, val_dataloader, tokenizer, model, optimizer,
                scheduler, num_epochs, max_code_len, max_desc_len, grad_accum_steps, eval_measure_opt,
                model_dir, save_model_state, device):

    best_eval_measure = 1e10 if eval_measure_opt == 'loss' else 0

    best_epoch = -1

    for epoch in range(num_epochs):

        print('\n  Epoch: {} / {}'.format(epoch + 1, num_epochs))

        val_desc = codebert_utils.translate_code(val_code, model, tokenizer, max_code_len, max_desc_len,
                                                 device, stage='test')

        print('\n    Val desc:', val_desc, '\n')

        train_loss = train(train_dataloader, model, optimizer, scheduler, grad_accum_steps, device)

        if eval_measure_opt == 'bleu':
            eval_bleu = evaluate_bleu(valid_data, model, tokenizer, max_code_len, max_desc_len, device)
            save_best_model = eval_bleu > best_eval_measure
            eval_measure = eval_bleu
        elif eval_measure_opt == 'rougel':
            eval_rougel = evaluate_rougel(valid_data, model, tokenizer, max_code_len, max_desc_len, device)
            save_best_model = eval_rougel > best_eval_measure
            eval_measure = eval_rougel
        elif eval_measure_opt == 'meteor':
            eval_meteor = evaluate_meteor(valid_data, model, tokenizer, max_code_len, max_desc_len, device)
            save_best_model = eval_meteor > best_eval_measure
            eval_measure = eval_meteor
        else:
            eval_loss = evaluate_loss(val_dataloader, model, device)
            save_best_model = eval_loss < best_eval_measure
            eval_measure = eval_loss

        if save_model_state and (epoch + 1) % 2 == 0:
            last_ck_dir = os.path.join(model_dir, 'checkpoint-last')
            if not os.path.exists(last_ck_dir):
                os.makedirs(last_ck_dir)
            last_ck_file = os.path.join(last_ck_dir, 'model.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': train_loss}, last_ck_file)

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


def train(train_dataloader, model, optimizer, scheduler, grad_accum_steps, device):
    model.train()
    train_loss = 0
    with tqdm(total=len(train_dataloader), file=sys.stdout, colour='red', desc='    Training') as train_bar:
        for train_step, batch in enumerate(train_dataloader, start=1):
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                               target_mask=target_mask)
            loss /= grad_accum_steps
            train_loss += loss.item()
            loss.backward()
            if train_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad(set_to_none=True)
            gpu_utilization = get_gpu_utilization()
            train_bar.set_description(f'    Training loss: {train_loss / train_step:.4f} - GPU: {gpu_utilization} MB')
            train_bar.update(1)
    return train_loss / train_step


def evaluate_loss(val_dataloader, model, device):
    model.eval()
    eval_loss = 0
    tokens_num = 0
    with tqdm(total=len(val_dataloader), file=sys.stdout, colour='blue', desc='    Evaluating') as valid_bar:
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            with torch.no_grad():
                _, loss, num = model(source_ids=source_ids, source_mask=source_mask,
                                     target_ids=target_ids, target_mask=target_mask)
            eval_loss += loss.sum().item()
            tokens_num += num.sum().item()
            gpu_utilization = get_gpu_utilization()
            valid_bar.set_description(f'    Validation loss: {eval_loss / tokens_num:.4f} - GPU: {gpu_utilization} MB')
            valid_bar.update(1)
    return eval_loss / tokens_num


def evaluate_bleu(valid_data, model, tokenizer, max_code_len, max_desc_len, device):
    model.eval()
    mean_bleu_score = 0
    with tqdm(total=len(valid_data[0]), file=sys.stdout, colour='blue', desc='    Validation') as pbar:
        for eval_step, (code, ref_desc) in enumerate(zip(valid_data[0], valid_data[1]), start=1):
            pred_desc = codebert_utils.translate_code(code, model, tokenizer, max_code_len, max_desc_len,
                                                      device, stage='test')
            gold_map, prediction_map = compute_maps([pred_desc], [ref_desc])
            bleu_scores = bleu_from_maps(gold_map, prediction_map)
            bleu_score = bleu_scores[0] / 100
            mean_bleu_score += bleu_score
            pbar.set_description('    Validation bleu {:.4f}'.format(mean_bleu_score / eval_step))
            pbar.update(1)
    return mean_bleu_score / len(valid_data[0])


def evaluate_rougel(valid_data, model, tokenizer, max_code_len, max_desc_len, device):
    model.eval()
    mean_rougel_score = 0
    with tqdm(total=len(valid_data[0]), file=sys.stdout, colour='blue', desc='    Validation') as pbar:
        for eval_step, (code, ref_desc) in enumerate(zip(valid_data[0], valid_data[1]), start=1):
            pred_desc = codebert_utils.translate_code(code, model, tokenizer, max_code_len, max_desc_len, device,
                                                      stage='test')
            rouge_scores = compute_rouge(ref_desc, pred_desc, max_desc_len)
            mean_rougel_score += rouge_scores['rouge-l']['f']
            pbar.set_description(f'    Validation rouge-l {mean_rougel_score/ eval_step:.4f}')
            pbar.update(1)
    return mean_rougel_score / len(valid_data[0])


def evaluate_meteor(valid_data, model, tokenizer, max_code_len, max_desc_len, device):
    model.eval()
    mean_meteor_score = 0
    with tqdm(total=len(valid_data[0]), file=sys.stdout, colour='blue', desc='    Validation') as pbar:
        for eval_step, (code, ref_desc) in enumerate(zip(valid_data[0], valid_data[1]), start=1):
            pred_desc = codebert_utils.translate_code(code, model, tokenizer, max_code_len, max_desc_len, device,
                                                      stage='test')
            meteor_score = compute_meteor(ref_desc.split(' '), pred_desc.split(' '))
            mean_meteor_score += meteor_score
            pbar.set_description(f'    Validation meteor {mean_meteor_score/ eval_step:.4f}')
            pbar.update(1)
    return mean_meteor_score / len(valid_data[0])
