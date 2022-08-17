import sys
import torch

from tqdm import tqdm
from bleu import compute_maps, bleu_from_maps


def train(train_dataloader, model, optimizer, scheduler, grad_accum_steps, device):
    model.train()
    with tqdm(total=len(train_dataloader), file=sys.stdout, desc='    Training') as train_bar:
        tr_loss = 0
        train_steps = 0
        train_loss = 0
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                               target_ids=target_ids, target_mask=target_mask)
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
            tr_loss += loss.item()
            train_steps += 1
            train_loss = round(tr_loss * grad_accum_steps / train_steps, 4)
            train_bar.set_description('    Training loss {:.4f}'.format(train_loss))
            train_bar.update(1)
            loss.backward()
            if (train_steps + 1) % grad_accum_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
    return train_loss


def evaluate_loss(val_dataloader, model, device):
    model.eval()
    eval_loss, tokens_num = 0, 0
    with tqdm(total=len(val_dataloader), file=sys.stdout, desc='    Evaluating') as valid_bar:
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            with torch.no_grad():
                _, loss, num = model(source_ids=source_ids, source_mask=source_mask,
                                     target_ids=target_ids, target_mask=target_mask)
            eval_loss += loss.sum().item()
            tokens_num += num.sum().item()
            valid_bar.set_description('    Validation loss {:.4f}'.format(eval_loss / tokens_num))
            valid_bar.update(1)
    return eval_loss / tokens_num


def evaluate_bleu(valid_data, model, tokenizer, max_code_len, max_desc_len, translate_code, device):
    model.eval()
    mean_bleu_score = 0
    with tqdm(total=len(valid_data[0]), file=sys.stdout, desc='    Validation') as pbar:
        for i, (code, ref_desc) in enumerate(zip(valid_data[0], valid_data[1])):
            pred_desc = translate_code(code, model, tokenizer, max_code_len, max_desc_len, device,
                                       stage='test')
            gold_map, prediction_map = compute_maps([pred_desc], [ref_desc])
            bleu_scores = bleu_from_maps(gold_map, prediction_map)
            bleu_score = bleu_scores[0] / 100
            mean_bleu_score += bleu_score
            pbar.set_description('    Validation bleu {:.4f}'.format(mean_bleu_score / (i+1)))
            pbar.update(1)
    return mean_bleu_score / len(valid_data[0])
