import sys
import torch

from tqdm import tqdm
from codetrans_utils import build_data
from torch.utils.data import DataLoader, SequentialSampler
from bleu import compute_maps, bleu_from_maps


def train(train_dataloader, model, optimizer, scheduler, tokenizer, grad_accum_steps, device):
    model.train()
    train_loss = 0
    train_steps = 0
    with tqdm(total=len(train_dataloader), file=sys.stdout, desc='    Training') as pbar:
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids, target_ids = batch
            source_mask = source_ids.ne(tokenizer.pad_token_id)
            target_mask = target_ids.ne(tokenizer.pad_token_id)
            outputs = model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids,
                            decoder_attention_mask=target_mask)
            loss = outputs.loss
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
            train_loss += loss.item()
            train_steps += 1
            loss.backward()
            if train_steps % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                pbar.set_description('    Training loss {:.4f}'.format(train_loss / train_steps))
            pbar.update(1)
    return train_loss / train_steps


def evaluate_loss(valid_data, model, tokenizer, max_code_len, max_desc_len, batch_size, device):
    model.eval()
    valid_examples, valid_features = build_data(valid_data[0], valid_data[1], tokenizer, max_code_len,
                                                max_desc_len)
    valid_sampler = SequentialSampler(valid_features)
    valid_dataloader = DataLoader(valid_features, sampler=valid_sampler, batch_size=batch_size)
    eval_loss = 0
    eval_steps = 0
    with tqdm(total=len(valid_dataloader), file=sys.stdout, desc='    Validation') as pbar_valid:
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids, target_ids = batch
            source_mask = source_ids.ne(tokenizer.pad_token_id)
            target_mask = target_ids.ne(tokenizer.pad_token_id)
            with torch.no_grad():
                outputs = model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids,
                                decoder_attention_mask=target_mask)
                loss = outputs.loss
            eval_loss += loss.item()
            eval_steps += 1
            pbar_valid.set_description('    Validation loss {:.4f}'.format(eval_loss / eval_steps))
            pbar_valid.update(1)
    return eval_loss / eval_steps


def evaluate_bleu(valid_data, model, tokenizer, max_code_len, max_desc_len, num_beams, device):
    model.eval()
    mean_bleu_score = 0
    with tqdm(total=len(valid_data[0]), file=sys.stdout, desc='    Validation') as pbar:
        for i, (code, ref_desc) in enumerate(zip(valid_data[0], valid_data[1])):
            pred_desc = translate_code(code, model, tokenizer, max_code_len, max_desc_len, num_beams, device)
            gold_map, prediction_map = compute_maps([pred_desc], [ref_desc])
            bleu_scores = bleu_from_maps(gold_map, prediction_map)
            bleu_score = bleu_scores[0] / 100
            mean_bleu_score += bleu_score
            pbar.set_description('    Validation bleu {:.4f}'.format(mean_bleu_score / (i+1)))
            pbar.update(1)
    return mean_bleu_score / len(valid_data[0])


def translate_code(code, model, tokenizer, max_code_len, max_desc_len, num_beams, device):
    input_ids = tokenizer(code, add_special_tokens=True, return_tensors='pt',
                          max_length=max_code_len, truncation=True).input_ids
    input_ids = input_ids.to(device)
    generated_ids = model.generate(input_ids, max_length=max_desc_len, min_length=4, num_beams=num_beams)
    desc = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return desc
