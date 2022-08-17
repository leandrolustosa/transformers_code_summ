import torch

from classes import Example, InputFeatures
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader


def convert_examples(codes, descriptions):
    examples = []
    for idx, (code, desc) in enumerate(zip(codes, descriptions)):
        examples.append(Example(idx=idx+1, source=code, target=desc))
    return examples


def convert_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # Source
        source_tokens = tokenizer.tokenize(example.source)[:max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        # Target
        if stage == 'test':
            target_tokens = tokenizer.tokenize('None')
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length
        features.append(InputFeatures(example_index, source_ids, target_ids, source_mask, target_mask))
    return features


def get_features(examples, max_source_len, max_target_len, tokenizer):
    features = convert_to_features(examples, tokenizer, max_source_len, max_target_len, stage='test')
    all_source_ids = torch.tensor([f.source_ids[: max_source_len] for f in features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask[: max_source_len] for f in features], dtype=torch.long)
    return TensorDataset(all_source_ids, all_source_mask)


def build_dataloader(codes, descriptions, tokenizer, max_source_len, max_target_len, batch_size, stage):
    examples = convert_examples(codes, descriptions)
    features = convert_to_features(examples, tokenizer, max_source_len, max_target_len, stage=stage)
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)
    data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
    if stage == 'train':
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def translate_code(test_code, model, tokenizer, max_source_len, max_target_len, device, stage):
    model.eval()
    examples = convert_examples([test_code], [test_code])
    features = convert_to_features(examples, tokenizer, max_source_len, max_target_len, stage=stage)
    test_code_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    test_code_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
    with torch.no_grad():
        test_code_ids = test_code_ids.to(device)
        test_code_mask = test_code_mask.to(device)
        pred = model(source_ids=test_code_ids, source_mask=test_code_mask)
        pred = pred[0]
        tokens_ids = pred[0].cpu().numpy()
        tokens_ids = list(tokens_ids)
        if 0 in tokens_ids:
            tokens_ids = tokens_ids[:tokens_ids.index(0)]
        test_desc_train = tokenizer.decode(tokens_ids, clean_up_tokenization_spaces=False)
    return test_desc_train


def inference(data, model, tokenizer, device='cuda'):
    eval_sampler = SequentialSampler(data)
    eval_dataloader = DataLoader(data, sampler=eval_sampler, batch_size=len(data))
    model.eval()
    p = []
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask = batch
        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask)
            for pred in preds:
                pred_cpu = pred[0].cpu().numpy()
                pred_cpu = list(pred_cpu)
                if 0 in pred_cpu:
                    pred_cpu = pred_cpu[: pred_cpu.index(0)]
                text = tokenizer.decode(pred_cpu, clean_up_tokenization_spaces=False)
                p.append(text)
    return p, source_ids.shape[-1]
