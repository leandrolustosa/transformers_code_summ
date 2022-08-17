import torch

from codetrans_classes import Example, InputFeatures
from torch.utils.data import TensorDataset


def build_data(codes, descriptions, tokenizer, max_code_len, max_desc_len, stage='Train'):
    examples = build_examples(codes, descriptions)
    data_features = build_features(examples, tokenizer, max_code_len, max_desc_len)
    all_source_ids = torch.tensor([f.source_ids for f in data_features], dtype=torch.long)
    if stage == 'Train':
        all_target_ids = torch.tensor(
            [f.target_ids for f in data_features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_target_ids)
    else:
        data = TensorDataset(all_source_ids)
    return examples, data


def build_examples(codes, descriptions):
    examples = []
    if descriptions is not None:
        for idx, (code, desc) in enumerate(zip(codes, descriptions)):
            examples.append(Example(idx=idx+1, source=code, target=desc))
    else:
        for idx, code in enumerate(codes):
            examples.append(Example(idx=idx + 1, source=code))
    return examples


def build_features(examples, tokenizer, max_code_len, max_desc_len):
    data_features = []
    for example in examples:
        source_str = example.source
        source_str = source_str.replace('</s>', '<unk>')
        source_ids = tokenizer.encode(source_str, max_length=max_code_len, padding='max_length',
                                      truncation=True)
        target_ids = None
        if example.target is not None:
            target_str = example.target
            target_str = target_str.replace('</s>', '<unk>')
            target_ids = tokenizer.encode(target_str, max_length=max_desc_len, padding='max_length',
                                          truncation=True)
            target_ids = [t if t != 0 else -100 for t in target_ids]
        input_features = InputFeatures(example.idx, source_ids, target_ids)
        data_features.append(input_features)
    return data_features
