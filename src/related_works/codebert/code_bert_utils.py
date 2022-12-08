import torch
import torch.nn as nn
import sys

from model import Seq2Seq
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm


def inference(data, model, tokenizer, device='cpu'):
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


def get_features(examples, tokenizer, max_code_len):
    features = convert_examples_to_features(examples, tokenizer, stage='test')
    all_source_ids = torch.tensor([f.source_ids[: max_code_len] for f in features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask[: max_code_len] for f in features], dtype=torch.long)
    return TensorDataset(all_source_ids, all_source_mask)


def build_model(model_class, model_file, config, tokenizer, max_len, beam_size, device='cpu'):
    encoder = model_class(config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config, beam_size=beam_size, max_length=max_len,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device),), strict=False)
    return model


class Example(object):
    def __init__(self, source, target,):
        self.source = source
        self.target = target


class InputFeatures(object):
    def __init__(self, example_id, source_ids, target_ids, source_mask, target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        source_tokens = tokenizer.tokenize(example.source)[: 256 - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = 256 - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        if stage == 'test':
            target_tokens = tokenizer.tokenize('None')
        else:
            target_tokens = tokenizer.tokenize(example.target)[:128 - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = 128 - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length
        features.append(InputFeatures(example_index, source_ids, target_ids, source_mask, target_mask))
    return features


def generate_descriptions(test_codes, tokenizer, model, max_code_len, device):
    total_examples = len(test_codes)
    generated_descriptions = []
    with tqdm(total=total_examples, file=sys.stdout, colour='green', desc='  Generating summaries') as pbar:
        for i in range(total_examples):
            code = test_codes[i]
            example = [Example(source=code, target=None)]
            features_code = get_features(example, tokenizer, max_code_len)
            generated_desc, length = inference(features_code, model, tokenizer, device=device)
            generated_descriptions.append(generated_desc[0])
            pbar.update(1)
    return generated_descriptions
