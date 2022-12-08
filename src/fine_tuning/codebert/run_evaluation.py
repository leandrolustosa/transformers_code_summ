import codebert_utils as utils
import torch.nn as nn
import torch
import os
import sys

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from model import Seq2Seq
from tqdm import tqdm
from src.utils.utils import read_corpus_csv


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

    model_path = f'../../../resources/fine_tuning/models/codebert/{eval_measure_opt}/{lang}/{corpus_name}' \
                 f'/best_model/codebert.bin'

    gen_desc_dir = f'../../../resources/fine_tuning/descriptions/{lang}/{corpus_name}'

    os.makedirs(gen_desc_dir, exist_ok=True)

    desc_file_name = f'codebert_{preproc_config}_{eval_measure_opt}_ft.txt'

    beam_size = 5

    max_code_len = 300

    max_desc_len = 20

    size_threshold = -1

    test_file_path = f'../../../resources/corpora/{lang}/{corpus_name}/csv/test_{preproc_config}.csv'

    _, _, test_data = read_corpus_csv(test_file_path=test_file_path, sample_size=size_threshold)

    test_codes = test_data[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base', do_lower_case=False)

    config = RobertaConfig.from_pretrained('microsoft/codebert-base')

    print('\n', config)

    encoder = RobertaModel.from_pretrained('microsoft/codebert-base', config=config)

    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)

    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config, beam_size=beam_size,
                    max_length=max_desc_len, sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=False)

    model.to(device)

    print(f'\nCorpus: {corpus_name} - {preproc_config} - {eval_measure_opt}')

    print('\nDevice:', device)

    print('\n  Test set:', len(test_codes))
    print('    Code:', test_codes[0], '\n')

    total_examples = len(test_codes)

    print('\nGenerating descriptions\n')

    generated_descriptions = []

    with tqdm(total=total_examples, file=sys.stdout, colour='green', desc='  Generating summaries') as pbar:

        for code in test_codes:

            example = [utils.Example(idx=None, source=code, target=None)]

            features_code = utils.get_features(example, max_code_len, max_desc_len, tokenizer)

            description, length = utils.inference(features_code, model, tokenizer, device)

            generated_descriptions.append(description[0])

            pbar.update(1)

    generated_desc_file = os.path.join(gen_desc_dir, desc_file_name)

    with open(generated_desc_file, 'w') as file:
        file.write('\n'.join(generated_descriptions))
