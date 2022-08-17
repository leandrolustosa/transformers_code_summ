import os
import re

from datasets import load_dataset


if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    corpus_dir = f'../../resources/corpora/{lang}/codexglue'

    corpus_dir = os.path.join(corpus_dir, lang)

    os.makedirs(corpus_dir, exist_ok=True)

    dataset = load_dataset('code_x_glue_ct_code_to_text', lang)

    divisions = ['train', 'validation', 'test']

    for division in divisions:

        examples = dataset[division]

        print('\nOriginal Division:', division, '-', len(examples))

        codes = []
        descs = []

        for example in examples:
            desc = example['docstring'].replace('\n', '')
            desc = re.sub('<.*?>', ' ', desc)
            desc = re.sub('\\s+', ' ', desc)
            code = example['code']
            code = code.replace('\t', ' DCSP ')
            code = code.replace('\r\n', ' DCNL ')
            code = code.replace('\n', ' DCNL ')
            codes.append(code)
            descs.append(desc)

        print('\n  Generated Division:', division, '-', len(codes))

        codes_data = '\n'.join(codes)
        descs_data = '\n'.join(descs)

        code_file_path = os.path.join(corpus_dir, division + '.code')

        with open(code_file_path, 'w', encoding='utf-8') as file:
            file.write(codes_data)

        desc_file_path = os.path.join(corpus_dir, division + '.comment')

        with open(desc_file_path, 'w', encoding='utf-8') as file:
            file.write(descs_data)
