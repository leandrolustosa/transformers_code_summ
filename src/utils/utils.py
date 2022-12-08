import os
import json
import pandas as pd
import re
import numpy as np
import sys

from tqdm import tqdm
from io import BytesIO

from sklearn.model_selection import train_test_split

from src.utils import constants as const
from src.utils import nlp as nlp


def clean_python_data(codes, descriptions, threshold=5):
    all_set = {'code': [], 'desc': []}

    with tqdm(total=len(codes), file=sys.stdout,
              desc='  Cleaning Python Data') as pbar:

        for i, (code, desc) in enumerate(zip(codes, descriptions)):

            if not(nlp.is_english_text(code) and nlp.is_english_text(desc)) \
                    or nlp.starts_with_annotation(code):
                    # or nlp.has_nested_functions(code):
                pbar.update(1)
                continue

            all_set["code"].append(code)
            all_set["desc"].append(desc.replace(':return: ', 'Return ').split(const.DCNL)[0])

            pbar.update(1)

    return all_set


def clean_python_code(text):
    text = text.replace(const.DCNL_DCSP, '\n\t')
    text = text.replace(const.DCNL, '\n')
    text = text.replace(const.DCSP, '\t')
    return text


def clean_python_comment(description):
    description = description.split(const.DCNL)[0]

    description = description.lower()

    description = description.replace("this's", 'this is')
    description = description.replace("that's", 'that is')
    description = description.replace("there's", 'there is')

    description = description.replace('\\', '')
    description = description.replace('``', '')
    description = description.replace('`', '')
    description = description.replace('\'', '')

    removes = re.findall("(?<=[(])[^()]+[^()]+(?=[)])", description)

    for r in removes:
        description = description.replace('(' + r + ')', '')

    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', description)

    for url in urls:
        description = description.replace(url, 'URL')

    description = description.split('.')[0]
    description = description.split(',')[0]
    description = description.split(':param')[0]
    description = description.split('@param')[0]
    description = description.split('>>>')[0]

    description = description.strip().strip('\n') + ' .'

    return description


def generate_python_pairs(original_path, spllited_path):
    processed_code_file = os.path.join(spllited_path, const.PYTHON_CORPUS_ALL_CODE)
    with open(processed_code_file, 'w', encoding='utf8') as code_file:
        processed_desc_file = os.path.join(spllited_path, const.PYTHON_CORPUS_ALL_COMMENT)
        with open(processed_desc_file, 'w', encoding='utf8') as comment_file:
            original_code_file = os.path.join(original_path, 'data_ps_old.declbodies')
            with open(original_code_file, 'r', encoding='utf8') as declbodies_file:
                original_desc_file = os.path.join(original_path, 'data_ps_old.descriptions')
                with open(original_desc_file, 'r', encoding='utf8') as descriptions_file:
                    declbodies_lines = declbodies_file.readlines()
                    descriptions_lines = descriptions_file.readlines()
                    for i in range(len(declbodies_lines)):
                        code = clean_python_code(declbodies_lines[i])
                        comment = clean_python_comment(descriptions_lines[i])
                        if not comment.startswith('todo') and comment[0].isalpha():
                            code_file.write(code)
                            comment_file.write(comment + '\n')


def split_python_dataset(processed_path, spllited_path, train_portion, test_portion=0.5):
    print('   Spliting Python corpus:')

    all_code_file = os.path.join(processed_path, const.PYTHON_CORPUS_ALL_CODE)
    all_comment_file = os.path.join(processed_path, const.PYTHON_CORPUS_ALL_COMMENT)

    all_codes, all_descs = extract_code_desc_python(all_code_file, all_comment_file)

    codes_train, codes_valid_test, descs_train, descs_valid_test = train_test_split(all_codes, all_descs, train_size=train_portion)

    codes_valid, codes_test, descs_valid, descs_test = train_test_split(codes_valid_test, descs_valid_test, test_size=test_portion)

    print('\n  Train set: %d.' % len(codes_train))
    print('  Validation set: %d.' % len(codes_valid))
    print('  Test set: %d.' % len(codes_test))

    train_code_file = os.path.join(spllited_path, const.PYTHON_CORPUS_TRAIN_CODE)
    with open(train_code_file, 'w', encoding='utf8') as train_file:
        valid_code_file = os.path.join(spllited_path, const.PYTHON_CORPUS_VALID_CODE)
        with open(valid_code_file, 'w', encoding='utf8') as valid_file:
            test_code_file = os.path.join(spllited_path, const.PYTHON_CORPUS_TEST_CODE)
            with open(test_code_file, 'w', encoding='utf8') as test_file:
                train_file.write("\n".join(codes_train))
                valid_file.write("\n".join(codes_valid))
                test_file.write("\n".join(codes_test))

    train_desc_file = os.path.join(spllited_path, const.PYTHON_CORPUS_TRAIN_COMMENT)
    with open(train_desc_file, 'w', encoding='utf8') as train_file:
        valid_desc_file = os.path.join(spllited_path, const.PYTHON_CORPUS_VALID_COMMENT)
        with open(valid_desc_file, 'w', encoding='utf8') as valid_file:
            test_desc_file = os.path.join(spllited_path, const.PYTHON_CORPUS_TEST_COMMENT)
            with open(test_desc_file, 'w', encoding='utf8') as test_file:
                train_file.write("\n".join(descs_train))
                valid_file.write("\n".join(descs_valid))
                test_file.write("\n".join(descs_test))


def read_python_corpus(corpus_dir, convert_data_=False):

    train_code_file = os.path.join(corpus_dir, const.PYTHON_CORPUS_TRAIN_CODE)
    train_desc_file = os.path.join(corpus_dir, const.PYTHON_CORPUS_TRAIN_COMMENT)

    valid_code_file = os.path.join(corpus_dir, const.PYTHON_CORPUS_VALID_CODE)
    valid_desc_file = os.path.join(corpus_dir, const.PYTHON_CORPUS_VALID_COMMENT)

    test_code_file = os.path.join(corpus_dir, const.PYTHON_CORPUS_TEST_CODE)
    test_desc_file = os.path.join(corpus_dir, const.PYTHON_CORPUS_TEST_COMMENT)

    train_code, train_desc = get_data(train_code_file, train_desc_file)

    valid_code, valid_desc = get_data(valid_code_file, valid_desc_file)

    test_code, test_desc = get_data(test_code_file, test_desc_file)

    if convert_data_:

        train_code = [[int(code) for code in codes.split()]
                      for codes in train_code]
        valid_code = [[int(code) for code in codes.split()]
                      for codes in valid_code]
        test_code = [[int(code) for code in codes.split()]
                     for codes in test_code]

        train_desc = [[int(code) for code in codes.split()]
                      for codes in train_desc]
        valid_desc = [[int(code) for code in codes.split()]
                      for codes in valid_desc]
        test_desc = [[int(code) for code in codes.split()]
                     for codes in test_desc]

    corpus = {
        'train_code': train_code, 'train_desc': train_desc,
        'valid_code': valid_code, 'valid_desc': valid_desc,
        'test_code': test_code, 'test_desc': test_desc
    }

    return corpus


def preprocess_python_corpus(origin_corpus_dir, processed_corpus_dir):

    all_code_file = os.path.join(origin_corpus_dir, const.PYTHON_CORPUS_ALL_CODE)
    all_comment_file = os.path.join(origin_corpus_dir, const.PYTHON_CORPUS_ALL_COMMENT)

    all_codes, all_descs = extract_code_desc_python(all_code_file, all_comment_file)

    all_set = clean_python_data(all_codes, all_descs)

    print('\n  All size:', len(all_codes))

    proc_all_code_file = os.path.join(processed_corpus_dir, const.PYTHON_CORPUS_ALL_CODE)
    proc_all_comment_file = os.path.join(processed_corpus_dir, const.PYTHON_CORPUS_ALL_COMMENT)

    with open(proc_all_code_file, 'w', encoding='utf8') as code_file:
        with open(proc_all_comment_file, 'w', encoding='utf8') as comment_file:
            code_file.write("\n".join(all_set["code"]))
            comment_file.write("\n".join(all_set["desc"]))


def extract_code_desc_python(code_file, desc_file):
    with open(code_file, encoding='utf-8') as file:
        code = [line.replace('\n', '').strip()
                for line in file.readlines()]
    with open(desc_file, encoding='utf-8') as file:
        desc = [line.replace('\n', '').strip()
                for line in file.readlines()]
    return code, desc


def read_java_corpus(corpus_dir):

    train_file = os.path.join(corpus_dir, const.JAVA_CORPUS_TRAIN_NAME)
    valid_file = os.path.join(corpus_dir, const.JAVA_CORPUS_VALID_NAME)
    test_file = os.path.join(corpus_dir, const.JAVA_CORPUS_TEST_NAME)

    train_code, train_desc = extract_code_desc(train_file)
    valid_code, valid_desc = extract_code_desc(valid_file)
    test_code, test_desc = extract_code_desc(test_file)

    corpus = {
        'train_code': train_code, 'train_desc': train_desc,
        'valid_code': valid_code, 'valid_desc': valid_desc,
        'test_code': test_code, 'test_desc': test_desc
    }

    return corpus


def read_corpus(corpus_dir):

    train_code_file = os.path.join(corpus_dir, 'train.code')
    train_desc_file = os.path.join(corpus_dir, 'train.comment')

    valid_code_file = os.path.join(corpus_dir, 'validation.code')
    valid_desc_file = os.path.join(corpus_dir, 'validation.comment')

    test_code_file = os.path.join(corpus_dir, 'test.code')
    test_desc_file = os.path.join(corpus_dir, 'test.comment')

    train_code, train_desc = get_data(train_code_file, train_desc_file)
    valid_code, valid_desc = get_data(valid_code_file, valid_desc_file)
    test_code, test_desc = get_data(test_code_file, test_desc_file)

    corpus = {
        'train_code': train_code, 'train_desc': train_desc,
        'valid_code': valid_code, 'valid_desc': valid_desc,
        'test_code': test_code, 'test_desc': test_desc
    }

    return corpus


def get_data(code_file, desc_file):
    with open(code_file, encoding='utf-8') as file:
        lines = file.readlines()
    codes = []
    for line in lines:
        code = line.replace('\n', '').strip()
        code = code.replace(' DCNL DCSP ', '\r\n  ')
        code = code.replace(' DCSP ', '  ')
        code = code.replace('DCSP ', '  ')
        code = code.replace(' DCNL ', '\r\n')
        code = code.replace(' DCNL ', '\n')
        codes.append(code)
    with open(desc_file, encoding='utf-8') as file:
        descs = [line.replace('\n', '').strip() for line in file.readlines()]
    return codes, descs


def extract_code_desc(file_path):
    codes = []
    descs = []
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            dict_line = json.loads(line)
            code = dict_line['code'].replace('\n', '').strip()
            desc = dict_line['comment'].replace('\n', '').strip()
            codes.append(code)
            descs.append(desc)
    return codes, descs


def filter_duplicates(data, aux_data):
    aux_data = np.asarray(aux_data)
    set_tokens_code = list(aux_data[:, 0])
    new_data = []
    for tokens_code, tokens_desc in data:
        if tokens_code not in set_tokens_code:
            new_data.append((tokens_code, tokens_desc))
    return new_data


def read_descriptions(systems_dir):
    assert os.path.exists(systems_dir)
    systems_names = os.listdir(systems_dir)
    sys_descriptions = {}
    for system_name in systems_names:
        system_file = os.path.join(systems_dir, system_name)
        with open(system_file) as file:
            content = file.read()
        lines = content.split('\n')
        if '.' in system_name:
            system_name = system_name.split('.')[0]
        sys_descriptions[system_name] = lines
    return sys_descriptions


def save_processed_file(examples, corpus_dir, file_name,
                        is_convert=False):
    processed_file = os.path.join(corpus_dir, file_name)
    if is_convert:
        string_ids = map(str, examples)
        data = '\n'.join(string_ids)
        data = data.replace('[', '').replace(']', '')
    else:
        data = '\n'.join(examples)
    with open(processed_file, 'w', encoding='utf-8') as file:
        file.write(data)


def load_corpus(base_path, file_input_path, file_target_path):

    with open(base_path + '/' + file_input_path, encoding='utf-8') as file_input:
        input_lines = file_input.readlines()

    with open(base_path + '/' + file_target_path, encoding='utf-8') as file_target:
        target_lines = file_target.readlines()

    inputs = [input_.replace('\n', '') for input_ in input_lines]
    targets = [target.replace('\n', '') for target in target_lines]

    return inputs, targets


def read_corpus_csv(train_file_path=None, valid_file_path=None,
                    test_file_path=None, sample_size=-1):
    train_data = read_csv_file(train_file_path, sample_size)
    valid_data = read_csv_file(valid_file_path, sample_size)
    test_data = read_csv_file(test_file_path, sample_size)
    return train_data, valid_data, test_data


def read_csv_file(file_path, sample_size=-1):
    if file_path is not None and os.path.exists(file_path):
        if sample_size > 0:
            df = pd.read_csv(file_path, sep='\t', na_filter=False, nrows=sample_size)
        else:
            df = pd.read_csv(file_path, sep='\t', na_filter=False)
        tokens = df['code'].tolist()
        descriptions = df['desc'].tolist()
        return tokens, descriptions
    return None
