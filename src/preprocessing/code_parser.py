import re
import javalang
import tokenize
import io
import sys

from tqdm import tqdm
from src.utils.nlp import is_english_text, has_nested_functions


def transform_java_code(code, args):
    comment_regex = re.compile(r'(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', re.MULTILINE)
    space_regex = re.compile(r'[\n\r$\s]+', re.MULTILINE)
    code = comment_regex.sub('', code)
    code = space_regex.sub(' ', code)
    processed_code = process_code_java(code)
    tokens_code = [t for t in processed_code.split()]
    if args.preprocessing == '1':
        tokens_code_aux = [t.replace('_', ' ') for t in tokens_code]
        tokens_code = []
        for token in tokens_code_aux:
            tokens_splited = split_camel_case(token)
            tokens_code.extend(tokens_splited)
    else:
        tokens_code = [t.lower() for t in tokens_code]
    return ' '.join(tokens_code)


def process_code_java(code):
    code = code.replace('\n', ' ').strip()
    tokens_lang = list(javalang.tokenizer.tokenize(code, ignore_errors=True))
    tokens = []
    for token in tokens_lang:
        if token.__class__.__name__ == 'String' or token.__class__.__name__ == 'Character':
            tokens.append('STR_')
        elif 'Integer' in token.__class__.__name__ or 'FloatingPoint' in token.__class__.__name__:
            tokens.append('NUM_')
        elif token.__class__.__name__ == 'Boolean':
            tokens.append('BOOL_')
        else:
            tokens.append(token.value)
    result_code = ' '.join(tokens)
    return result_code


def split_camel_case(token):
    p = re.compile(r'([a-z\d]+)([A-Z])|([A-Z])([a-z\d]+)')
    sub = re.sub(p, r'\1 \2', token).lower().strip()
    if sub == '':
        sub = re.sub(p, r'\3\4 \2', token).lower().strip()
    return sub.split()


def transform_python_code(code, args):
    code = code.strip()
    try:
        processed_code = process_code_python(code)
    except tokenize.TokenError:
        print('\n  ERROR TOKENIZATION.')
        processed_code = code
    tokens_code = processed_code.split()
    if args.preprocessing == '1':
        tokens_code_aux = [t.replace('_', ' ').strip() for t in tokens_code]
        tokens_code = []
        for token_aux in tokens_code_aux:
            for token in token_aux.split():
                tokens_splited = split_camel_case(token)
                tokens_code.extend(tokens_splited)
    else:
        tokens_code = [t.lower() for t in tokens_code]
    code_seq = ' '.join(tokens_code)
    return code_seq


def process_code_python(code):
    result = []
    buffer = io.StringIO(code)
    for toktype, tok, start, end, line in tokenize.generate_tokens(buffer.readline):
        if toktype != tokenize.COMMENT and tok != '\n' and tok != '\t' and tok != '\r\n' and tok.strip() != '':
            if toktype == tokenize.STRING:
                result.append('STR_')
            elif toktype == tokenize.NUMBER:
                result.append('NUM_')
            elif tok == 'True' or tok == 'False':
                result.append('BOOL_')
            elif len(tok.strip()) >= 1:
                result.append(str(tok))
    return ' '.join(result)


def generate_csv_file(codes, descs, args, code_transformer, file_path, msg):
    csv_content = 'code\tdesc\n'
    with tqdm(total=len(codes), file=sys.stdout, desc=msg) as pbar:
        for code, desc in zip(codes, descs):
            code_seq = code_transformer(code, args)
            csv_content += f'{code_seq}\t{desc}\n'
            pbar.update(1)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(csv_content)


def remove_duplicates(codes, descs):
    new_codes = []
    new_descs = []
    dist_codes = set()
    for code, desc in zip(codes, descs):
        t_code = code.lower().replace(' ', '')
        if t_code not in dist_codes:
            new_codes.append(code)
            new_descs.append(desc)
            dist_codes.add(t_code)
    return new_codes, new_descs


def remove_duplicates_from(codes, descs, other_codes):
    dist_codes = []
    for code in other_codes:
        t_code = code.lower().replace(' ', '')
        dist_codes.append(t_code)
    dist_codes = set(dist_codes)
    new_codes = []
    new_descs = []
    for code, desc in zip(codes, descs):
        t_code = code.lower().replace(' ', '')
        if t_code not in dist_codes:
            new_codes.append(code)
            new_descs.append(desc)
    return new_codes, new_descs


def filter_examples(codes, descs, nlp, lang):
    new_codes = []
    new_descs = []
    comment_regex = re.compile(r'(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', re.MULTILINE)
    with tqdm(total=len(codes), file=sys.stdout, desc='  Filtering') as pbar:
        for code, desc in zip(codes, descs):
            doc = nlp(desc)
            words = [t.text for t in doc if len(t.text.strip()) >= 1]
            space_regex = re.compile(r'[\n\r$\s]+', re.MULTILINE)
            if lang == 'java':
                code = comment_regex.sub('', code)
                code = space_regex.sub(' ', code)
                processed_code = process_code_java(code)
            else:
                try:
                    processed_code = process_code_python(code)
                    desc = desc.replace(':return: ', 'Return ')
                except IndentationError:
                    pbar.update(1)
                    continue
            if not(is_english_text(code) and is_english_text(desc)):
                pbar.update(1)
                continue
            if lang == 'python' and has_nested_functions(code):
                pbar.update(1)
                continue
            tokens_code = [t for t in processed_code.split()]
            if 4 <= len(words) <= 100 and 5 <= len(tokens_code) <= 300:
                new_codes.append(code)
                new_descs.append(desc)
            pbar.update(1)
    return new_codes, new_descs


def split_descriptions_by_period(descs):
    for i in range(len(descs)):
        if descs[i].startswith('.'):
            descs[i] = descs[i][1:len(descs[i])]
        if '@' in descs[i]:
            descs[i] = descs[i].split('@')[0]
        if descs[i].count('.') > 1:
            aux = descs[i].split('. ')[0].strip()
            if len(aux.split(' ')) >= 4:
                descs[i] = aux
        if descs[i][-1] != '.':
            descs[i] += '.'
