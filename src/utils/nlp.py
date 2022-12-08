import re
import string
import sys
import javalang
import spacy
import os

from nltk.stem import PorterStemmer
from math import log
from unicodedata import normalize
from tokenize import tokenize
from io import BytesIO
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import constants as const

# https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).

if os.path.exists('../../resources/nlp/words-by-frequency.txt'):
    words = open('../../resources/nlp/words-by-frequency.txt').read().split()
    wordcost = dict((k, log((i+1)*log(len(words)))) for i, k in enumerate(words))
    maxword = max(len(x) for x in words)


def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return list(reversed(out))


def get_abbreviations():
    if os.path.exists('../../resources/nlp/abbreviations.txt'):
        with open('../../resources/nlp/abbreviations.txt') as fin:
            rows = (line.split('\t') for line in fin)
            d = {row[0]: row[1].replace('\n', '') for row in rows}
        return d


abbreviations = get_abbreviations()


spacy.prefer_gpu()

nlp = spacy.load('en_core_web_sm')


def clean_java_data(codes, descriptions):
    cleaned_data = []
    for i, code in enumerate(codes):
        tokens_generator = javalang.tokenizer.tokenize(code)
        tokens_code = [t.value for t in tokens_generator]
        tokens_code = filter_tokens(tokens_code)
        desc = descriptions[i]
        tokens_desc = tokenize_text(desc)
        cleaned_data.append((tokens_code, tokens_desc))
        if (i+1) % 100 == 0:
            sys.stdout.write('  ' + str(i+1))
    return cleaned_data


def clean_python_data(codes, descriptions, threshold=5):
    data = []
    all_tokens_code = []
    all_tokens_desc = []
    for i, (code, desc) in enumerate(zip(codes, descriptions)):
        if not(is_english_text(code) and is_english_text(desc)) \
                or starts_with_annotation(code) or has_nested_functions(code):
            continue

        code = code.replace('_', ' ')
        tokens_generator = tokenize(BytesIO(code.encode('utf-8')).readline)
        tokens_code = [value for _, value, _, _, _ in tokens_generator if value != 'utf-8']
        tokens_code = camel_case_split(tokens_code)
        desc = desc.replace('_', ' ')
        tokens_desc = tokenize_text(desc)
        tokens_desc = camel_case_split(tokens_desc)
        tokens_code = filter_tokens(tokens_code)
        tokens_desc = filter_tokens(tokens_desc)
        tokens_code_size = len(tokens_code)
        tokens_desc_size = len(tokens_desc)
        if const.CODE_MIN_PYTHON > tokens_code_size or tokens_code_size > const.CODE_MAX_PYTHON:
            continue
        if const.DESC_MIN_PYTHON > tokens_desc_size or tokens_desc_size > const.DESC_MAX_PYTHON:
            continue
        all_tokens_code.extend(tokens_code)
        all_tokens_desc.extend(tokens_desc)
        data.append((tokens_code, tokens_desc))
        if (i + 1) % 1000 == 0:
            sys.stdout.write('  ' + str(i + 1))
    counter_code = Counter(all_tokens_code)
    counter_desc = Counter(all_tokens_desc)
    cleaned_data = []
    for d in data:
        tokens_code = d[0]
        tokens_desc = d[1]
        tokens_code = [t for t in tokens_code if counter_code[t] >= threshold]
        tokens_desc = [t for t in tokens_desc if counter_desc[t] >= threshold]
        cleaned_data.append((tokens_code, tokens_desc))
    return cleaned_data


def get_java_tokens_code(code, args, threshold=5):

    if not is_english_text(code) \
            or starts_with_annotation(code) or has_nested_functions(code):
        return []

    code = code.replace(const.DCNL, ' ').replace(const.DCSP, ' ').replace('  ', ' ')
    code = replace_literals(code)
    code = replace_2_annotation(code)
    code = code.replace('_', ' ')
    tokens_generator = tokenize(BytesIO(code.encode('utf-8')).readline)
    tokens_code = [value for _, value, _, _, _ in tokens_generator if value != 'utf-8']
    tokens_code = camel_case_split(tokens_code)
    replace_abbreviations(tokens_code)
    tokens_code = get_tokens_from_infer_spaces(tokens_code)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit_transform([" ".join(tokens_code)])
    tokens_code = tfidf.vocabulary_
    weights_matrix = tfidf.idf_

    return tokens_code, weights_matrix


def get_tokens_code(code, threshold=5):

    if not is_english_text(code) \
            or starts_with_annotation(code) or has_nested_functions(code):
        return []

    code = code.replace(const.DCNL, ' ').replace(const.DCSP, ' ').replace('  ', ' ')
    code = replace_literals(code)
    code = replace_2_annotation(code)
    code = code.replace('_', ' ')
    tokens_generator = tokenize(BytesIO(code.encode('utf-8')).readline)
    tokens_code = [value for _, value, _, _, _ in tokens_generator if value != 'utf-8']
    tokens_code = camel_case_split(tokens_code)
    replace_abbreviations(tokens_code)
    #tokens_code = get_tokens_from_infer_spaces(tokens_code)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit_transform([" ".join(tokens_code)])
    tokens_code = tfidf.vocabulary_
    weights_matrix = tfidf.idf_

    return tokens_code, weights_matrix


def tokenize_text(text):
    text_treated = text.replace(':return: ', 'Return ')

    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    re_print = re.compile('[^%s]]' % re.escape(string.printable))

    text_treated = text_treated.split(const.DCNL)[0]
    normalized_text = normalize('NFD', text_treated).encode('ascii', 'ignore')
    normalized_text = normalized_text.decode('UTF-8')
    tokens = normalized_text.split()
    tokens = [t.lower() for t in tokens]
    tokens = [re_punc.sub('', t) for t in tokens]
    tokens = [re_print.sub('', t) for t in tokens]
    tokens = [t for t in tokens if t.isalpha()]
    return tokens


def filter_tokens(tokens):
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    re_print = re.compile('[^%s]]' % re.escape(string.printable))
    tokens = [t.lower() for t in tokens]
    tokens = [re_punc.sub('', t) for t in tokens]
    tokens = [re_print.sub('', t) for t in tokens]
    tokens = [t for t in tokens if t.isalpha()]
    return tokens


def camel_case_split(tokens):
    new_tokens = []
    for token in tokens:
        fragments = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', token)
        if len(fragments) > 1:
            new_tokens.extend(fragments)
        else:
            new_tokens.append(token)
    return new_tokens


def extract_keywords(descriptions):
    nlp = spacy.load("en_core_web_sm",
                     disable=['parser', 'ner', 'textcat'])
    all_keywords = []
    #stop_words = ['return', 'use', 'be', 'get', 'give', 'function', 'string', 'object', 'test']
    for description in descriptions:
        description = description.replace('_', '')
        doc = nlp(description)
        desc_keywords = set([t.lemma_.lower() for t in doc if len(t) > 1
                         #and t.lemma_.lower() not in stop_words
                         and (t.pos_ == 'NOUN' or t.pos_ == 'VERB')])
        all_keywords.append(desc_keywords)
    return all_keywords


def get_keywords(tokens):
    #nlp = spacy.load("en_core_web_sm",
    #                 disable=['parser', 'ner', 'textcat'])
    #all_keywords = []
    #for token in tokens:
    #    doc = nlp(token)
    #    keywords = set([t.lemma_.lower() for t in doc if len(t) > 1])
    #    all_keywords.append(keywords)
    porter = PorterStemmer()
    all_keywords = []
    for token in tokens:
        if token == "def" or token == "lambda":
            continue
        token_stem = porter.stem(token)
        if token_stem not in all_keywords:
            all_keywords.append(token_stem)

    return all_keywords


def is_english_text(text):
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def starts_with_annotation(text):
    matches = re.findall(r'^@[.]*', text)
    return len(matches) >= 1


def has_nested_functions(text):
    matches = re.findall(r'def(.*?):', text)
    return len(matches) > 1


def replace_2_annotation(text):
    code = text
    matches = re.findall(r'[\w]+2[\w]+', text)
    for match in matches:
        code = code.replace(match, match.replace('2', '_to_'))
    return code


def replace_literals(text):
    code = text
    matches = re.findall(r'([\"][\w.\#\%\d\-_\(\)\'\:\.\s]*[\"]|[\'][\w.\#\%\d\-_\(\)\'\:\.\s]*[\'])', text)
    for match in matches:
        code = code.replace(match, '')
    return code


def replace_abbreviations(tokens):
    for i, token in enumerate(tokens):
        if token in abbreviations:
            tokens[i] = token.replace(token, abbreviations[token])


def get_tokens_from_infer_spaces(tokens):
    code_tokens = []
    sentence = " ".join(tokens)
    sp_tokens = nlp(sentence)
    for i, token in enumerate(sp_tokens):
        if not token.is_oov:
            code_tokens.append(token.text)
        else:
            infered_tokens = infer_spaces(token.text)
            code_tokens.extend(infered_tokens)
    return code_tokens
