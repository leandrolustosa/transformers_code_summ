import os
import argparse
import spacy

from src.utils import utils
from code_parser import transform_python_code, transform_java_code, remove_duplicates, \
    remove_duplicates_from, filter_examples, split_descriptions_by_period, generate_csv_file


if __name__ == '__main__':

    # Initialize parser

    parser = argparse.ArgumentParser()

    # Adding optional argument

    parser.add_argument('-p', '--preprocessing', default='1', const='0', nargs='?', choices=['0', '1'],
                        help='Replace the name of elements to generic names. [0-None; '
                             '1-Camel Case and Snake Case]')

    parser.add_argument('-l', '--language', default='p', const='p', nargs='?', choices=['j', 'p'],
                        help='Choose the language. [j-Java; p-Python]')

    parser.add_argument('-c', '--corpus', default='codexglue', const='codexglue', nargs='?',
                        choices=['huetal', 'codexglue', 'wanetal'],
                        help='Choose the corpus. [HuEtal; CodexGlue; WanEtal]')

    # Read arguments from command line

    args = parser.parse_args()

    preprocessing = 'none' if args.preprocessing == '0' else 'camelsnakecase'

    lang = 'java' if args.language == 'j' else 'python'

    corpus = None
    code_transformer = None

    corpus_dir = f'../../resources/corpora/{lang}/{args.corpus}/original'
    csv_corpus_dir = f'../../resources/corpora/{lang}/{args.corpus}/csv'

    if lang == 'python':
        code_transformer = transform_python_code
        if args.corpus == 'codexglue':
            corpus = utils.read_corpus(corpus_dir)
        elif args.corpus == 'wanetal':
            corpus = utils.read_python_corpus(corpus_dir)
    elif lang == 'java':
        code_transformer = transform_java_code
        if args.corpus == 'huetal':
            corpus = utils.read_java_corpus(corpus_dir)
        elif args.corpus == 'codexglue':
            corpus = utils.read_corpus(corpus_dir)
    else:
        print(f'\n\nERROR. {args.lang} INVALID!')
        exit(-1)

    if corpus is None:
        print(f'\n\nERROR. {args.corpus} INVALID!')
        exit(-1)

    print(f'\nLanguage: {lang}')
    
    print(f'\nCorpus: {args.corpus}')

    print(f'\nProcessing: {preprocessing}')

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

    os.makedirs(csv_corpus_dir, exist_ok=True)

    train_codes = corpus['train_code']
    train_descs = corpus['train_desc']

    valid_codes = corpus['valid_code']
    valid_desc = corpus['valid_desc']

    test_codes = corpus['test_code']
    test_desc = corpus['test_desc']

    print('\nOriginal corpus ...')

    print('\n  Train size:', len(train_codes), '-', len(train_descs))
    print('  Valid size:', len(valid_codes), '-', len(valid_desc))
    print('  Test size:', len(test_codes), '-', len(test_desc))

    print('\nRemoving duplicates inside the subsets ...')

    train_codes, train_descs = remove_duplicates(train_codes, train_descs)
    valid_codes, valid_desc = remove_duplicates(valid_codes, valid_desc)
    test_codes, test_desc = remove_duplicates(test_codes, test_desc)

    print('\n  Train size:', len(train_codes), '-', len(train_descs))
    print('  Valid size:', len(valid_codes), '-', len(valid_desc))
    print('  Test size:', len(test_codes), '-', len(test_desc))

    print('\nRemoving duplicates among the subsets ...')

    train_codes, train_descs = remove_duplicates_from(train_codes, train_descs, valid_codes)
    train_codes, train_descs = remove_duplicates_from(train_codes, train_descs, test_codes)
    valid_codes, valid_desc = remove_duplicates_from(valid_codes, valid_desc, test_codes)

    print('\n  Train size:', len(train_codes), '-', len(train_descs))
    print('  Valid size:', len(valid_codes), '-', len(valid_desc))
    print('  Test size:', len(test_codes), '-', len(test_desc))

    print('\nRemoving examples ...\n')

    test_codes, test_desc = filter_examples(test_codes, test_desc, nlp, lang)
    valid_codes, valid_desc = filter_examples(valid_codes, valid_desc, nlp, lang)
    train_codes, train_descs = filter_examples(train_codes, train_descs, nlp, lang)

    print('\n  Train size:', len(train_codes), '-', len(train_descs))
    print('  Valid size:', len(valid_codes), '-', len(valid_desc))
    print('  Test size:', len(test_codes), '-', len(test_desc))

    split_descriptions_by_period(train_descs)
    split_descriptions_by_period(valid_desc)
    split_descriptions_by_period(test_desc)

    train_path = os.path.join(csv_corpus_dir, f'train_{preprocessing}.csv')
    valid_path = os.path.join(csv_corpus_dir, f'valid_{preprocessing}.csv')
    test_path = os.path.join(csv_corpus_dir, f'test_{preprocessing}.csv')

    print('\nGenerating CSV Files\n')

    generate_csv_file(test_codes, test_desc, args, code_transformer, test_path,
                      msg='  Processing Test Codes')

    print()

    generate_csv_file(valid_codes, valid_desc, args, code_transformer, valid_path,
                      msg='  Processing Validation Codes')

    print()

    generate_csv_file(train_codes, train_descs, args, code_transformer, train_path,
                      msg='  Processing Train Codes')

    print('\nFiles Generated\n')
