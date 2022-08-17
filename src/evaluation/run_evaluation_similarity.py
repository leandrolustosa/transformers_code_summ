import os
import sys
import spacy
import statistics
import csv
from tqdm import tqdm

from scipy import spatial
from sentence_transformers import SentenceTransformer
from itertools import combinations
from src.utils import nlp as nlp_utils

#language = 'java'
language = 'python'

#datasets = ['codexglue', 'huetal']
datasets = ['codexglue', 'wanetal']
#datasets = ['codexglue']
#datasets = ['huetal']

models = ['codebert',
          'codet5',
          'plbart',
          'codetrans_mt_tf_small',
          'codetrans_mt_tf_base',
          'codetrans_mt_tf_large'
          ]

title_models = ['CodeBERT',
          'CodeT5',
          'PLBART',
          '$CodeTrans_{Small}$',
          '$CodeTrans_{Base}$',
          '$CodeTrans_{Large}$'
          ]

title_csv_models = ['CodeBERT',
          'CodeT5',
          'PLBART',
          'CodeTrans Small',
          'CodeTrans Base',
          'CodeTrans Large'
          ]

preprocessing_types = ['none', 'camelsnakecase']

similarity_folder_path = f'../../resources/corpora/{language}/similarity'
summarizations_folder_path = f'../../resources/related_works/descriptions/{language}'

limit_rows = -1


def convert_words_to_number(words_sets):
    words_set1, words_set2 = words_sets
    total_words = words_set1 | words_set2
    vocabulary = {word: i for i, word in enumerate(sorted(total_words))}
    value_to_word = {value: word for word, value in vocabulary.items()}

    return value_to_word, vocabulary


def get_binary_vector(vocabulary, summarization):
    vector = []
    for word, index in vocabulary.items():
        vector.append(1 if word in summarization else 0)

    return vector


def get_jaccard_sim(str1, str2):
    a = set(str1)
    b = set(str2)

    c = a.intersection(b)
    d = a.union(b)

    if len(d) == 0:
        return 1

    return len(c) / len(d)


def get_cosine_sim(row_1, row_2):
    global nlp

    words_list = [row_1, row_2]
    words_set = [set(words) for words in words_list]
    value_to_word, vocabulary = convert_words_to_number(words_set)

    vector_1 = get_binary_vector(vocabulary, row_1)
    vector_2 = get_binary_vector(vocabulary, row_2)

    return 1 - spatial.distance.cosine(vector_1, vector_2)


def get_sbert_cosine_sim(row_1, row_2):
    global sbert_model

    sentences = [
        ' '.join(row_1),
        ' '.join(row_2),
    ]
    embeddings = sbert_model.encode(sentences)

    return 1 - spatial.distance.cosine(embeddings[0], embeddings[1])


def define_global_variables(dataset):
    global files
    global summarizations_by_index

    files = []
    summarizations_by_index = {}
    summarizations = {}

    for i, model in enumerate(models):
        for j, preprocessing_type in enumerate(preprocessing_types):
            file_path = f'pre_{model}_{dataset}_{preprocessing_type}.txt'
            index = (i * 2) + j
            method = f'{model}_{preprocessing_type}'

            files.append(file_path)
            summarizations_by_index[index] = method
            summarizations[method] = []

    return summarizations


def read_and_preprocessing_summarizations(dataset, summarizations):
    global files
    global summarizations_by_index
    global limit_rows

    for index, file_path in enumerate(files):
        method = summarizations_by_index[index]

        rows = []
        with open(os.path.join(summarizations_folder_path, dataset, file_path), 'r', encoding='utf-8') as file:
            file_rows = file.readlines()[:limit_rows]
            with tqdm(total=len(file_rows), file=sys.stdout,
                      desc=f'Reading summarizations {method}') as pbar:
                for text in file_rows:
                    words = []
                    t = nlp_utils.tokenize_text(text)
                    words.extend([token.lemma_ for token in nlp(' '.join(t)) if not token.is_stop])
                    rows.append(words)
                    pbar.update(1)

        summarizations[method] = rows

        with open(os.path.join(similarity_folder_path, dataset, file_path), 'w+', encoding='utf-8') as file:
            file.write('\n'.join([' '.join(row) for row in rows]))

    return summarizations


def read_summarizations(dataset, summarizations):
    global files
    global summarizations_by_index
    global limit_rows

    for index, file_path in enumerate(files):
        method = summarizations_by_index[index]

        rows = []
        with open(os.path.join(similarity_folder_path, dataset, file_path), 'r', encoding='utf-8') as file:
            file_rows = file.readlines()[:limit_rows]
            with tqdm(total=len(file_rows), file=sys.stdout,
                      desc=f'Reading summarizations {method}') as pbar:
                for text in file_rows:
                    words = []
                    words.extend([word for word in text.replace('\n', '').split(' ')])
                    rows.append(words)
                    pbar.update(1)

        summarizations[method] = rows

    return summarizations


def get_methods(comb):
    global summarizations_by_index

    return summarizations_by_index[comb[0]], summarizations_by_index[comb[1]]


def calc_sim(func, similarities, comparisons, _texts, combination):
    for comb in combination:
        method_1, method_2 = get_methods(comb)

        file_path_similarity = f'{method_1}-{method_2}'
        comparisons.append(file_path_similarity)

        results = []
        with tqdm(total=len(_texts[method_1]), file=sys.stdout,
                  desc=f'Calculating similarity for {file_path_similarity}') as pbar:
            for row_1, row_2 in zip(_texts[method_1], _texts[method_2]):
                similarity = func(row_1, row_2)

                results.append(similarity)

                pbar.update(1)

        similarities[file_path_similarity] = results


def write_result(dataset, similarities, comparisons, by):
    with open(os.path.join(similarity_folder_path, dataset, f'similaraty_{by}.txt'),
              'w+', encoding='utf-8') as file:
        file.write('\t'.join(header for header in comparisons) + '\n')

        comb_1 = similarities[comparisons[0]]
        comb_2 = similarities[comparisons[1]]
        comb_3 = similarities[comparisons[2]]
        comb_4 = similarities[comparisons[3]]
        comb_5 = similarities[comparisons[4]]
        comb_6 = similarities[comparisons[5]]

        file.write('\n'.join(['\t'.join(map(str, row)) for row in
                              zip(comb_1, comb_2, comb_3, comb_4, comb_5, comb_6)]))


def calc_agg(agg, similarities):
    keys = similarities.keys()
    with tqdm(total=len(keys), file=sys.stdout,
              desc=f'Calculating aggregates avg / stdev') as pbar:

        for key, values in similarities.items():
            agg[key] = {'avg': statistics.mean(values), 'stdev': statistics.stdev(values)}
            methods = key.split('-')
            agg[f'{methods[1]}-{methods[0]}'] = agg[key]

            pbar.update(1)


def write_similaraty_matrix(dataset, agg, titles, by, print_devpd=True, print_latex=False):
    global files

    sufix = ''
    if print_devpd:
        sufix = '_devpd'

    if print_latex:
        sufix += '_latex'

    with open(os.path.join(similarity_folder_path, dataset, f'similaraty_matrix_{dataset}_{by}{sufix}.txt'),
              'w+', encoding='utf-8') as file:

        header = ['Modelos', 'Pré']

        if not print_latex:
            header.extend(titles)
            file.write('\t'.join(header) + '\n')
        else:
            for title_model in title_models:
                header.append(f'\\multicolumn{{2}}{{c|}}{{{title_model}}}')
            file.write(' & '.join(header) + ' \\\\\n\\rowcolor{lightgray}\n\\hline\n')

        matrix = []
        for i in range(0, len(files)):
            if i == len(files) - 1:
                continue

            rest = i % 2
            if rest == 0:
                matrix.append(['-'])
            else:
                matrix.append(['CSC'])

        for i, method_1 in enumerate(titles):
            for j, method_2 in enumerate(titles):
                if j == 0 or i == len(titles) - 1:
                    continue

                if i >= j:
                    matrix[i].append(' ')
                else:
                    comparison = f'{method_1}-{method_2}'
                    if print_devpd:
                        matrix[i].append(f'{round(agg[comparison]["stdev"],4):.4f}'.replace('.', ','))
                    else:
                        matrix[i].append(f'{round(agg[comparison]["avg"], 4):.4f}'.replace('.', ','))

        for i, title in enumerate(titles):
            if i == len(titles) - 1:
                continue

            if not print_latex:
                row = [title]
                row.extend(matrix[i])
                file.write('\t'.join(row_text for row_text in row) + '\n')
            else:
                if i == 0:
                    subheader = ['', 'Proc.']
                    for subheader_index in range(0, len(files)):
                        if subheader_index == 0:
                            continue

                        rest = subheader_index % 2
                        if rest == 0:
                            subheader.append('-')
                        else:
                            subheader.append('CSC')
                    file.write(' & '.join(subheader) + ' \\\\\n\\hline\n')

                rest = i % 2
                title_index = int(i / 2)
                if rest == 0:
                    row = [f'\\multirow{{2}}{{*}}{{{title_models[title_index]}}}']
                else:
                    row = [f'\\cline{{2-13}}']
                row.extend(matrix[i])
                file.write(' & '.join(row_text for row_text in row) + ' \\\\\n' + ('\\hline\n' if rest == 1 else ''))


def write_similaraty_csv(dataset, agg, titles, by, print_devpd=True):
    global files

    sufix = ''
    if print_devpd:
        sufix = '_devpd'

    with open(os.path.join(similarity_folder_path, dataset, f'similaraty_{dataset}_{by}{sufix}.csv'),
              'w+', encoding='utf-8') as file:
        csv_writer = csv.writer(file, lineterminator='\n', delimiter=';')

        header = ['Modelo 1', 'SubModelo 1', 'Modelo 2', 'SubModelo 2', 'Similaridade']

        csv_writer.writerow(header)

        for i, method_1 in enumerate(titles):
            for j, method_2 in enumerate(titles):
                model_1_index = int(i / 2)
                model_2_index = int(j / 2)
                model_1 = title_csv_models[model_1_index]
                model_2 = title_csv_models[model_2_index]

                rest_1 = i % 2
                subheader_1 = '-'
                if rest_1 != 0:
                    subheader_1 = 'CSC'

                rest_2 = j % 2
                subheader_2 = '-'
                if rest_2 != 0:
                    subheader_2 = 'CSC'

                if (i==0 and j==0) or (i==j):
                    csv_writer.writerow([model_1, subheader_1, model_2, subheader_2, '-1.0'])
                    continue

                comparison = f'{method_1}-{method_2}'

                value = f'{round(agg[comparison]["avg"], 4):.4f}'
                if print_devpd:
                    value = f'{round(agg[comparison]["stdev"],4):.4f}'
                csv_writer.writerow([model_1, subheader_1, model_2, subheader_2, value])

def calculate_similarity(dataset, texts, calculate_sim, by, print_devpd=True, print_latex=False):
    global files
    global summarizations_by_index

    combination = list(combinations(list(range(0, len(files))), 2))

    similarities = {}
    agg = {}
    comparisons = []

    calc_sim(calculate_sim, similarities, comparisons, texts, combination)

    #write_result(similarities, comparisons, by)d=True

    calc_agg(agg, similarities)

    write_similaraty_matrix(dataset, agg, texts.keys(), by, print_devpd, print_latex)

    write_similaraty_csv(dataset, agg, texts.keys(), by, print_devpd)


if __name__ == '__main__':
    global nlp
    global sbert_model

    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_sm')

    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    for corpus in datasets:

        summarizations = define_global_variables(corpus)

        summarizations = read_and_preprocessing_summarizations(corpus, summarizations)
        #summarizations = read_summarizations(corpus, summarizations)

        #calculate_similarity(corpus, summarizations, get_sbert_cosine_sim, 'by_sbert_cosine')
        calculate_similarity(corpus, summarizations, get_jaccard_sim, 'by_jaccard', print_devpd=False, print_latex=True)
