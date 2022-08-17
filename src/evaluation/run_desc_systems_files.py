import os
import json
import sys

from src.utils import utils
from src.evaluation.evaluation_measures import compute_rouge, \
    compute_bleu, compute_meteor
from tqdm import tqdm


if __name__ == '__main__':

    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    corpus_name = 'codexglue'
    # corpus_name = 'wanetal'

    # kind = 'related_works'
    kind = 'fine_tuning'

    systems_dir = f'../../resources/{kind}/descriptions/{lang}/{corpus_name}'

    results_dir = f'../../resources/{kind}/results/{lang}/{corpus_name}'

    size_threshold = -1

    max_desc_len = 30

    test_file_path = f'../../resources/corpora/{lang}/{corpus_name}/csv/test_none.csv'

    _, _, test_data = utils.read_corpus_csv(test_file_path=test_file_path,
                                            sample_size=size_threshold)

    os.makedirs(results_dir, exist_ok=True)

    sys_descriptions = utils.read_descriptions(systems_dir)

    codes = test_data[0]
    descriptions = test_data[1]

    print('\nCorpus:', corpus_name, '-', len(codes), '-', len(descriptions))

    print('\nSystems:', len(sys_descriptions), '\n')

    data = []

    with tqdm(total=len(codes), file=sys.stdout, desc='  Evaluating ') as pbar:

        systems_descs_csv = ['Id\tSistema\tCode\tRef.Desc.\tDesc.\trouge-1\trouge-2\trouge-l\tmeteor\tbleu-4']
        for i, (code, ref_desc) in enumerate(zip(codes, descriptions)):

            # print('\nID:', i+1)
            # print('  Code:', code)
            # print('  Desc:', ref_desc)

            dict_example = {
                'id': i+1,
                'code': code,
                'ref_desc': ref_desc
            }

            ref_desc_tokens = ref_desc.split(' ')

            systems_descs = []

            # print('\n')

            for sys_name in sys_descriptions.keys():

                sys_desc = sys_descriptions[sys_name][i]

                system_desc = str(i + 1)
                system_desc += '\t' + sys_name
                system_desc += '\t' + code
                system_desc += '\t' + ref_desc
                system_desc += '\t' + sys_desc

                # print('\n    System:', sys_name)
                # print('      Desc:', sys_desc)

                tokens_desc = sys_desc.split(' ')

                if len(tokens_desc) > max_desc_len:
                    tokens_desc = tokens_desc[:max_desc_len]
                    sys_desc = ' '.join(tokens_desc).strip()

                rouge_scores = compute_rouge(sys_desc, ref_desc, max_desc_len)

                bleu_4 = compute_bleu(sys_desc, ref_desc)

                meteor_score = compute_meteor(tokens_desc, ref_desc_tokens)

                measures = {}

                for rouge_n, metrics in rouge_scores.items():
                    system_desc += '\t' + str(metrics['f'])
                    for metric, value in metrics.items():
                        metric_name = rouge_n.replace('-', '') + \
                                      '_' + metric
                        measures[metric_name] = value

                measures['meteor_score'] = meteor_score
                system_desc += '\t' + str(meteor_score)

                measures['bleu_4'] = bleu_4
                system_desc += '\t' + str(bleu_4)

                system_dict = {
                    'name': sys_name,
                    'desc': sys_desc,
                    'measures': measures
                }

                systems_descs.append(system_dict)
                systems_descs_csv.append(system_desc)

            dict_example['systems'] = systems_descs

            data.append(dict_example)

            pbar.update(1)

    json_path = os.path.join(results_dir, corpus_name + '.json')
    csv_path = os.path.join(results_dir, corpus_name + '.csv')

    with open(json_path, 'w') as file:
        json.dump(data, file)

    with open(csv_path, 'w') as file:
        file.write('\n'.join(systems_descs_csv))
