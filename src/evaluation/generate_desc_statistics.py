import os
import numpy as np

from src.utils import utils


if __name__ == '__main__':

    lang = 'java'
    # lang = 'python'

    # corpus_name = 'huetal'    
    corpus_name = 'codexglue'    
    # corpus_name = 'wanetal'

    systems_dir = f'../../resources/related_works/descriptions/{lang}'

    systems_dir = os.path.join(systems_dir, corpus_name)

    print('\nCorpus:', corpus_name)

    sys_descriptions = utils.read_descriptions(systems_dir)

    all_results = {}

    for cont, (sys_name, descriptions) in enumerate(sys_descriptions.items()):

        print('\nSystem {} / {}: {} - {}'.format(cont + 1, len(sys_descriptions),
                                                 sys_name, len(descriptions)))

        desc_sizes = [len(desc.split(' ')) for desc in descriptions]

        print('\n  Min:', min(desc_sizes))
        print('  Max:', max(desc_sizes))
        print('  Mean:', np.mean(desc_sizes), '~', np.std(desc_sizes))

