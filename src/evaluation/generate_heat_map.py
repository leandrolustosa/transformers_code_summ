import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import locale

languages = ['java', 'python']
models = {
    'java': ['codexglue', 'huetal'],
    'python': ['codexglue', 'wanetal']
}

if __name__ == "__main__":

    locale.setlocale(locale.LC_ALL, locale='pt_BR.UTF-8')

    for language in languages:

        for model in models[language]:

            similaridades = pd.read_csv(f'../../resources/corpora/{language}/similarity/{model}/similaraty_{model}_by_jaccard.csv', sep=';')
            similaridades['Similaridade'] = similaridades['Similaridade'].fillna(0.0)

            cols = similaridades[similaridades['Modelo 1'] == similaridades['Modelo 1'][0]]['Modelo 2'].tolist()[:12]
            subs = similaridades[similaridades['Modelo 1'] == similaridades['Modelo 1'][0]]['SubModelo 2'].tolist()[:12]
            arrays = [cols, subs]
            tuples = list(zip((*arrays)))

            similaridades = similaridades.pivot(index=('Modelo 1', 'SubModelo 1'), columns=('Modelo 2', 'SubModelo 2'))
            indexes = pd.MultiIndex.from_tuples(tuples, names=["-", "CSC"])

            similaridades = similaridades.reindex(indexes)

            sns.set_theme(style="white")

            # Generate a mask for the lower triangle
            mask = np.tril(np.ones_like(similaridades, dtype=bool))

            # Set up the matplotlib figure
            f, ax = plt.subplots(figsize=(12, 7))

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 20, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            ax = sns.heatmap(similaridades, mask=mask, cmap=cmap, annot=True,
                             vmax=1, vmin=0, center=.5, #robust=True,
                        annot_kws={'fontsize': 'xx-small'}, fmt='.4n',
                        linewidths=.5,
                        cbar_kws={"shrink": .5},
                        xticklabels=['#' if i==0 else f'{t[0]} - {t[1]}' if i%2==0 else t[1] for i, t in enumerate(tuples)],
                        yticklabels=[f'{t[0]} - {t[1]}' if i%2==0 else t[1] for i, t in enumerate(tuples[:-1])])

            cbar = ax.collections[0].colorbar
            cbar.set_ticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
            cbar.set_ticklabels(['0,0', '0,1', '0,2', '0,3', '0,4', '0,5', '0,6', '0,7', '0,8', '0,9', '1,0'])

            plt.savefig(f'../../resources/corpora/images/similaraty_{language}_{model}_by_jaccard.png',
                        dpi=250.0)