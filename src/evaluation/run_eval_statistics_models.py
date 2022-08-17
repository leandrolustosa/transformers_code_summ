import pandas as pd

if __name__ == "__main__":
    # lang = 'java'
    lang = 'python'

    # corpus_name = 'huetal'
    # corpus_name = 'codexglue'
    corpus_name = 'wanetal'

    # kind = 'related_works'
    kind = 'fine_tuning'

    results_dir = f'../../resources/{kind}/results/{lang}/{corpus_name}/{corpus_name}.csv'

    result = pd.read_csv(results_dir, sep='\t')

    sistemas = result["Sistema"].unique()

    for sistema in sorted(sistemas):
        print("\n\n"+sistema)
        model_result = result[result["Sistema"]==sistema]
        print(model_result[["rouge-l", "meteor", "bleu-4"]].describe(percentiles=[.25, .5, .75, .80, .85, .9, .95]))