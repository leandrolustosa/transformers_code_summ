from rouge import Rouge
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

"""
    https://github.com/Diego999/py-rouge
"""


def compute_rouge(reference, candidate, max_len, use_stemming=True, max_ngram=2):
    evaluator = Rouge(metrics=['rouge-n', 'rouge-l'], max_n=max_ngram,
                      limit_length=True, length_limit=max_len,
                      length_limit_type='words', apply_avg=True,
                      apply_best=False, alpha=0.5, weight_factor=1.0,
                      stemming=use_stemming)
    rouge_scores = evaluator.get_scores(candidate, reference)
    return rouge_scores


def compute_bleu(reference, candidate):
    smooth = SmoothingFunction()
    return sentence_bleu([reference.split()], candidate.split(),
                         smoothing_function=smooth.method2)

# def compute_bleu(reference, candidate):
#     return bleu([reference], candidate, ground=0,
#                 smooth=1)[0]


def compute_meteor(reference, candidate):
    return single_meteor_score(reference, candidate)
