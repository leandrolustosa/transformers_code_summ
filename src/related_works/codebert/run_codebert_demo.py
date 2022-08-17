import code_bert_utils as utils

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


if __name__ == '__main__':

    config = RobertaConfig.from_pretrained('microsoft/codebert-base')

    model_file = '../../../resources/related_works/models/codebert/pytorch_model.bin'

    tokenizer = \
        RobertaTokenizer.from_pretrained('microsoft/codebert-base',
                                         do_lower_case=False)

    model = utils.build_model(model_class=RobertaModel,
                              model_file=model_file, config=config,
                              tokenizer=tokenizer, max_len=30,
                              beam_size=10,).to('cpu')

    # code = 'def add_tensors(t, t1) -> Any:\n    return t + t1'
    # code = 'def sum(x, y):\n    return x + y'
    # code = 'public int mult(int x, int y) {\n  return x * y;\n}'
    # code = 'def f(numbers, n):\n  if n not in numbers:\n  numbers.append(n)'

    code = 'public int hashcode ( ) { return value . hashcode ( ) ; }'

    # code = """public static double get Similarity(String phrase1, String phrase2) {
    #     return (get SC(phrase1, phrase2) + getSC(phrase2, phrase1)) / 2.0;
    # }"""

    example = [utils.Example(source=code, target=None)]

    features_code = utils.get_features(example, tokenizer,
                                       max_code_len=300)

    description, length = utils.inference(features_code, model, tokenizer)

    print('\nDescription:', description[0])