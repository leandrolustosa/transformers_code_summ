from transformers import RobertaTokenizer, T5ForConditionalGeneration


"""
    https://github.com/salesforce/CodeT5
"""

if __name__ == '__main__':

    model_name_tok = 'Salesforce/codet5-base'
    model_name = 'Salesforce/codet5-base-multi-sum'

    tokenizer = RobertaTokenizer.from_pretrained(model_name_tok)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # code = "def greet(user): print(f'hello <extra_id_0>!')"
    # code = 'def add_tensors(t, t1) -> Any:    return t + t1'
    # code = 'def sum(x, y):\n    return x + y'
    # code = 'def f(numbers, n):\n  if n not in numbers:\n  numbers.append(n)'
    # code = "protected String renderUri(URI uri){\n  return uri.toASCIIString();\n}\n"
    code = 'public int mult(int x, int y) {\n  return x * y;\n}'

    input_ids = tokenizer(code, return_tensors='pt').input_ids

    generated_ids = model.generate(input_ids, max_length=30)

    desc = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print('\nDescription:', desc)

