import tokenize
import io
import javalang


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def python_tokenizer(line):
    result = []
    line = io.StringIO(line)
    for toktype, tok, start, end, line \
            in tokenize.generate_tokens(line.readline):
        if not toktype == tokenize.COMMENT:
            if toktype == tokenize.STRING:
                result.append('CODE_STRING')
            elif toktype == tokenize.NUMBER:
                result.append('CODE_INTEGER')
            elif (not tok == '\n') and (not tok == '    '):
                result.append(str(tok))
    return ' '.join(result)


def tokenize_java_code(code):
    token_list = []
    tokens = list(javalang.tokenizer.tokenize(code))
    for token in tokens:
        token_list.append(token.value)
    return ' '.join(token_list)


def example_python():

    # model_name = 'SEBIS/code_trans_t5_base_source_code_summarization_python'
    model_name = 'SEBIS/code_trans_t5_small_source_code_summarization_python'

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              skip_special_tokens=True)

    # code = 'def add_tensors(t, t1) -> Any:\n    return t + t1'
    # code = 'def sum(x, y):\n    return x + y'
    code = 'def f(numbers, n):\n  if n not in numbers:\n  numbers.append(n)'

    tokenized_code = python_tokenizer(code)

    print('\nCode after tokenization:', tokenized_code)

    code_seq = tokenizer.encode(tokenized_code,
                                return_tensors='pt',
                                truncation=True,
                                max_length=256).to('cuda')

    desc_ids = model.generate(code_seq,
                              min_length=10,
                              max_length=30,
                              num_beams=10,
                              early_stopping=True)

    description = \
        [tokenizer.decode(g, skip_special_tokens=True,
                          clean_up_tokenization_spaces=True)
         for g in desc_ids]

    description = description[0].strip()

    print('\nDescription:', description)


def example_java():

    model_name = 'SEBIS/code_trans_t5_base_code_comment_generation_java'
    # model_name = 'SEBIS/code_trans_t5_small_code_comment_generation_java'

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              skip_special_tokens=True)

    # code = "protected String renderUri(URI uri){\n  return uri.toASCIIString();\n}\n"
    code = 'public int mult(int x, int y) {\n  return x * y;\n}'

    print('\nCode:', code)

    tokenized_code = tokenize_java_code(code)

    print('\nTokenized code:', tokenized_code)

    code_seq = tokenizer.encode(tokenized_code,
                                return_tensors='pt',
                                truncation=True,
                                max_length=256).to('cuda')

    desc_ids = model.generate(code_seq,
                              min_length=10,
                              max_length=30,
                              num_beams=10,
                              early_stopping=True)

    description = \
        [tokenizer.decode(g, skip_special_tokens=True,
                          clean_up_tokenization_spaces=True)
         for g in desc_ids]

    description = description[0].strip()

    print('\nDescription:', description)


def get_string_from_code(node, lines, code_list):
    line_start = node.start_point[0]
    line_end = node.end_point[0]
    char_start = node.start_point[1]
    char_end = node.end_point[1]
    if line_start != line_end:
        code_list.append(
            ' '.join([lines[line_start][char_start:]] + lines[line_start + 1:line_end] + [lines[line_end][:char_end]]))
    else:
        code_list.append(lines[line_start][char_start:char_end])


def python_traverse(node, code, code_list):
    lines = code.split('\n')
    if node.child_count == 0:
        get_string_from_code(node, lines, code_list)
    elif node.type == 'string':
        get_string_from_code(node, lines, code_list)
    else:
        for n in node.children:
            python_traverse(n, code, code_list)
    return ' '.join(code_list)


def java_traverse(node, code, code_list):
    lines = code.split('\n')
    if node.child_count == 0:
        line_start = node.start_point[0]
        line_end = node.end_point[0]
        char_start = node.start_point[1]
        char_end = node.end_point[1]
        if line_start != line_end:
            code_list.append(' '.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]]))
        else:
            code_list.append(lines[line_start][char_start:char_end])
    else:
        for n in node.children:
            java_traverse(n, code, code_list)
    return ' '.join(code_list)


"""
    https://github.com/agemagician/CodeTrans
"""

if __name__ == '__main__':

    example_python()

    # example_java()
