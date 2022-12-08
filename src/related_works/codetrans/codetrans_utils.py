import sys

from tqdm import tqdm


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
            code_list.append(' '.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] +
                                      [lines[line_end][:char_end]]))
        else:
            code_list.append(lines[line_start][char_start:char_end])
    else:
        for n in node.children:
            java_traverse(n, code, code_list)
    return ' '.join(code_list)


def generate_descriptions(test_codes, tokenizer, model, max_code_len, max_len_desc, num_beams, device):
    total_examples = len(test_codes)
    generated_descriptions = []
    with tqdm(total=total_examples, file=sys.stdout, colour='green', desc='  Generating summaries') as pbar:
        for i in range(total_examples):
            code = test_codes[i]
            code_seq = tokenizer.encode(code, return_tensors='pt', truncation=True,
                                        max_length=max_code_len).to(device)
            desc_ids = model.generate(code_seq, max_length=max_len_desc, num_beams=num_beams,
                                      early_stopping=True)
            description = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for g in desc_ids]
            description = description[0].strip()
            generated_descriptions.append(description)
            pbar.update(1)
    return generated_descriptions
