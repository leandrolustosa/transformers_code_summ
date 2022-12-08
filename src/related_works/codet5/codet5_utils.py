import sys

from tqdm import tqdm


def generate_descriptions(test_codes, tokenizer, model, max_code_len, max_desc_len, num_beams, device):
    total_examples = len(test_codes)
    generated_descriptions = []
    with tqdm(total=total_examples, file=sys.stdout, colour='green', desc='  Generating summaries') as pbar:
        for code in test_codes:
            input_ids = tokenizer.encode(code, return_tensors='pt', max_length=max_code_len, truncation=True)
            input_ids = input_ids.to(device)
            desc_ids = model.generate(input_ids=input_ids, bos_token_id=model.config.bos_token_id,
                                      eos_token_id=model.config.eos_token_id, length_penalty=2.0,
                                      max_length=max_desc_len, num_beams=num_beams)
            desc = tokenizer.decode(desc_ids[0], skip_special_tokens=True)
            generated_descriptions.append(desc)
            pbar.update(1)
    return generated_descriptions
