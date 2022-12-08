import sys

from tqdm import tqdm


def generate_descriptions(test_codes, tokenizer, model, max_code_len, max_desc_len, num_beams, start_token_id,
                          device):
    total_examples = len(test_codes)
    generated_descriptions = []
    with tqdm(total=total_examples, file=sys.stdout, colour='green', desc='  Generating summaries') as pbar:
        for code in test_codes:
            input_ids = tokenizer(code, return_tensors='pt', max_length=max_code_len, truncation=True)
            input_ids = input_ids.to(device)
            translated_tokens = model.generate(**input_ids, max_length=max_desc_len, num_beams=num_beams,
                                               decoder_start_token_id=start_token_id)
            desc = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            generated_descriptions.append(desc)
            pbar.update(1)
    return generated_descriptions
