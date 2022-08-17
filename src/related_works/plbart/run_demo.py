
from transformers import PLBartTokenizer, PLBartForConditionalGeneration

device = 'cpu'

tokenizer = PLBartTokenizer.from_pretrained(
    'uclanlp/plbart-java-en_XX', src_lang='java',
    tgt_lang='en_XX')

model = PLBartForConditionalGeneration.from_pretrained(
    'uclanlp/plbart-java-en_XX')

code = 'public int mult ( int x , int y ) { return x * y ; }'

inputs = tokenizer(code, return_tensors='pt', max_length=200, truncation=True)

print(inputs)

translated_tokens = model.generate(
    **inputs, decoder_start_token_id=tokenizer.lang_code_to_id['en_XX'],
    max_length=30, num_beams=10)

print(translated_tokens)

desc = tokenizer.batch_decode(translated_tokens,
                              skip_special_tokens=True)[0]

print(desc)