import codebert_utils as utils
import torch.nn as nn
import torch

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from model import Seq2Seq


if __name__ == '__main__':

    model_file = '../../resources/models/codebert/best_model/codebert_model.bin'

    # model_file = '../../resources/related_works/models/codebert/pytorch_model.bin'

    beam_size = 10

    max_source_len = 256
    max_target_len = 40

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = \
        RobertaTokenizer.from_pretrained('microsoft/codebert-base',
                                         do_lower_case=False)

    # Build model

    config = RobertaConfig.from_pretrained('microsoft/codebert-base')

    print('\n', config)

    encoder = RobertaModel.from_pretrained('microsoft/codebert-base',
                                           config=config)

    decoder_layer = \
        nn.TransformerDecoderLayer(d_model=config.hidden_size,
                                   nhead=config.num_attention_heads)

    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=beam_size, max_length=max_target_len,
                    sos_id=tokenizer.cls_token_id,
                    eos_id=tokenizer.sep_token_id)

    model.load_state_dict(
        torch.load(model_file, map_location=torch.device(device)),
        strict=False)

    model.to(device)

    # code = 'def add_tensors(t, t1) -> Any: return t + t1'
    # code = 'def sum(x, y): return x + y'
    # code = 'def f(numbers, n): if n not in numbers:  numbers.append(n)'

    # code = 'public int mult(int x, int y) { return x * y; }'
    # code = 'public int hashcode ( ) { return value . hashcode ( ) ; }'

    # code = """public static double getSimilarity(String phrase1, String phrase2) {
    #     return (getSC(phrase1, phrase2) + getSC(phrase2, phrase1)) / 2.0;
    # }"""

    # code = """
    #      public boolean search(List<Integer> numbers, int q) {
    #          boolean found = false;
    #          for(int n: numbers)
    #              if( n == q ) {
    #                  found = true;
    #                  break;
    #              }
    #          return found;
    #      }
    #  """

    code = 'public int add(int x, int y) { return x + y; }'

    example = [utils.Example(idx=None, source=code, target=None)]

    features_code = utils.get_features(example, max_source_len,
                                       max_target_len, tokenizer)

    description, length = utils.inference(features_code, model,
                                          tokenizer, device)

    print('Description:', description[0])
