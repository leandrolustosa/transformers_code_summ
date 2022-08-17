import torch

from transformers import T5Config, RobertaTokenizer, T5ForConditionalGeneration


if __name__ == '__main__':

    model_dir = '../../resources/models/codet5/best_model/codet5_model.bin'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

    model_config = T5Config.from_pretrained('Salesforce/codet5-base-multi-sum')

    model = T5ForConditionalGeneration(config=model_config)

    model = model.to(device)

    model.load_state_dict(torch.load(model_dir))

    codes = []

    code = 'public int mult(int x, int y) {\n  return x * y;\n}'

    codes.append(code)

    code = "protected String renderUri(URI uri){\n  return uri.toASCIIString();\n}\n"

    codes.append(code)

    code = '''
    public static void copyFile( File in, File out )
                throws IOException
        {
            FileChannel inChannel = new FileInputStream( in ).getChannel();
            FileChannel outChannel = new FileOutputStream( out ).getChannel();
            try
            {
    //          inChannel.transferTo(0, inChannel.size(), outChannel);      // original -- apparently has trouble copying large files on Windows

                // magic number for Windows, 64Mb - 32Kb)
                int maxCount = (64 * 1024 * 1024) - (32 * 1024);
                long size = inChannel.size();
                long position = 0;
                while ( position < size )
                {
                   position += inChannel.transferTo( position, maxCount, outChannel );
                }
            }
            finally
            {
                if ( inChannel != null )
                {
                   inChannel.close();
                }
                if ( outChannel != null )
                {
                    outChannel.close();
                }
            }
        }
        '''

    codes.append(code)

    code = """
        private static int exitWithStatus(int status) {
            if (ToolIO.getMode() == ToolIO.SYSTEM){
                System.exit(status);
            }
            return status;
        }
    """

    codes.append(code)

    code = """
        public boolean search(List<Integer> numbers, int q) {
            boolean found = false;
            for(int n: numbers)
                if( n == q ) {
                    found = true;
                    break;
                }
            return found;
        }
    """

    codes.append(code)

    code = 'public int add(int x, int y) { return x + y; }'

    codes.append(code)

    code = """public static double getSimilarity(String phrase1, String phrase2) {
        return (getSC(phrase1, phrase2) + getSC(phrase2, phrase1)) / 2.0;
    }"""

    codes.append(code)

    for code in codes:

        print('\nCode:', code.replace('\n', ' '))

        input_ids = tokenizer(code, add_special_tokens=True, return_tensors='pt').input_ids

        input_ids = input_ids.to(device)

        generated_ids = model.generate(input_ids, max_length=30, min_length=10, num_beams=10)

        desc = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print('  Description:', desc)

