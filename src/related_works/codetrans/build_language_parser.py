import os

from tree_sitter import Language


if __name__ == '__main__':

    os.makedirs('../../../resources/languages/', exist_ok=True)

    Language.build_library('../../../resources/languages/code_languages.so',
                           ['/home/lab902/Documentos/Hilario/Projetos/code_summarization/tree-sitter-java'])