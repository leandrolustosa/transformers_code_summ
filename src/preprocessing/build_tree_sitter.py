from tree_sitter import Language
import os

"""
    git clone https://github.com/tree-sitter/tree-sitter-python
    git clone https://github.com/tree-sitter/tree-sitter-java
"""

if __name__ == '__main__':

    print('\nBuilding languages ...')

    Language.build_library('languages/code_languages.so',
                           ['../../tree-sitter-python',
                            '../../tree-sitter-java'])




    print('\nLanguages built...')