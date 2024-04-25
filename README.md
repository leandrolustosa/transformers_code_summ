# Investigação de Modelos Neurais Baseados na Arquitetura *Transformer* para Sumarização Automática de Código-Fonte.

## Dados Gerais

Repositório para disponibilização do código-fonte do programa que foi desenvolvido para executar as tarefas da Investigação de Modelos Neurais Baseados na Arquitetura *Transformer* para Sumarização Automática de Código-Fonte.

Este protótipo foi construido para ser utilizado como apoio na execução dos experimentos necessários para a dissertação apresentada ao [Programa de Pós-Graduação em Computação Aplicada do Instituto Federal do Espírito Santo (IFES)](https://www.ifes.edu.br/cursos/pos-graduacao/mestrado-em-computacao-aplicada), como requisito parcial para obtenção do título de Mestre em Computação Aplicada.

Aluno: [Leandro Baêta Lustosa Pontes](http://lattes.cnpq.br/2529360031927429)  
Orientador: [Prof. Dr. Hilário Tomaz Alves de Oliveira](http://lattes.cnpq.br/8980213630090119)  
Repositório do artigo publicado: "[Avaliação de Modelos Neurais para Sumarização de Código-Fonte](https://github.com/laicsiifes/code_summarization)"  

> Leia a [dissertação](https://repositorio.ifes.edu.br/handle/123456789/2993) disponibilizada no repositório do IFES, para entender o contexto em que o programa foi utilizado.  
> Código testado no Linux com [Python 3.10.4](https://www.python.org/downloads/release/python-3104/) (Release Date: Apr 2, 2022).  
> Código testado no Windows com [Python 3.10.5](https://www.python.org/downloads/release/python-3105/) (Release Date: June 6, 2022).  
> Consulte os '[requirements](https://github.com/leandrolustosa/transformers_code_summ/blob/main/requirements.txt)' para saber as versões de todas as bibliotecas usadas.  

## Organização do código-fonte

```
-- resources - Pastas contendo os corpora utilizados no projeto
---- corpora/** - Pastas com códigos-fonte e descrições de referência
---- fine_tuning/** - Pastas com descrições geradas pelos modelos com ajuste fino
---- related_works/** - Pastas com descrições geradas pelos modelos pré-treinados
-- src - Pastas contendo o código-fonte
---- corpora - Código-fonte para dividir os Corpora em Corpus de Treinamento, Teste e Validação
---- evaluation - Código-fonte para gerar as medidas BLEU, ROUGE e METEOR, entre outras classes utilitárias para auxiliar a tarefa de avaliação dos resultados dos modelos
---- fine_tuning - Modelos treinados com os *Corpora* de Teste, em uma fase adicional de ajuste fino (*fine tuning*), para validar a hipótese se essa etapa adicionar poderia melhorar as medidas avaliadas
---- preprocessing - Código-fonte para a etapa de pré-processamento dos *Corpora*, remoção de registros duplicados, separação de palavras por espaço em branco ou convenções de programação, como Camel Case e Snake Case, entre outros processamentos
---- related_works - Modelos pré-treinados pelos autores originais de cada um, servindo de Linha Base para esse trabalho
---- util - Código-fonte de classes utilitárias, utilizadas por todos os demais módulos
requirements.txt - Arquivo com as dependências de módulos Python utilizados nesse trabalho.
```
