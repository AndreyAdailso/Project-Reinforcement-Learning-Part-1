# Projeto parte 1 da Disciplina de Reinforcement Learning

## Descrição

Esse repositorio é uma atividade realizanda durante a disciplica Advanced Deep Learning - PPGEE - UFPA.

## Descrição da atividade

Cada estudante deve usar seu próprio ambiente FMDP personalizado.

1. Encontre o v* (valores de estado) ótimo, q* (valores de ação) e política . Descreva a
política de uma maneira que possamos entender facilmente seu significado. Olhe para o
ambiente verbose_kd_env.py, que fornece suporte para descrever ações e
estados usando strings em vez de inteiros.

2. Compare um agente executando sua política ótima do Problema 1 com o Q-learning treinado com 8 conjuntos diferentes de hiperparâmetros, que você deve escolher (use valores sensatos):

* Escolha 2 valores distintos para o fator de desconto

* Escolha 2 valores distintos para alpha ("taxa de aprendizado")

* Escolha 2 valores distintos para o número máximo de episódios de treinamento

Escolha as melhores 3 combinações entre os 8 agentes treinados e compare seu desempenho. Para essa comparação, congele o treinamento do agente (não o treine mais) e verifique o retorno médio para vários episódios de teste, disjuntos dos episódios de treinamento. O gráfico deve ser o retorno acumulado (cumsum) G de cada episódio, e a abscissa é o índice do episódio. Use 100 episódios de teste (mostre as 3 curvas junto com a curva do agente de política ótima). Discuta seus resultados.

3. Em relação ao treinamento dos 8 agentes, compare sua convergência (de todas as alternativas de hiperparâmetros do q-learning que você tentou). Neste caso, a abscissa é o número de iteração I e você pode escolher uma figura de mérito para representar o desempenho ao longo do treinamento na iteração i. Discuta o resultado.

4. Agora vamos usar redes neurais (NNs). Escolha uma configuração de hiperparâmetros (você pode usar o padrão se desejar) para o algoritmo DQN e tente o DQN usando 2 topologias de redes neurais diferentes. Em seguida, trace as 2 curvas de complexidade de amostragem, uma para cada "agente" DQN (topologia de NN) e discuta o resultado. Organize todo o seu código como um repositório público do GitHub e informe a URL no seu relatório.

## Instalação

Para executar este projeto, siga as etapas abaixo:

1. Clone este repositório em sua máquina local.
2. Instale as dependências necessárias executando o seguinte comando:

    ```bash
    apt install numpy
    ```

## Uso

Para executar o projeto, utilize o seguinte comando:


