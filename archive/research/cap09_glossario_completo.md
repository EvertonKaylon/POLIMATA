# 📖 Capítulo 9 — Glossário Completo de Conceitos da IA e Neurociência

> Documento expandido do índice principal `fundamentos_ia_pesquisa.md`

---

## A

**Ação, Potencial de (Action Potential):** Impulso elétrico rápido (~1ms) que viaja pelo axônio quando o limiar de ativação é atingido. Opera no princípio "tudo ou nada" — ou dispara completamente, ou não dispara. Velocidade: 0.5 a 120 m/s.

**Adaline (Adaptive Linear Neuron):** Modelo de neurônio artificial criado por Widrow & Hoff (1960). Diferencia-se do perceptron por usar erro contínuo (antes da ativação) para ajustar pesos, precursor do gradiente descendente.

**AlexNet:** CNN profunda de Krizhevsky, Sutskever & Hinton que venceu o ImageNet 2012 com margem sem precedentes. Marcou o início da revolução do deep learning.

**Atenção, Mecanismo de (Attention):** Técnica que permite ao modelo focar seletivamente em partes relevantes da entrada. Base dos Transformers. Analogia biológica: atenção seletiva do córtex pré-frontal.

**Axônio:** Fibra longa que transmite o potencial de ação do soma para o terminal sináptico. Pode ter até 1 metro de comprimento (neurônios motores).

## B

**Backpropagation:** Algoritmo que calcula o gradiente do erro em relação a cada peso da rede, propagando o erro da saída para a entrada usando a regra da cadeia. Popularizado por Rumelhart, Hinton & Williams (1986).

**Batch Size:** Número de exemplos processados antes de atualizar os pesos. Batch maior = treino mais estável mas mais memória. Batch menor = mais ruído mas menos memória.

**BERT (Bidirectional Encoder Representations from Transformers):** Modelo encoder-only do Google (2018) que compreende texto bidireccionalmente. Revolucionou NLP.

**Bias (Viés):** (1) Na rede neural: parâmetro adicional que desloca a função de ativação, controlando o limiar de disparo. (2) Em dados: preconceito sistemático nos dados de treino que leva a previsões injustas.

**Boole, George:** Matemático inglês (1815-1864) que criou a álgebra booleana — operações AND, OR, NOT sobre 0 e 1. Base de toda a computação digital.

## C

**Cajal, Santiago Ramón y:** Neurocientista espanhol (1852-1934) que demonstrou a Doutrina do Neurônio — células nervosas são unidades discretas. Nobel de 1906.

**CNN (Convolutional Neural Network):** Rede neural com camadas convolucionais que detectam features visuais hierarquicamente. Inspirada pelo córtex visual (Hubel & Wiesel). Usada em visão computacional.

**Cibernética:** Campo fundado por Norbert Wiener (1948) que estuda controle e comunicação em máquinas e seres vivos. Conceitos-chave: feedback, homeostase, informação.

**Conexionismo:** Paradigma que propõe que cognição emerge de redes de unidades simples interconectadas, em oposição à IA simbólica (baseada em regras lógicas).

**Convergência (Teorema de):** Para o perceptron: se os dados são linearmente separáveis, o algoritmo sempre encontra uma solução em número finito de passos.

## D

**Dartmouth, Conferência de (1956):** Workshop que fundou a IA como campo. Organizado por McCarthy, Minsky, Shannon e Rochester. Cunhou o termo "Inteligência Artificial".

**Deep Learning:** Subcampo de machine learning que usa redes neurais com muitas camadas (profundas). Domina IA desde 2012.

**Dendrito:** Extensão ramificada do neurônio que recebe sinais de outros neurônios. Coberto por espinhas dendríticas que formam sinapses.

**Dropout:** Técnica de regularização que desativa neurônios aleatoriamente durante treinamento para prevenir overfitting. Inventada por Hinton et al.

## E

**Embedding:** Representação vetorial densa de dados discretos (palavras, tokens). Ex: Word2Vec, onde palavras com significado similar têm vetores próximos.

**Epoch (Época):** Uma passagem completa por todo o dataset de treinamento. Treinamento tipicamente envolve 2-100+ épocas.

**Excitatória, Sinapse:** Sinapse que aumenta a probabilidade de disparo do neurônio pós-sináptico (usa glutamato). Análogo a peso positivo na IA.

## F

**Feedback (Retroalimentação):** Mecanismo onde a saída de um sistema influencia sua entrada. Negativo: estabiliza (termostato). Positivo: amplifica. Central na cibernética e no aprendizado.

**Fine-Tuning:** Ajuste de um modelo pré-treinado para uma tarefa ou domínio específico. Muito mais barato que treinar do zero.

**Forward Pass:** Fluxo de dados da entrada para a saída da rede, gerando uma previsão. Precede o backward pass (backpropagation).

**Função de Ativação:** Função não-linear aplicada à soma ponderada para decidir a saída do neurônio. Tipos: Step (perceptron), Sigmoid, Tanh, ReLU, GELU.

## G

**GABA (Ácido gama-aminobutírico):** Principal neurotransmissor inibitório do cérebro. Análogo a pesos negativos em redes neurais.

**GAN (Generative Adversarial Network):** Arquitetura com dois modelos competindo — gerador (cria dados) vs discriminador (avalia autenticidade). Criada por Goodfellow (2014).

**Glutamato:** Principal neurotransmissor excitatório do cérebro. Análogo a pesos positivos.

**Gradiente Descendente:** Algoritmo de otimização que ajusta parâmetros na direção que minimiza o erro. É como "descer uma montanha" seguindo a inclinação mais íngreme.

**GPT (Generative Pre-trained Transformer):** Família de LLMs decoder-only da OpenAI. GPT-3 (175B), GPT-4 (~1.7T estimado).

## H

**Hebb, Donald:** Psicólogo canadense (1904-1985) que propôs que sinapses se fortalecem quando neurônios disparam juntos. "Neurons that fire together, wire together."

**Hopfield, John:** Físico que introduziu redes recorrentes com memória associativa (1982). Nobel de Física 2024.

**Hubel & Wiesel:** Neurocientistas que descobriram células simples e complexas no córtex visual (1959-62). Nobel 1981. Inspiraram CNNs.

**Hiperparâmetro:** Parâmetro definido antes do treino (learning rate, batch size, número de camadas) vs parâmetro aprendido (pesos).

## I

**ImageNet:** Dataset de ~14 milhões de imagens rotuladas em ~21.000 categorias. A competição ILSVRC (2010-2017) acelerou o deep learning.

**Inibitória, Sinapse:** Sinapse que diminui a probabilidade de disparo (usa GABA). Análogo a peso negativo.

**Inverno da IA:** Período de estagnação causado por expectativas frustradas e corte de financiamento. Primeiro: ~1969-1982. Segundo: ~1987-1993.

## L

**Learning Rate (Taxa de Aprendizado):** Hiperparâmetro que controla o tamanho do passo no gradiente descendente. Muito alto: instável. Muito baixo: lento.

**LLM (Large Language Model):** Modelo de linguagem com bilhões de parâmetros treinado em trilhões de tokens. Ex: GPT-4, Gemini, Claude, Llama.

**LoRA (Low-Rank Adaptation):** Técnica de fine-tuning eficiente que congela o modelo base e treina apenas adaptadores de rank baixo. Reduz VRAM necessária drasticamente.

**Loss (Perda):** Função que mede o erro entre a previsão do modelo e o valor real. Tipos: MSE, Cross-Entropy, MAE.

**LSTM (Long Short-Term Memory):** Tipo de RNN com portas que controlam fluxo de informação, resolvendo vanishing gradient. Hochreiter & Schmidhuber (1997).

**LTP (Long-Term Potentiation):** Fortalecimento duradouro de uma sinapse após estimulação repetida. Base biológica da memória e aprendizado. Descoberta por Bliss & Lømo (1973).

**LTD (Long-Term Depression):** Enfraquecimento duradouro de uma sinapse. Oposto da LTP. Análogo a diminuir um peso na IA.

## M

**McCulloch-Pitts (Neurônio):** Primeiro modelo matemático de neurônio (1943). Estados binários, função threshold, pesos fixos. Provou universalidade computacional.

**Mielina (Bainha de):** Camada lipídica isolante que envolve axônios, acelerando transmissão de 2 m/s para 120 m/s via condução saltatória.

**Minsky, Marvin:** Co-fundador do MIT AI Lab. Escreveu *Perceptrons* (1969) com Papert, causando o primeiro inverno da IA. Prêmio Turing 1969.

**MLP (Multilayer Perceptron):** Rede neural com uma ou mais camadas ocultas entre entrada e saída. Pode resolver problemas não-linearmente separáveis.

## N

**Neurogênese:** Criação de novos neurônios. Ocorre no hipocampo adulto e bulbo olfatório. Não modelada em IA convencional.

**Neurônio:** Célula do sistema nervoso especializada em receber, processar e transmitir sinais. O cérebro humano tem ~86 bilhões.

**Neurotransmissor:** Molécula química que transmite sinais entre neurônios na sinapse. Tipos: glutamato (excita), GABA (inibe), dopamina (recompensa), serotonina (humor).

## O

**Overfitting:** Quando o modelo memoriza os dados de treino em vez de aprender padrões gerais. Performance alta no treino, baixa em dados novos.

## P

**Perceptron:** Primeiro neurônio artificial com aprendizado automático, criado por Rosenblatt (1958). Só resolve problemas linearmente separáveis.

**Plasticidade Sináptica:** Capacidade das sinapses de mudar sua força. Inclui LTP (fortalecimento) e LTD (enfraquecimento). Base biológica do aprendizado.

**Pooling:** Operação que reduz a dimensão espacial de feature maps em CNNs. Max pooling = pega o valor máximo. Análogo a células complexas de Hubel & Wiesel.

## Q

**QLoRA (Quantized LoRA):** Combinação de LoRA com quantização de 4 bits. Permite fine-tuning de modelos 7B com apenas 6-8 GB de VRAM.

**Quantização:** Redução da precisão numérica dos pesos (float32 → int8 ou int4). Reduz tamanho e VRAM necessária em 4-8x com perda mínima de qualidade.

## R

**ReLU (Rectified Linear Unit):** Função de ativação: f(x) = max(0, x). Padrão em deep learning por eficiência e por evitar vanishing gradient.

**ResNet:** Rede com conexões residuais (skip connections) de Kaiming He et al. (2015). Permitiu treinar redes com 152+ camadas.

**RLHF (Reinforcement Learning from Human Feedback):** Técnica de alinhamento que ajusta LLMs usando avaliações humanas de qualidade de resposta. Usado no ChatGPT.

**Rosenblatt, Frank:** Criador do perceptron (1958). Psicólogo e cientista da computação na Cornell University. Morreu aos 43 anos.

**RNN (Recurrent Neural Network):** Rede com conexões cíclicas que processa sequências. Mantém "memória" de passos anteriores. Substituída por Transformers.

## S

**Separabilidade Linear:** Propriedade de dados que podem ser divididos em classes por uma reta (2D), plano (3D) ou hiperplano (nD). Perceptrons simples só resolvem problemas linearmente separáveis.

**Sigmoid (Sigmoide):** Função de ativação: σ(x) = 1/(1+e⁻ˣ). Saída entre 0 e 1. Usada para probabilidades. Historicamente importante, hoje substituída por ReLU em camadas ocultas.

**Sinapse:** Junção entre dois neurônios onde ocorre transmissão de sinal. Pode ser química (neurotransmissores) ou elétrica (gap junctions).

**SNN (Spiking Neural Network):** Rede neural de terceira geração que processa informação via pulsos discretos no tempo, mais similar ao cérebro biológico.

**Soma (Corpo Celular):** Centro metabólico do neurônio. Contém o núcleo e integra sinais dos dendritos. O axon hillock decide se o neurônio dispara.

**STDP (Spike-Timing Dependent Plasticity):** Forma de plasticidade sináptica onde a mudança de força depende do timing relativo dos disparos pré e pós-sinápticos.

## T

**Transformer:** Arquitetura baseada exclusivamente em mecanismos de atenção. "Attention Is All You Need" (Vaswani et al., 2017). Base de todos os LLMs modernos.

**Turing, Alan:** Matemático britânico (1912-1954). Máquina de Turing (1936), Enigma (WWII), Teste de Turing (1950). Pai da ciência da computação.

**Teste de Turing:** Proposto em 1950: se um interrogador não consegue distinguir respostas de uma máquina das de um humano (por texto), a máquina é "inteligente".

## V

**Vanishing Gradient:** Problema onde gradientes ficam exponencialmente menores em camadas profundas, impedindo aprendizado. Resolvido por ReLU, ResNet, LSTM.

**VRAM (Video RAM):** Memória dedicada da GPU. O principal gargalo para treinar/rodar modelos de IA. Modelos grandes requerem 16-80+ GB.

## W

**Weight (Peso):** Parâmetro numérico que controla a importância de uma conexão entre neurônios. Análogo à força sináptica. Ajustado durante treinamento.

**Wiener, Norbert:** Matemático americano (1894-1964) que fundou a cibernética. Autor de "Cybernetics" (1948). Estabeleceu princípios de feedback e controle.

## X

**XOR (Ou Exclusivo):** Função lógica que retorna verdadeiro quando as entradas são diferentes. Famosamente impossível para perceptrons de camada única. Resolvível com MLPs.

---

> **Nota:** Este glossário é um documento vivo que deve ser atualizado conforme novos conceitos surgem no campo.
