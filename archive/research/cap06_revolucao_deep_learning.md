# 🚀 Capítulo 6 — A Revolução do Deep Learning

> Documento expandido do índice principal `fundamentos_ia_pesquisa.md`

---

## 1. O Gatilho: AlexNet e o ImageNet (2012)

### 1.1 O Desafio ImageNet (ILSVRC)

O **ImageNet Large Scale Visual Recognition Challenge** era a competição mais importante de visão computacional:
- **Dataset:** ~1.2 milhões de imagens de treino, 1.000 categorias
- **Tarefa:** Classificar corretamente o objeto principal da imagem
- **Métrica:** Top-5 error rate (a resposta correta deve estar entre as 5 melhores previsões)

**Resultados antes de 2012:** Os melhores sistemas usavam features engineered manualmente (SIFT, HOG) + classificadores como SVM. Error rate: ~26%.

### 1.2 AlexNet — A Ruptura

**Equipe:** Alex Krizhevsky (estudante), Ilya Sutskever (pós-doc), Geoffrey Hinton (orientador) — University of Toronto.

**Resultados:**
- **Error rate: 15.3%** — redução de ~10 pontos percentuais sobre o segundo lugar
- Segundo lugar: 26.2% (usando métodos tradicionais)
- A margem de vitória foi **chocante** — nunca vista na história da competição

**Arquitetura:**
```
Input (224×224×3 - RGB)
  → Conv1: 96 filtros 11×11, stride 4 → ReLU → MaxPool → Normalização
    → Conv2: 256 filtros 5×5 → ReLU → MaxPool → Normalização
      → Conv3: 384 filtros 3×3 → ReLU
        → Conv4: 384 filtros 3×3 → ReLU
          → Conv5: 256 filtros 3×3 → ReLU → MaxPool
            → FC1: 4096 → Dropout(0.5) → ReLU
              → FC2: 4096 → Dropout(0.5) → ReLU
                → FC3: 1000 (softmax) → output
```

**~60 milhões de parâmetros, treinado em 2 GPUs NVIDIA GTX 580 (3GB VRAM cada).**

### 1.3 Inovações Técnicas do AlexNet

| Inovação | Descrição | Impacto |
|---|---|---|
| **ReLU** | Usou ReLU em vez de sigmoid/tanh | 6x mais rápido de treinar |
| **GPU Training** | Treinou em GPUs (CUDA) | Viabilizou redes profundas |
| **Dropout** | Desligava neurônios aleatoriamente durante treino | Reduziu overfitting |
| **Data Augmentation** | Rotações, reflexões, recortes aleatórios | Mais dados virtuais |
| **Overlapping Pooling** | Pooling com overlap | Ligeira melhora de acurácia |

### 1.4 O Efeito Dominó

Após AlexNet, TODOS os vencedores do ImageNet usaram deep learning:

| Ano | Modelo | Camadas | Top-5 Error | Inovação |
|---|---|---|---|---|
| 2012 | **AlexNet** | 8 | 15.3% | GPUs + ReLU + Dropout |
| 2013 | **ZFNet** | 8 | 11.2% | Visualização de features |
| 2014 | **GoogLeNet** | 22 | 6.7% | Inception modules |
| 2014 | **VGGNet** | 19 | 7.3% | Filtros 3×3 empilhados |
| 2015 | **ResNet** | 152 | 3.6% | Conexões residuais |
| 2017 | **SENet** | 152+ | 2.3% | Atenção por canal |
| Humano | — | — | ~5.1% | — |

**Em 2015, ResNet SUPEROU a acurácia humana!**

---

## 2. ResNet — Redes Residuais (2015)

### 2.1 O Problema da Profundidade

Teoricamente, redes mais profundas deveriam ser melhores. Na prática, redes com mais de ~20 camadas tinham performance PIOR que redes menores — o **problema da degradação**.

Causa: **vanishing/exploding gradients** — gradientes ficam exponencialmente menores (ou maiores) ao passar por muitas camadas.

### 2.2 A Solução Elegante: Skip Connections

**Kaiming He et al.** propuseram conexões residuais:

```
Em vez de:  x → Camadas → F(x)

Fazer:      x → Camadas → F(x) + x ← conexão residual (skip)
```

A rede aprende o **resíduo** F(x) = H(x) - x, em vez da transformação completa H(x). Se a transformação ideal for identidade (nenhuma mudança), basta que F(x) = 0, que é muito mais fácil de aprender.

**Resultado:** Redes com **152 camadas** que treinam melhor que redes de 20 camadas. Hoje existem redes com milhares de camadas.

---

## 3. GANs — Redes Adversárias Generativas (2014)

### 3.1 A Ideia de Ian Goodfellow

Goodfellow concebeu as GANs numa noite de bar em Montreal (literalmente):

**Duas redes competindo:**
- **Gerador (G):** Tenta criar dados falsos convincentes (ex: imagens de rostos)
- **Discriminador (D):** Tenta distinguir dados reais de falsos

```
Ruído → [Gerador] → Imagem Fake → [Discriminador] → Real ou Fake?
                                        ↑
                   Imagens Reais ────────┘
```

**Analogia:** Falsificador de dinheiro (G) vs detetive (D). Ambos melhoram com o tempo — o falsificador faz notas cada vez melhores, o detetive fica cada vez mais perspicaz.

### 3.2 Evolução das GANs

| Modelo | Ano | Resolução | Inovação |
|---|---|---|---|
| GAN original | 2014 | 64×64, borrada | Conceito base |
| DCGAN | 2015 | 64×64, melhor | Conv layers no gerador |
| Progressive GAN | 2018 | 1024×1024 | Crescimento progressivo |
| StyleGAN | 2019 | 1024×1024, fotorrealista | Controle de estilo por camada |
| StyleGAN 3 | 2021 | 1024×1024+ | Aliasing-free |

---

## 4. Transformers — A Revolução de 2017

### 4.1 O Problema das RNNs

Antes dos Transformers, processamento de linguagem usava **RNNs/LSTMs**:
- Processamento **sequencial** (palavra por palavra) — lento, não paralelizável
- Dificuldade com **dependências de longo alcance** (esquecem o início de textos longos)
- Treinamento lento (cada passo depende do anterior)

### 4.2 "Attention Is All You Need" (Vaswani et al., 2017)

Paper de pesquisadores do **Google Brain** que introduziu a arquitetura Transformer:

**A inovação central: Self-Attention (Auto-Atenção)**

Para cada palavra em uma frase, o mecanismo de atenção calcula quão "relevante" cada outra palavra é:

```
Frase: "O gato sentou no tapete porque ele estava cansado"

Para a palavra "ele":
  "gato"    → peso de atenção ALTO (referente provável)
  "tapete"  → peso de atenção BAIXO
  "cansado" → peso de atenção MÉDIO
  "sentou"  → peso de atenção MÉDIO
```

**Mecanismo matemático (Query, Key, Value):**
```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

Q = Query (o que estou procurando?)
K = Key (o que eu ofereço?)
V = Value (qual é meu conteúdo?)
d_k = dimensão das keys (para normalização)
```

### 4.3 Componentes do Transformer

```
Encoder:                          Decoder:
┌──────────────────────┐         ┌──────────────────────┐
│ Multi-Head Attention │         │ Masked Multi-Head    │
│          +           │         │ Attention + Residual │
│ Add & LayerNorm      │         ├──────────────────────┤
├──────────────────────┤         │ Cross-Attention      │
│ Feed-Forward Network │         │ (encoder → decoder)  │
│          +           │         ├──────────────────────┤
│ Add & LayerNorm      │         │ Feed-Forward + Norm  │
└──────────────────────┘         └──────────────────────┘
     (× N camadas)                    (× N camadas)
```

### 4.4 Multi-Head Attention

Em vez de uma única "visão" de atenção, usa-se múltiplas "cabeças" em paralelo:
- Cada cabeça pode aprender a prestar atenção a diferentes aspectos (sintaxe, semântica, coreference, etc.)
- Os resultados das cabeças são concatenados e projetados

### 4.5 Positional Encoding

Como o Transformer processa tudo em paralelo (sem ordem sequencial), precisa de **codificação posicional** para saber a ordem das palavras:
- Funções seno e cosseno com diferentes frequências
- Versões mais modernas: RoPE (Rotary Positional Embeddings)

### 4.6 A Família de Modelos

| Tipo | Estrutura | Exemplo | Uso |
|---|---|---|---|
| **Encoder-only** | Só encoder | BERT | Compreensão (classificação, NER) |
| **Decoder-only** | Só decoder | GPT-1/2/3/4 | Geração de texto |
| **Encoder-Decoder** | Ambos | T5, BART | Tradução, sumarização |

### 4.7 Conexão Neurocientífica

O mecanismo de atenção tem paralelos com a **atenção biológica**:

| Conceito no Transformer | Análogo Biológico |
|---|---|
| Self-attention (foco seletivo) | Atenção seletiva (córtex pré-frontal filtra estímulos) |
| Multi-head attention | Processamento paralelo em diferentes áreas corticais |
| Positional encoding | Codificação temporal no hipocampo (place cells, grid cells) |
| Softmax (competição entre pesos) | Competição inibitória lateral entre neurônios |
| Residual connections | Conexões diretas entre camadas corticais (bypass) |

---

## 5. Large Language Models (LLMs) — A Era Atual

### 5.1 A Escalada

| Modelo | Ano | Parâmetros | Dados de Treino | Custo Estimado |
|---|---|---|---|---|
| GPT-1 | 2018 | 117M | BookCorpus | ~$50K |
| GPT-2 | 2019 | 1.5B | WebText (40GB) | ~$250K |
| GPT-3 | 2020 | 175B | 499B tokens | ~$4.6M |
| PaLM | 2022 | 540B | 780B tokens | ~$10M+ |
| GPT-4 | 2023 | ~1.7T (estimado) | Não revelado | ~$100M |
| Gemini Ultra | 2024 | Não revelado | Não revelado | ~$100M+ |

### 5.2 Emergent Abilities — Capacidades Emergentes

À medida que modelos crescem, habilidades **novas** surgem de forma não-linear:
- **Few-shot learning:** Aprender tarefas com poucos exemplos no prompt
- **Chain-of-thought:** Raciocínio passo a passo
- **Código:** Geração e debug de programas
- **Raciocínio aritmético:** Contas complexas (com erros, mas melhorando)

### 5.3 RLHF (Reinforcement Learning from Human Feedback)

O segredo do ChatGPT: após pré-treino, o modelo é **alinhado** com preferências humanas:
1. Humanos avaliam respostas (A é melhor que B?)
2. Um modelo de recompensa é treinado com essas avaliações
3. O LLM é ajustado via aprendizado por reforço (PPO) para maximizar a recompensa

---

## 6. Computação Neuromórfica — O Futuro Bio-Inspirado

### 6.1 Redes Neurais de Pulsos (SNNs)

SNNs são a terceira geração de redes neurais, mais próximas da biologia:

| Geração | Modelo | Sinal | Exemplo |
|---|---|---|---|
| 1ª | Perceptron | Binário (0/1) | McCulloch-Pitts, Rosenblatt |
| 2ª | Redes contínuas | Valores reais contínuos | MLP, CNN, Transformer |
| 3ª | **SNNs** | **Pulsos discretos no tempo** | Modelos LIF, Izhikevich |

**Vantagem teórica:** Codificam informação no **timing** dos pulsos (como neurônios reais), não apenas na magnitude.

### 6.2 Hardware Neuromórfico

| Chip | Criador | Ano | Neurônios | Sinapses | Consumo |
|---|---|---|---|---|---|
| **TrueNorth** | IBM | 2014 | 1M | 256M | 70 mW |
| **Loihi** | Intel | 2017 | 128K | 128M | ~30 mW |
| **Loihi 2** | Intel | 2021 | 1M | 120M | ~1 W |
| **SpiNNaker 2** | Univ. Manchester | 2023 | 10M+ | Bilhões | Watts |
| **Cérebro humano** | Natureza | — | 86B | 100-500T | 20 W |

### 6.3 O Projeto Human Brain

O **Human Brain Project** (UE, 2013-2023, €1 bilhão) e o **BRAIN Initiative** (EUA, 2013+) buscam mapear e simular o cérebro humano. A convergência entre neurociência e IA está acelerando:

- **Neurociência → IA:** Inspirações biológicas para novas arquiteturas
- **IA → Neurociência:** Modelos de deep learning ajudam a analisar dados cerebrais (fMRI, EEG, registros neurais)
- **Objetivo de longo prazo:** Compreensão mútua que beneficie tanto o tratamento de doenças neurológicas quanto o desenvolvimento de IA mais eficiente

---

> **Referências:** Krizhevsky, Sutskever & Hinton "ImageNet Classification" (2012), He et al. "ResNet" (2015), Goodfellow et al. "GANs" (2014), Vaswani et al. "Attention Is All You Need" (2017), Brown et al. "GPT-3" (2020), Merolla et al. "TrueNorth" (2014).
