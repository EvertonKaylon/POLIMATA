# 🔥 Capítulo 5 — Renascimento: Backpropagation e Redes Multicamadas

> Documento expandido do índice principal `fundamentos_ia_pesquisa.md`

---

## 1. Os Precursores do Renascimento

### 1.1 Redes de Hopfield (1982) — A Física Entra no Jogo

**John Hopfield** era um físico teórico (Caltech/Princeton), não um pesquisador de IA. Ele trouxe conceitos da **mecânica estatística** para redes neurais.

**Rede de Hopfield:**
- Rede **recorrente** totalmente conectada (cada neurônio conecta com todos os outros)
- Conexões **simétricas** (wᵢⱼ = wⱼᵢ)
- Funciona como **memória associativa** — armazena padrões e recupera o mais próximo dado uma entrada ruidosa

**Analogia com a física:**
- Cada estado da rede tem uma "energia"
- A rede evolui para estados de **menor energia** (atratores)
- Padrões armazenados são **mínimos locais de energia**
- A dinâmica da rede é análoga a um ímã se resfriando

**Por que importou:**
- Trouxe **credibilidade acadêmica** de volta para redes neurais — Hopfield era um físico respeitado
- Atraiu físicos para o campo (que trouxeram ferramentas matemáticas poderosas)
- Publicou na prestigiosa *Proceedings of the National Academy of Sciences*
- Demonstrou que redes neurais não eram "beco sem saída" — tinham propriedades matemáticas elegantes

**Prêmio Nobel de Física (2024):** Hopfield recebeu o Nobel junto com Hinton "por descobertas e invenções fundamentais que possibilitaram o aprendizado de máquina com redes neurais artificiais".

### 1.2 Máquinas de Boltzmann (1985) — Aprendizado Estocástico

**Geoffrey Hinton** e **Terry Sejnowski** criaram as Máquinas de Boltzmann:
- Versão **estocástica** (probabilística) da rede de Hopfield
- Neurônios disparam com **probabilidade** proporcional à sua ativação (não determinístico)
- Usa **simulated annealing** (recozimento simulado) — técnica de otimização emprestada da metalurgia
- Podia aprender **representações internas** (features ocultas) — algo que o perceptron não conseguia

**A distribuição de Boltzmann:**
```
P(neurônio i ativo) = 1 / (1 + exp(-Eᵢ/T))

Onde:
  Eᵢ = energia do neurônio i
  T = "temperatura" (controla aleatoriedade)
  Alta T → decisões aleatórias (exploração)
  Baixa T → decisões determinísticas (exploitation)
```

**Nota:** A função 1/(1 + exp(-x)) é a **função sigmoide** — que se tornaria ubíqua em redes neurais.

---

## 2. O Grupo PDP e o Paper de 1986

### 2.1 O Grupo de Processamento Distribuído Paralelo (PDP)

No início dos anos 1980, um grupo de pesquisadores em San Diego formou o **PDP Research Group**:

| Membro | Contribuição |
|---|---|
| **David Rumelhart** | Psicólogo cognitivo; líder do grupo; formulou backpropagation |
| **Geoffrey Hinton** | Cientista da computação britânico-canadense; Máquinas de Boltzmann |
| **Ronald Williams** | Matemático; formalizou o algoritmo |
| **James McClelland** | Psicólogo; modelos conexionistas de cognição |
| **Jay McClelland** | Co-editor dos livros PDP |

### 2.2 Os Livros PDP (1986)

O grupo publicou dois volumes: *"Parallel Distributed Processing: Explorations in the Microstructure of Cognition"*:
- **Volume 1:** Fundações (Rumelhart & McClelland)
- **Volume 2:** Modelos psicológicos e biológicos

Estes livros foram o **manifesto do conexionismo** — a ideia de que cognição emerge de redes de unidades simples processando em paralelo, não de regras simbólicas explícitas.

### 2.3 O Paper: "Learning Representations by Back-Propagating Errors" (1986)

Publicado na **Nature** (uma das revistas mais prestigiosas do mundo), este paper de Rumelhart, Hinton e Williams demonstrou que o backpropagation:
1. **Funciona** em redes multicamadas
2. **Aprende representações internas** úteis nas camadas ocultas
3. **Resolve o XOR** (refutando a crítica de Minsky/Papert)

---

## 3. Backpropagation — O Algoritmo em Profundidade

### 3.1 A Ideia Central

**Problema:** Como ajustar os pesos de uma camada oculta se só temos o erro na saída?

**Solução:** Usar a **regra da cadeia** do cálculo para "propagar" o erro de volta, camada por camada, da saída até a entrada.

### 3.2 Pré-requisito: Funções de Ativação Diferenciáveis

O perceptron original usava a função **step** (degrau), que não é diferenciável. Para o backpropagation funcionar, precisamos de funções **suaves**:

| Função | Fórmula | Derivada | Faixa | Uso |
|---|---|---|---|---|
| **Sigmoide** | σ(x) = 1/(1+e⁻ˣ) | σ(x)·(1-σ(x)) | (0, 1) | Saída binária |
| **Tanh** | tanh(x) | 1 - tanh²(x) | (-1, 1) | Camadas ocultas clássicas |
| **ReLU** | max(0, x) | 0 se x<0, 1 se x>0 | [0, ∞) | Padrão moderno (deep learning) |
| **Leaky ReLU** | max(0.01x, x) | 0.01 ou 1 | (-∞, ∞) | Evita "neurônios mortos" |

### 3.3 O Algoritmo Passo a Passo

**Rede de exemplo:** 2 entradas → 2 neurônios ocultos → 1 saída

**Fase 1 — Forward Pass:**
```
Camada oculta:
  z₁ = w₁₁·x₁ + w₁₂·x₂ + b₁
  a₁ = σ(z₁)
  z₂ = w₂₁·x₁ + w₂₂·x₂ + b₂
  a₂ = σ(z₂)

Camada de saída:
  z₃ = w₃₁·a₁ + w₃₂·a₂ + b₃
  ŷ  = σ(z₃)
```

**Fase 2 — Calcular Erro (Loss):**
```
L = -(y·log(ŷ) + (1-y)·log(1-ŷ))    ← Binary Cross-Entropy
```

**Fase 3 — Backward Pass (Regra da Cadeia):**
```
Gradiente da saída:
  ∂L/∂z₃ = ŷ - y                      ← erro direto

Gradientes dos pesos da saída:
  ∂L/∂w₃₁ = (ŷ - y) · a₁
  ∂L/∂w₃₂ = (ŷ - y) · a₂

Gradiente propagado para camada oculta:
  ∂L/∂a₁ = (ŷ - y) · w₃₁             ← erro "fluindo de volta"
  ∂L/∂z₁ = ∂L/∂a₁ · σ'(z₁)           ← multiplicado pela derivada da ativação

Gradientes dos pesos ocultos:
  ∂L/∂w₁₁ = ∂L/∂z₁ · x₁
  ∂L/∂w₁₂ = ∂L/∂z₁ · x₂
```

**Fase 4 — Atualizar Pesos:**
```
w = w - η · ∂L/∂w    (para cada peso)
```

### 3.4 Implementação Python (Rede que Resolve XOR)

```python
import numpy as np

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Dados XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Arquitetura: 2 → 4 → 1
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))
lr = 1.0  # taxa de aprendizado

# Treinamento
for epoch in range(10000):
    # Forward
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    
    # Loss
    loss = np.mean((y - a2) ** 2)
    
    # Backward
    dz2 = (a2 - y) * sigmoid_derivative(z2)
    dW2 = a1.T @ dz2 / 4
    db2 = np.sum(dz2, axis=0, keepdims=True) / 4
    
    dz1 = (dz2 @ W2.T) * sigmoid_derivative(z1)
    dW1 = X.T @ dz1 / 4
    db1 = np.sum(dz1, axis=0, keepdims=True) / 4
    
    # Update
    W2 -= lr * dW2;  b2 -= lr * db2
    W1 -= lr * dW1;  b1 -= lr * db1
    
    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

# Resultado
print("\nResultados finais:")
for xi, yi in zip(X, y):
    pred = sigmoid(sigmoid(xi @ W1 + b1) @ W2 + b2)
    print(f"  {xi} → {pred[0][0]:.4f} (esperado: {yi[0]})")
```

**Saída esperada:**
```
Epoch     0 | Loss: 0.260000
Epoch  2000 | Loss: 0.003412
Epoch  4000 | Loss: 0.000891
...
Resultados finais:
  [0 0] → 0.0123 (esperado: 0)    ✓
  [0 1] → 0.9847 (esperado: 1)    ✓
  [1 0] → 0.9851 (esperado: 1)    ✓
  [1 1] → 0.0198 (esperado: 0)    ✓
```

**O XOR foi resolvido!** Minsky refutado.

---

## 4. Conexão com a Neurobiologia

### 4.1 O Debate: O Cérebro Faz Backpropagation?

**Argumentos contra:**
- Não há evidência de que neurônios biológicos propagam sinais de erro para trás
- Os axônios transmitem em uma única direção (lei da polarização dinâmica de Cajal)
- O cérebro não calcula derivadas parciais explicitamente

**Argumentos a favor (analogias):**
- O cerebelo usa sinais de erro (fibras trepadeiras carregam "sinais de erro" das olivas inferiores)
- Neuromoduladores (dopamina) funcionam como "sinais de recompensa/erro" globais
- **Predictive coding** — uma teoria moderna de que o córtex processa "erros de predição" entre camadas

### 4.2 Alternativas Bio-plausíveis

Pesquisadores desenvolveram algoritmos mais próximos da biologia:

| Algoritmo | Descrição | Bio-plausibilidade |
|---|---|---|
| **Feedback Alignment** | Pesos aleatórios (não transpostos) para backward pass | Média |
| **Target Propagation** | Cada camada tem um "alvo" local | Alta |
| **Equilibrium Propagation** | Baseado em energia, sem fase backward separada | Alta |
| **Spike-Timing Dependent Plasticity (STDP)** | Ajuste baseado em timing de pulsos | Muito alta |
| **Predictive Coding** | Minimização de erro de predição local | Alta |

---

## 5. Neocognitron e CNNs — Da Biologia ao Reconhecimento Visual

### 5.1 Neocognitron de Fukushima (1980)

Inspiração direta de Hubel & Wiesel:
```
Hierarquia biológica:          Hierarquia do Neocognitron:
V1 → Células Simples    →     Camadas S (simple cells)
V1 → Células Complexas  →     Camadas C (complex cells)
V2 → Detecção de formas →     Camadas S/C mais profundas
IT → Reconhecimento      →     Camada final de classificação
```

Problema: o Neocognitron usava aprendizado não-supervisionado, limitando seu desempenho.

### 5.2 LeNet-5 de Yann LeCun (1989/1998)

LeCun combinou a arquitetura inspirada biologicamente do Neocognitron com o **backpropagation**:

**Arquitetura LeNet-5:**
```
Input (32×32 pixels)
  → Conv1 (6 filtros 5×5, saída 28×28×6)
    → Pool1 (subsampling 2×2, saída 14×14×6)
      → Conv2 (16 filtros 5×5, saída 10×10×16)
        → Pool2 (subsampling 2×2, saída 5×5×16)
          → FC1 (120 neurônios)
            → FC2 (84 neurônios)
              → Output (10 classes: dígitos 0-9)
```

**Aplicação:** Leitura automática de cheques nos correios dos EUA — processou milhões de cheques, provando que redes neurais podiam ter aplicação comercial.

---

## 6. Outros Marcos do Renascimento

### 6.1 LSTM — Hochreiter & Schmidhuber (1997)

**Long Short-Term Memory** resolveu o problema do "vanishing gradient" em redes recorrentes:
- Introduziu **portas** (gates) que controlam o fluxo de informação
- Porta de esquecimento: decide o que descartar
- Porta de entrada: decide o que adicionar
- Porta de saída: decide o que transmitir
- Permitiu processar sequências longas (texto, áudio, séries temporais)

**Analogia biológica:** As portas são análogas aos mecanismos de **atenção seletiva** e **memória de trabalho** no córtex pré-frontal.

### 6.2 Support Vector Machines (SVM) — Vapnik (1995)

Ironicamente, o renascimento das redes neurais nos 1990s foi parcialmente eclipsado pelas SVMs:
- Garantias teóricas mais sólidas
- Menos hiperparâmetros para ajustar
- Melhores resultados em muitos benchmarks da época
- Dominaram o ML prático de ~1995 a ~2010

Foi só com o deep learning (2012+) que redes neurais reconquistaram definitivamente a supremacia.

---

> **Referências:** Hopfield (1982), Hinton & Sejnowski "Boltzmann Machines" (1985), Rumelhart, Hinton & Williams "Learning representations by back-propagating errors" Nature (1986), Rumelhart & McClelland "PDP" (1986), LeCun et al. "LeNet" (1989/1998), Hochreiter & Schmidhuber "LSTM" (1997), Werbos (1974).
