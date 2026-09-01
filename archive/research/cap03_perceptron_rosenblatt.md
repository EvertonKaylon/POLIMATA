# ⚡ Capítulo 3 — O Perceptron de Rosenblatt (1958)

> Documento expandido do índice principal `fundamentos_ia_pesquisa.md`

---

## 1. Frank Rosenblatt — O Homem por Trás da Máquina

### 1.1 Biografia

- **Nascido:** 11 de julho de 1928, New Rochelle, Nova York
- **Falecido:** 11 de julho de 1971 (afogamento em acidente de barco, no seu 43° aniversário)
- **Formação:** PhD em Psicologia pela Cornell University
- **Cargo:** Pesquisador no Cornell Aeronautical Laboratory

### 1.2 O Contexto Intelectual

Rosenblatt estava na **interseção perfeita** de disciplinas:
- **Psicologia:** Entendia como humanos aprendem e percebem
- **Neurociência:** Conhecia o trabalho de McCulloch-Pitts e Hebb
- **Matemática:** Tinha formação quantitativa suficiente para formalizar modelos
- **Engenharia:** Trabalhava em um laboratório aeronáutico com acesso a hardware

Sua motivação era criar uma máquina que replicasse a capacidade humana de **reconhecimento de padrões** — algo que humanos fazem sem esforço mas que era impossível para computadores da época.

### 1.3 A Publicação Original

O paper seminal: *"The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"*, publicado em 1958 na revista *Psychological Review*.

**Nota importante:** Rosenblatt publicou numa revista de **psicologia**, não de engenharia ou matemática. Ele via o perceptron como um modelo do cérebro, não como uma ferramenta de engenharia.

---

## 2. Anatomia Completa do Perceptron

### 2.1 A Estrutura Matemática

O perceptron é o modelo mais simples possível de neurônio artificial:

```
Entradas:     x₁, x₂, x₃, ..., xₙ    (dados de entrada)
Pesos:        w₁, w₂, w₃, ..., wₙ    (importância de cada entrada)
Bias:         b                         (deslocamento/threshold)

Soma ponderada:
    z = (x₁·w₁) + (x₂·w₂) + ... + (xₙ·wₙ) + b
    z = Σᵢ(xᵢ·wᵢ) + b

Função de ativação (step/degrau):
    Se z ≥ 0 → saída y = 1 (classe positiva)
    Se z < 0 → saída y = 0 (classe negativa)
```

### 2.2 Interpretação Geométrica

O perceptron define um **hiperplano** no espaço de entradas:

- Em 2D (duas entradas): o hiperplano é uma **reta** que divide o plano em duas regiões
- Em 3D (três entradas): é um **plano** que divide o espaço
- Em nD: é um hiperplano (n-1)-dimensional

**Exemplo em 2D:**

A equação w₁x₁ + w₂x₂ + b = 0 define uma reta. Pontos de um lado são classe 1, do outro, classe 0.

```
     x₂
     │
     │  Classe 1
     │  ●  ●
     │    ●  ●
     │─────────── ← reta de decisão (w₁x₁ + w₂x₂ + b = 0)
     │  ○  ○
     │    ○
     │  Classe 0
     └──────────── x₁
```

### 2.3 Tabela de Correspondências Biológicas

| Elemento Biológico | Elemento do Perceptron | Detalhes |
|---|---|---|
| Sinais nos dendritos | Entradas xᵢ | Valores numéricos vindos do mundo externo |
| Força das sinapses | Pesos wᵢ | Determinam a importância de cada entrada |
| Integração somática | Soma ponderada Σ | Combina todos os sinais |
| Limiar de disparo (~-55mV) | Bias b | Controla quão "sensível" o neurônio é |
| Potencial de ação (tudo ou nada) | Função step | Saída binária: 0 ou 1 |
| Plasticidade sináptica (Hebb) | Regra de aprendizado | Ajuste dos pesos baseado no erro |
| Neurônio completo | Perceptron | Uma unidade de decisão |

---

## 3. O Algoritmo de Aprendizado do Perceptron

### 3.1 A Regra de Atualização

```
Para cada exemplo de treino (x, y_desejado):
    1. Calcular saída: y_obtido = step(Σ wᵢxᵢ + b)
    2. Calcular erro: erro = y_desejado - y_obtido
    3. Atualizar pesos: wᵢ = wᵢ + η · erro · xᵢ
    4. Atualizar bias:  b  = b  + η · erro

Onde η (eta) = taxa de aprendizado (ex: 0.01, 0.1)
```

### 3.2 O que acontece em cada caso

| Situação | y_desejado | y_obtido | Erro | O que acontece |
|---|---|---|---|---|
| Acerto | 1 | 1 | 0 | Nada muda — pesos corretos |
| Acerto | 0 | 0 | 0 | Nada muda — pesos corretos |
| Falso negativo | 1 | 0 | +1 | Pesos **aumentam** (fortalecer conexões que deveriam ativar) |
| Falso positivo | 0 | 1 | -1 | Pesos **diminuem** (enfraquecer conexões que não deveriam ativar) |

### 3.3 Teorema de Convergência do Perceptron

**Se os dados são linearmente separáveis**, o algoritmo do perceptron **sempre converge** — ou seja, encontra uma solução em um número finito de passos.

Este teorema foi provado por Rosenblatt e depois formalmente por **Novikoff (1962)** e **Block (1962)**.

**Limitação crucial:** O teorema só se aplica a dados linearmente separáveis. Para dados não-separáveis (como XOR), o algoritmo **nunca converge** — oscila infinitamente.

### 3.4 Implementação Python Completa

```python
import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = np.zeros(n_inputs)
        self.bias = 0
        self.lr = learning_rate
        self.history = []  # para rastrear o aprendizado
    
    def predict(self, x):
        """Forward pass: soma ponderada + função step"""
        z = np.dot(x, self.weights) + self.bias
        return 1 if z >= 0 else 0
    
    def train(self, X, y, epochs=100):
        """Treinar o perceptron"""
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = yi - prediction
                
                # Atualizar pesos e bias
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                
                errors += abs(error)
            
            self.history.append(errors)
            if errors == 0:
                print(f"Convergiu na época {epoch}!")
                break
        
        return self
    
    def evaluate(self, X, y):
        """Avaliar acurácia"""
        correct = sum(self.predict(xi) == yi for xi, yi in zip(X, y))
        return correct / len(y) * 100


# === TESTE: Porta AND ===
print("=== Porta AND ===")
X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
y_and = np.array([0, 0, 0, 1])

p_and = Perceptron(2, learning_rate=0.1)
p_and.train(X_and, y_and, epochs=100)

for xi, yi in zip(X_and, y_and):
    pred = p_and.predict(xi)
    print(f"  {xi} -> esperado: {yi}, obtido: {pred} {'✓' if pred == yi else '✗'}")

# === TESTE: Porta OR ===
print("\n=== Porta OR ===")
y_or = np.array([0, 1, 1, 1])

p_or = Perceptron(2, learning_rate=0.1)
p_or.train(X_and, y_or, epochs=100)

for xi, yi in zip(X_and, y_or):
    pred = p_or.predict(xi)
    print(f"  {xi} -> esperado: {yi}, obtido: {pred} {'✓' if pred == yi else '✗'}")

# === TESTE: Porta XOR (vai falhar!) ===
print("\n=== Porta XOR (impossível para perceptron simples!) ===")
y_xor = np.array([0, 1, 1, 0])

p_xor = Perceptron(2, learning_rate=0.1)
p_xor.train(X_and, y_xor, epochs=1000)

for xi, yi in zip(X_and, y_xor):
    pred = p_xor.predict(xi)
    print(f"  {xi} -> esperado: {yi}, obtido: {pred} {'✓' if pred == yi else '✗'}")

print(f"\n  O XOR NÃO converge! Erros finais: {p_xor.history[-1]}")
```

---

## 4. O Mark I Perceptron — A Máquina Física

### 4.1 Especificações Técnicas

O Mark I Perceptron foi construído no Cornell Aeronautical Laboratory em 1957-1958:

| Componente | Especificação |
|---|---|
| **Entrada** | 400 fotocélulas (CdS) dispostas em grade 20×20 |
| **Conexões** | Fios conectando fotocélulas a neurônios artificiais |
| **Pesos** | Potenciômetros motorizados (resistores ajustáveis) |
| **Processamento** | Circuitos analógicos (soma e threshold) |
| **Saída** | 8 neurônios de saída |
| **Aprendizado** | Motores ajustavam potenciômetros automaticamente |
| **Peso** | Vários racks de equipamento (tamanho de uma sala) |

### 4.2 O que ele podia fazer

- Distinguir triângulos de quadrados
- Classificar letras manuscritas simples
- Aprender padrões visuais com ~50 exemplos de treino
- Acurácia limitada, mas demonstrava **aprendizado genuíno**

### 4.3 A Cobertura da Mídia

O New York Times reportou em 1958:

> *"O embrião de um computador eletrônico que [a Marinha] espera que possa andar, falar, ver, escrever, reproduzir-se e ter consciência de sua existência."*

Esta cobertura exagerada contribuiu para expectativas irrealistas que depois alimentaram a decepção e o corte de financiamento.

---

## 5. ADALINE e MADALINE (1960)

### 5.1 Bernard Widrow e Marcian "Ted" Hoff

Na Stanford University, Widrow e seu aluno Hoff (que depois co-inventaria o microprocessador Intel 4004) desenvolveram o **ADALINE** (Adaptive Linear Neuron).

### 5.2 Diferença Chave: Regra Widrow-Hoff (LMS)

```
Perceptron:  erro = y_desejado - step(z)     ← erro DEPOIS da ativação
ADALINE:     erro = y_desejado - z            ← erro ANTES da ativação (valor contínuo)
```

**Vantagens da regra LMS:**
- O erro é contínuo e diferenciável (não é 0 ou 1)
- Permite convergência mais suave
- É precursora direta do **gradiente descendente** usado no deep learning
- Matematicamente: minimiza o **erro quadrático médio** (MSE)

### 5.3 MADALINE (Multiple ADALINE)

Widrow e Hoff também criaram a **MADALINE** — a primeira rede com **múltiplos neurônios ADALINE** conectados em camadas. Embora primitiva, é considerada uma das primeiras redes multicamadas funcionais.

**Aplicação prática:** Usado pela companhia telefônica para eliminar ecos em linhas telefônicas — um dos primeiros usos comerciais de redes neurais.

---

## 6. O Perceptron Multicamadas (MLP) — A Teoria que Faltava

### 6.1 A Ideia

Já se sabia que adicionar **camadas ocultas** entre entrada e saída permitiria resolver problemas como XOR. O problema era: **como treinar os pesos das camadas ocultas?**

```
Perceptron simples:     Entrada → [Neurônio] → Saída
                        (só linearmente separáveis)

MLP (multicamadas):     Entrada → [Camada Oculta] → [Saída]
                        (qualquer função — em teoria)
```

### 6.2 O Problema do Crédito (Credit Assignment Problem)

- O erro é medido na **saída** da rede
- Mas como saber **qual neurônio da camada oculta** contribuiu para o erro?
- Como distribuir a "culpa" do erro entre todos os pesos internos?

Este problema só seria resolvido em 1986 com o **backpropagation** (Capítulo 5).

### 6.3 O Teorema da Aproximação Universal

Prova-se que um MLP com **uma camada oculta** e suficientes neurônios pode aproximar **qualquer função contínua** com precisão arbitrária (teorema de Cybenko, 1989; Hornik, 1991).

Ou seja: redes neurais multicamadas são **aproximadores universais**. O desafio é treiná-las.

---

## 7. O Legado de Rosenblatt

### 7.1 Morte Prematura

Rosenblatt morreu tragicamente em 1971, aos 43 anos, em um acidente de barco no seu aniversário. Ele não viveu para ver seu trabalho reivindicado pelo renascimento das redes neurais nos anos 1980.

### 7.2 A Importância Histórica

| Contribuição | Impacto |
|---|---|
| Primeiro modelo com **aprendizado automático** | Base de todo machine learning |
| Provou que máquinas podem **aprender de dados** | Paradigma dominante da IA moderna |
| Inspirou gerações de pesquisadores | Hinton, LeCun, Bengio citam Rosenblatt |
| Mostrou que neurociência pode guiar engenharia | Tradição conexionista |
| Mark I: primeira implementação física | Demonstrou viabilidade prática |

### 7.3 A Ponte entre Hebb e Backpropagation

Rosenblatt ocupou o papel histórico de traduzir a **teoria biológica de Hebb** em um **algoritmo computacional funcional**. Ele não completou a jornada (não resolveu o problema multicamadas), mas estabeleceu a direção que eventualmente levou ao deep learning.

---

> **Referências:** Rosenblatt "The Perceptron" (1958), Rosenblatt "Principles of Neurodynamics" (1962), Widrow & Hoff "Adaptive Switching Circuits" (1960), Block "The Perceptron" (1962), Novikoff "On convergence proofs" (1962), Cybenko "Approximation by superpositions" (1989).
