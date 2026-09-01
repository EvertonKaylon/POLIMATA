# 🧠 Trilha de Neurobiologia para IA Bioinspirada
## Volume II — Mês 5: Integração Hipocampo + Neocórtex

> **Pré-requisito:** Mês 4 completo. Você tem `SpatialPooler` e `TemporalMemory` rodando.
> E do Vol. I: `HippocampalIndexSystem` implementado.
>
> **O que acontece aqui:** Você vai conectar os dois módulos. Não é só chamar um dentro do outro —
> você vai entender **por que** o cérebro precisa dos dois, como eles se comunicam durante o sono,
> e como criar o protocolo completo de experiência → consolidação → recall.

---

## A QUESTÃO CENTRAL DESTE MÊS

> *"Se o hipocampo aprende tão rápido e bem, por que o neocórtex existe?"*

A resposta está na **Teoria dos Sistemas de Aprendizado Complementares** (Semana 17).
Entender isso biologicamente é o que vai tornar a sua arquitetura integrada *coerente*,
não apenas funcional.

---

---

# MÊS 5 — INTEGRAÇÃO

---

## SEMANA 17
**Tema: CLS — Por Que o Cérebro Tem Dois Sistemas de Memória**

### 📚 O que estudar (2h)

**O problema da catástrofe catastrófica:**
Treine uma rede em padrão A. Depois em padrão B. A rede *esquece* A.
Isso se chama *catastrophic interference* — e qualquer sistema de aprendizado sequencial sofre isso.

O cérebro humano não tem esse problema. Por quê?

**Complementary Learning Systems Theory — McClelland, McNaughton & O'Reilly (1995):**

| | Hipocampo | Neocórtex |
|---|-----------|-----------|
| Velocidade | Rápido (one-shot) | Lento (centenas de repetições) |
| Tipo de memória | Episódica (específica) | Semântica (geral, regras) |
| Esquecimento | Rápido — libera espaço | Lento — consolida padrões |
| Interferência | Alta se sobrecarregado | Baixa (interleaved learning) |
| Papel | Indexar episódios novos | Generalizar padrões repetidos |

**A solução — Interleaved Learning:**
O córtex não aprende de cada experiência individualmente.
O hipocampo armazena. Durante o sono (NREM), o hipocampo *replaya* episódios aleatórios para o córtex
— de forma **intercalada**. O córtex aprende de poucos de cada vez, sem interferência.
Isso explica por que dormir consolida aprendizado.

**Recursos gratuitos:**
- Paper: McClelland, McNaughton & O'Reilly 1995 — "Why there are complementary learning systems in the hippocampus and neocortex" — busque "McClelland 1995 complementary learning systems PMC" → acesso gratuito
- Paper recente: Kumaran, Hassabis & McClelland 2016 "What Learning Systems do Intelligent Agents Need?" — Cell, busque no PubMed/ResearchGate
- YouTube: "Complementary Learning Systems" — há aulas de universidades no YouTube (MIT, Stanford)

### ⚗️ Exercício Prático (1h30)

```python
# cls_demo.py — Semana 17
# Demonstra catástrofe catastrófica E a solução CLS
import numpy as np
import matplotlib.pyplot as plt


def make_pattern(size=100, n_active=20, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    p = np.zeros(size)
    p[rng.choice(size, n_active, replace=False)] = 1
    return p


def pattern_recall_accuracy(network_out, target):
    """Correlação de Pearson como métrica de recall."""
    if target.std() < 1e-8 or network_out.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(network_out, target)[0, 1])


# ─── Rede cortical simples (aprende devagar) ───────────────────────────────────
class SlowCortex:
    """
    Rede de 2 camadas com backprop simplificado.
    Modela o neocórtex na CLS Theory.
    """
    def __init__(self, input_size, hidden=64, lr=0.001):
        self.W1 = np.random.randn(hidden, input_size) * 0.1
        self.W2 = np.random.randn(input_size, hidden) * 0.1
        self.lr = lr

    def forward(self, x):
        h = np.tanh(self.W1 @ x)
        return self.W2 @ h

    def learn(self, x, target):
        """Uma iteração de gradiente descendente."""
        h      = np.tanh(self.W1 @ x)
        output = self.W2 @ h
        error  = target - output
        dW2    = np.outer(error, h) * self.lr
        delta  = self.W2.T @ error * (1 - h ** 2)
        dW1    = np.outer(delta, x) * self.lr
        self.W1 += dW1
        self.W2 += dW2
        return float(np.mean(error ** 2))


# ─── Buffer hipocampal (fast, one-shot) ───────────────────────────────────────
class FastHippocampus:
    """
    Buffer de memória instantâneo — simplifica o HippocampalIndexSystem
    para isolar o efeito CLS nesta demonstração.
    """
    def __init__(self, max_eps=500):
        self.memory   = []
        self.max_size = max_eps

    def store(self, pattern):
        if len(self.memory) >= self.max_size:
            self.memory.pop(0)
        self.memory.append(pattern.copy())

    def replay(self, n=1):
        if not self.memory:
            return []
        idx = np.random.choice(len(self.memory), min(n, len(self.memory)), replace=False)
        return [self.memory[i] for i in idx]


# ─── Experimento ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rng = np.random.default_rng(42)

    N_PATTERNS  = 12
    INPUT_SIZE  = 100
    patterns    = [make_pattern(INPUT_SIZE, 20, rng) for _ in range(N_PATTERNS)]

    # ── Experimento 1: Treino sequencial SEM hipocampo ──────────────────────
    print("=== SEM hipocampo (catástrofe catastrófica) ===")
    cortex_naive = SlowCortex(INPUT_SIZE, lr=0.05)   # lr ALTA → aprende rápido mas esquece
    recall_naive = []

    for i, p in enumerate(patterns):
        for _ in range(50):                           # treina 50x no padrão i
            cortex_naive.learn(p, p)
        recalls = [pattern_recall_accuracy(cortex_naive.forward(patterns[j]), patterns[j])
                   for j in range(i + 1)]
        recall_naive.append(np.mean(recalls))
        print(f"  Após padrão {i+1:2d}: recall médio = {recall_naive[-1]:.3f}")

    # ── Experimento 2: Com hipocampo + replay (CLS) ─────────────────────────
    print("\n=== COM hipocampo (CLS — aprendizado intercalado) ===")
    cortex_cls  = SlowCortex(INPUT_SIZE, lr=0.005)   # lr BAIXA → não esquece
    hippocampus = FastHippocampus()
    recall_cls  = []

    for i, p in enumerate(patterns):
        hippocampus.store(p)                          # armazena imediatamente (one-shot)

        # Córtex aprende via replay intercalado (não aprende o padrão diretamente)
        for _ in range(50):
            for old in hippocampus.replay(n=4):
                cortex_cls.learn(old, old)

        recalls = [pattern_recall_accuracy(cortex_cls.forward(patterns[j]), patterns[j])
                   for j in range(i + 1)]
        recall_cls.append(np.mean(recalls))
        print(f"  Após padrão {i+1:2d}: recall médio = {recall_cls[-1]:.3f}")

    # ── Visualização ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, N_PATTERNS + 1), recall_naive, 'r-o', label='Sem hipocampo (catástrofe)')
    ax.plot(range(1, N_PATTERNS + 1), recall_cls,   'g-s', label='Com hipocampo (CLS)')
    ax.set_xlabel('Número de padrões aprendidos')
    ax.set_ylabel('Recall médio de TODOS os padrões anteriores')
    ax.set_title('Catástrofe Catastrófica vs. CLS com Replay Hipocampal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('semana17_cls_demo.png')
    print("\nSalvo: semana17_cls_demo.png")
    print("\n→ Curva vermelha deve cair. Curva verde deve manter-se alta.")
```

### ✅ Como saber se absorveu

- [ ] A curva vermelha (sem hipocampo) cai drasticamente conforme mais padrões chegam
- [ ] A curva verde (CLS) mantém recall alto em todos os padrões
- [ ] Você consegue explicar em voz alta por que o hipocampo usa lr alta e o córtex usa lr baixa
- [ ] **Conexão obrigatória:** escreva como você vai implementar esta diferença de velocidade de aprendizado no seu projeto real (onde fica cada lr?)

---

## SEMANA 18
**Tema: Consolidação de Memória — O Que Acontece Enquanto Você Dorme**

### 📚 O que estudar (2h)

**O mecanismo biológico:**
Durante o sono NREM (sono lento), o hipocampo gera **sharp-wave ripples (SWR)**:
surtos de atividade de ~80-100ms em que sequências de episódios são re-executadas
em ~10-20× a velocidade original.

Modelo computacional do SWR:
1. Animal acorda → hipocampo cria índices de episódios
2. Sono NREM → hipocampo replaya em ordem temporal comprimida
3. Córtex recebe replay → atualiza conexões lentamente
4. Após consolidação → episódio "sai" do hipocampo para o córtex

**Evidências chave (todas com papers gratuitos):**
- Wilson & McNaughton 1994: place cells de rato re-ativam durante sono na *mesma sequência* que o labirinto
- Girardeau et al. 2009: bloquear SWRs prejudica aprendizado no dia seguinte
- Tse et al. 2007: memórias compatíveis com schema cortical consolidam 5-10× mais rápido

**Recursos gratuitos:**
- Paper: Wilson & McNaughton 1994 "Reactivation of hippocampal ensemble memories during sleep" — Science; busque "Wilson McNaughton 1994 sleep reactivation ResearchGate"
- Review: Carr, Jadhav & Frank 2011 "Hippocampal replay in the awake state" — Nature Neuroscience, PMC gratuito
- YouTube: "Memory Consolidation During Sleep" — Neuromatch Academy W3 (canal do YouTube)

### ⚗️ Exercício Prático (1h30)

```python
# sleep_consolidation.py — Semana 18
# Usa SpatialPooler e TemporalMemory do Mês 4
import numpy as np
import matplotlib.pyplot as plt


class MemoryConsolidationSystem:
    """
    Sistema de consolidação de memória com ciclos de sono.

    Protocolo biológico replicado:
      Acordado  → experiências → hipocampo (one-shot) + córtex (forward only)
      Sono      → hipocampo replaya → córtex aprende lentamente
    """
    def __init__(self, sp, tm, buffer_size: int = 500):
        self.sp  = sp
        self.tm  = tm
        self.buffer: list = []         # buffer hipocampal de episódios
        self.max_buf = buffer_size
        self.consolidation_log = []

    # ── fase acordado ──────────────────────────────────────────────────────────
    def awake_experience(self, input_pattern: np.ndarray) -> float:
        """
        Processa uma experiência nova.
        Hipocampo: armazena imediatamente.
        Córtex: forward pass SEM aprendizado.
        Retorna: anomaly score atual do córtex.
        """
        # Hipocampo — one-shot
        self.buffer.append(input_pattern.copy())
        if len(self.buffer) > self.max_buf:
            self.buffer.pop(0)

        # Córtex — apenas inferência
        sdr     = self.sp.compute(input_pattern, learn=False)
        _, anom = self.tm.compute(sdr, learn=False)
        return anom

    # ── fase sono ─────────────────────────────────────────────────────────────
    def sleep_cycle(self, n_replays: int = 20, batch_size: int = 5):
        """
        Simula um ciclo de sono (análogo a um período NREM):
          - Seleciona episódios aleatórios do buffer (interleaved)
          - Córtex aprende com o replay
          - Mede melhora no anomaly score
        """
        if not self.buffer:
            return 1.0, 1.0

        before = self._cortex_familiarity()

        for _ in range(n_replays):
            n = min(batch_size, len(self.buffer))
            batch = [self.buffer[i]
                     for i in np.random.choice(len(self.buffer), n, replace=False)]
            self.tm.reset()
            for p in batch:
                sdr = self.sp.compute(p, learn=True)
                self.tm.compute(sdr, learn=True)

        after = self._cortex_familiarity()
        self.consolidation_log.append({'before': before, 'after': after})
        return before, after

    def _cortex_familiarity(self, n_sample: int = 10) -> float:
        """Anomaly score médio do córtex para amostra do buffer."""
        if not self.buffer:
            return 1.0
        sample = [self.buffer[i]
                  for i in np.random.choice(len(self.buffer),
                                            min(n_sample, len(self.buffer)),
                                            replace=False)]
        scores = []
        for p in sample:
            sdr  = self.sp.compute(p, learn=False)
            _, a = self.tm.compute(sdr, learn=False)
            scores.append(a)
        return float(np.mean(scores))

    # ── protocolo completo ────────────────────────────────────────────────────
    def run_protocol(self, patterns: list, n_sessions: int = 6,
                     replays_per_sleep: int = 30):
        """
        Protocolo: N sessões de (acordado → sono → acordado → sono…).
        Imprime e retorna histórico de consolidação.
        """
        history = []
        print(f"{'Sessão':>7} | {'Antes sono':>10} | {'Depois sono':>11} | {'Melhora':>7}")
        print("-" * 45)
        for s in range(n_sessions):
            # Fase acordado
            for p in patterns:
                self.awake_experience(p)
            # Fase sono
            before, after = self.sleep_cycle(n_replays=replays_per_sleep)
            delta = before - after
            history.append({'session': s + 1, 'before': before, 'after': after})
            print(f"  {s+1:5d}   |   {before:.4f}   |    {after:.4f}   |  {delta:+.4f}")
        return history


# ─── Demonstração ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)

    from spatial_pooler import SpatialPooler
    from temporal_memory import TemporalMemory

    sp = SpatialPooler(input_size=200, column_count=512, active_frac=0.02)
    tm = TemporalMemory(n_columns=512, cells_per_column=16,
                        activation_threshold=8, learning_threshold=5)

    system   = MemoryConsolidationSystem(sp, tm)
    patterns = [(np.arange(200) % 15 == i).astype(float) for i in range(15)]

    print("Protocolo de Consolidação (Acordado → Sono × 6 sessões)\n")
    history = system.run_protocol(patterns, n_sessions=7, replays_per_sleep=40)

    # Gráfico
    sessions = [h['session'] for h in history]
    befores  = [h['before']  for h in history]
    afters   = [h['after']   for h in history]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sessions, befores, 'r--o', label='Anomaly antes do sono')
    ax.plot(sessions, afters,  'g-o',  label='Anomaly depois do sono')
    ax.fill_between(sessions, befores, afters, alpha=0.15, color='blue',
                    label='Melhora por sono')
    ax.set_xlabel('Sessão')
    ax.set_ylabel('Anomaly Score Médio (↓ = mais familiar)')
    ax.set_title('Consolidação Progressiva: Sono Reduz Anomaly Cortical')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('semana18_consolidacao.png')
    print("\nSalvo: semana18_consolidacao.png")
```

### ✅ Como saber se absorveu

- [ ] O anomaly score cai progressivamente sessão a sessão
- [ ] A diferença before/after sono é positiva (sono ajuda — sempre)
- [ ] Você consegue explicar: sharp-wave ripple biológico ↔ `sleep_cycle()` do código
- [ ] **Teste de stress:** execute `sleep_cycle(n_replays=0)`. O que acontece? Por quê?
- [ ] **Reflexão:** por que o replay precisa ser *aleatório* (e não na ordem original dos episódios)?

---

## SEMANA 19
**Tema: Codificação Preditiva — O Córtex como Máquina de Predição**

### 📚 O que estudar (2h)

**A ideia central:**
O neocórtex não passa informação passivamente de baixo para cima.
Cada área cortical *prediz* o que vai chegar das áreas abaixo.
Só o **erro de predição** (diferença entre o esperado e o real) sobe para cima.

```
Área Superior (PFC, IT…)
   ↓ predição (top-down)        ↑ erro (bottom-up)
Área Intermediária (V2, V4…)
   ↓ predição (top-down)        ↑ erro (bottom-up)
Área Primária (V1, S1…)
   ↓ predição (top-down)        ↑ erro (bottom-up)
Input sensorial
```

**Consequências diretas:**
- Atenção = amplificar erros das áreas de interesse
- Percepção = resolver ambiguidades com priors (predições top-down)
- Aprendizado = minimizar erros de predição (isso é literalmente o que backprop faz!)
- Alucinação = predições top-down dominando input bottom-up

**Conexão com o que você já tem:**
O anomaly score do seu `TemporalMemory` é exatamente o sinal de erro de predição do córtex.
O bursting = erro alto = "não esperava isso".

**Recursos gratuitos:**
- Paper: Rao & Ballard 1999 "Predictive coding in the visual cortex" — Nature Neuroscience; busque "Rao Ballard 1999 predictive coding ResearchGate" (PDF gratuito)
- YouTube: "Anil Seth - Your Brain Hallucinates Your Conscious Reality" — TEDx, ~17min (introdução acessível)
- YouTube: "Karl Friston - Predictive Coding" — várias palestras gratuitas, escolha uma de 30-40min
- Nota: **Não entre no Free Energy Principle agora** — ele aprofunda isso, mas é o próximo nível depois desta trilha

### ⚗️ Exercício Prático (1h30)

```python
# predictive_coding.py — Semana 19
import numpy as np
import matplotlib.pyplot as plt


class PredictiveCorticalLayer:
    """
    Camada cortical com codificação preditiva.

    Dois tipos de unidades por camada:
      r : representação  — o que esta camada acredita estar acontecendo
      e : erro de predição — diferença entre input e predição top-down

    Conexões:
      W_bottom_up  : aprende a codificar o input bottom-up
      W_top_down   : aprende a gerar a predição do input para a camada abaixo
    """
    def __init__(self, n_units: int, n_input: int, lr: float = 0.005):
        self.n  = n_units
        self.lr = lr

        # Representação e erro
        self.r = np.zeros(n_units)
        self.e = np.zeros(n_input)   # erro desta camada (sobe para cima)

        # Pesos
        self.W_bu = np.random.randn(n_units, n_input) * 0.05   # bottom-up
        self.W_td = np.random.randn(n_input, n_units) * 0.05   # top-down (gerador)

        self.prediction_for_below = np.zeros(n_input)

    def predict(self) -> np.ndarray:
        """Gera predição do que a camada abaixo deve estar enviando."""
        self.prediction_for_below = np.tanh(self.W_td @ self.r)
        return self.prediction_for_below

    def update(self, input_from_below: np.ndarray,
               prediction_from_above: np.ndarray = None) -> np.ndarray:
        """
        Atualiza a representação r.
        Retorna e (erro de predição que sobe).
        """
        predicted = self.predict()
        self.e    = input_from_below - predicted   # erro bottom-up

        # r é impulsionado pelo erro bottom-up e pelo sinal top-down
        dr = self.W_bu @ self.e
        if prediction_from_above is not None:
            n = min(len(prediction_from_above), self.n)
            dr[:n] += prediction_from_above[:n] - self.r[:n]

        self.r = np.tanh(self.r + self.lr * dr)
        return self.e

    def learn(self, input_from_below: np.ndarray):
        """Atualiza W_td para minimizar erro de predição."""
        predicted = self.predict()
        error     = input_from_below - predicted
        self.W_td += self.lr * np.outer(error, self.r)
        # Mantém W_bu proporcional a W_td (simplificação)
        self.W_bu = self.W_td.T * 0.5


class HierarchicalPredictiveCortex:
    """Córtex hierárquico com 2 camadas e codificação preditiva bidirecional."""
    def __init__(self, input_size, l1_size, l2_size, lr=0.005):
        self.L1 = PredictiveCorticalLayer(l1_size, input_size, lr)
        self.L2 = PredictiveCorticalLayer(l2_size, l1_size,    lr)

    def process(self, input_signal: np.ndarray, learn: bool = True) -> dict:
        # Top-down: L2 prediz o que L1 deve enviar
        l2_pred_for_l1 = self.L2.predict()

        # L1 processa o input com predição top-down de L2
        l1_error = self.L1.update(input_signal, l2_pred_for_l1)

        # L2 processa a representação de L1
        l2_error = self.L2.update(self.L1.r)

        if learn:
            self.L1.learn(input_signal)
            self.L2.learn(self.L1.r)

        total_error = float(np.mean(np.abs(l1_error)) + np.mean(np.abs(l2_error)))
        return {
            'l1_error_mag':   float(np.mean(np.abs(l1_error))),
            'l2_error_mag':   float(np.mean(np.abs(l2_error))),
            'total_pred_error': total_error,
            'l1_repr':        self.L1.r.copy(),
            'l2_repr':        self.L2.r.copy(),
        }


# ─── Demonstração ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)

    cortex = HierarchicalPredictiveCortex(input_size=50, l1_size=32, l2_size=16)

    # Dois estímulos alternados que o córtex vai aprender a prever
    stim_A = (np.arange(50) % 2 == 0).astype(float)
    stim_B = (np.arange(50) % 2 == 1).astype(float)
    sequence = [stim_A, stim_B] * 150   # 300 passos

    errors_L1, errors_L2 = [], []
    for step, stim in enumerate(sequence):
        result = cortex.process(stim, learn=True)
        errors_L1.append(result['l1_error_mag'])
        errors_L2.append(result['l2_error_mag'])

    print("Erro L1 inicial: {:.4f} → final: {:.4f}".format(
        np.mean(errors_L1[:10]), np.mean(errors_L1[-10:])))
    print("Erro L2 inicial: {:.4f} → final: {:.4f}".format(
        np.mean(errors_L2[:10]), np.mean(errors_L2[-10:])))

    # Suavização para visualização
    def smooth(x, w=20):
        return np.convolve(x, np.ones(w) / w, mode='valid')

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(smooth(errors_L1), label='Erro L1 (proximal ao input)')
    ax.plot(smooth(errors_L2), label='Erro L2 (mais abstrato)')
    ax.set_xlabel('Passo de tempo')
    ax.set_ylabel('Magnitude do erro de predição')
    ax.set_title('Codificação Preditiva: Erro Diminui com Aprendizado')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('semana19_predictive_coding.png')
    print("\nSalvo: semana19_predictive_coding.png")
    print("→ Ambos os erros devem cair progressivamente")
```

### ✅ Como saber se absorveu

- [ ] Erros de L1 e L2 decrescem com o tempo (córtex hierárquico aprendendo)
- [ ] Você consegue explicar em uma frase: o que sobe (bottom-up) vs o que desce (top-down)
- [ ] Você entende por que o anomaly score do seu TemporalMemory É o erro de predição cortical
- [ ] **Reflexão:** como CA1 no hipocampo faz algo análogo — recebendo de CA3 E do córtex entorrinal diretamente?

---

## SEMANA 20
**Tema: Neuromodulação — Os Controladores Globais do Cérebro**

### 📚 O que estudar (2h)

Dopamina, acetilcolina, serotonina e norepinefrina não são neurônios "normais".
São **sistemas difusos** que regulam *globalmente* como hipocampo e neocórtex operam.
Entender isso vai permitir que você module dinamicamente o seu sistema.

**Os 4 neuromoduladores relevantes para o seu projeto:**

| Neuromodulador | De onde vem | Função | Impacto computacional |
|---|---|---|---|
| **Dopamina (DA)** | VTA / Substância negra | Erro de predição de recompensa | Modula LTP/LTD, sinaliza "aprenda isso" |
| **Acetilcolina (ACh)** | Núcleo basal de Meynert | Atenção ao input atual | Alta plasticidade hipocampal, suprime recall |
| **Norepinefrina (NE)** | Locus coeruleus | Arousal, novidade, incerteza | Amplifica sinal, modula exploração |
| **Serotonina (5-HT)** | Núcleos da rafe | Humor, paciência | Modula horizonte temporal do planejamento |

**Para o seu projeto, três papéis diretos:**
1. **ACh alta** → hipocampo está em modo encoding (animal explorando algo novo) → LR alta
2. **ACh baixa** → hipocampo está em modo recall (tarefa familiar) → LR baixa
3. **NE** → novidade detectada (anomaly score alto) → direciona atenção

**Recursos gratuitos:**
- Review: Hasselmo 2006 "The role of acetylcholine in learning and memory" — Current Opinion in Neurobiology; busque PubMed/PMC gratuito
- Paper: Schultz, Dayan & Montague 1997 "A neural substrate of prediction and reward" — Science; busque no ResearchGate
- YouTube: "Dopamine as a Reward Prediction Error Signal" — há várias aulas de universidades

### ⚗️ Exercício Prático (1h30)

```python
# neuromodulation.py — Semana 20
import numpy as np


class NeuromodulatorSystem:
    """
    Sistema de neuromodulação: ajusta parâmetros de hipocampo e neocórtex
    dinamicamente baseado no contexto.

    Cada neuromodulador é um escalar normalizado [0-1].
    Valores típicos em repouso (~0.5), máximo em ação (~0.9), mínimo em descanso (~0.1).
    """
    def __init__(self):
        self.dopamine        = 0.5   # erro de predição de recompensa
        self.acetylcholine   = 0.5   # atenção ao input presente
        self.norepinephrine  = 0.3   # arousal / novelty
        self.serotonin       = 0.5   # paciência / horizonte temporal

    # ── sinais externos ───────────────────────────────────────────────────────

    def signal_reward(self, received: float, expected: float = 0.5) -> float:
        """
        Reward Prediction Error (RPE) — Schultz 1997.
        DA = f(recompensa recebida − recompensa esperada).
        DA > 0.5 → melhor que esperado → reforce o que fez
        DA < 0.5 → pior que esperado  → evite o que fez
        """
        rpe = received - expected
        self.dopamine = float(np.clip(0.5 + rpe, 0.0, 1.0))
        return rpe

    def signal_novelty(self, anomaly_score: float):
        """Novidade eleva norepinefrina (locus coeruleus ativado)."""
        self.norepinephrine = float(np.clip(0.2 + 0.8 * anomaly_score, 0.0, 1.0))

    def encoding_mode(self):
        """
        Modo exploração/encoding (animal em ambiente novo):
        ACh alta → hipocampo prioriza input presente (encoding)
        NE moderada-alta → atenção elevada
        """
        self.acetylcholine  = 0.85
        self.norepinephrine = 0.65

    def recall_mode(self):
        """
        Modo recall (tarefa conhecida):
        ACh baixa → hipocampo prioriza recuperação de memória
        NE baixa  → estado calmo
        """
        self.acetylcholine  = 0.20
        self.norepinephrine = 0.25

    def sleep_mode(self):
        """
        Modo sono NREM:
        ACh muito baixa → consolidação cortical dominante
        DA moderada → facilita retenção de informação útil
        """
        self.acetylcholine  = 0.05
        self.norepinephrine = 0.10
        self.dopamine       = 0.60

    # ── saídas para outros módulos ────────────────────────────────────────────

    def hippocampal_lr(self, base_lr: float = 0.01) -> float:
        """
        LR do hipocampo modulada por ACh.
        ACh alta → mais plasticidade (encoding).
        ACh baixa → menos plasticidade (recall ou sono).
        """
        return base_lr * (0.3 + 1.4 * self.acetylcholine)

    def cortical_lr(self, base_lr: float = 0.001) -> float:
        """
        LR do córtex modulada por DA.
        DA alta → reforçar aprendizado.
        """
        return base_lr * (0.3 + 1.4 * self.dopamine)

    def novelty_gain(self) -> float:
        """Fator de amplificação de atenção — NE alto = mais sensível a novidades."""
        return 1.0 + 1.5 * self.norepinephrine

    def summary(self):
        print(f"  DA={self.dopamine:.2f} | ACh={self.acetylcholine:.2f} | "
              f"NE={self.norepinephrine:.2f} | 5HT={self.serotonin:.2f}")


# ─── Demonstração ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    nm = NeuromodulatorSystem()

    scenarios = [
        ("Baseline",                   lambda: None,
         "situação de repouso"),
        ("Exploração (encoding_mode)", lambda: nm.encoding_mode(),
         "animal entra em ambiente novo"),
        ("Recall (recall_mode)",       lambda: nm.recall_mode(),
         "executando tarefa conhecida"),
        ("Recompensa inesperada",       lambda: nm.signal_reward(1.0, 0.1),
         "ganhou mais do que esperava"),
        ("Anomalia detectada",          lambda: nm.signal_novelty(0.9),
         "algo muito inesperado aconteceu"),
        ("Sono NREM (sleep_mode)",      lambda: nm.sleep_mode(),
         "consolidação durante o sono"),
    ]

    for name, action, desc in scenarios:
        nm = NeuromodulatorSystem()   # reset
        action()
        print(f"\n[{name}] — {desc}")
        nm.summary()
        print(f"  → Hipocampo LR: {nm.hippocampal_lr():.5f}")
        print(f"  → Córtex LR:    {nm.cortical_lr():.5f}")
        print(f"  → Novelty gain: {nm.novelty_gain():.2f}x")
```

### ✅ Como saber se absorveu

- [ ] Em `encoding_mode`, a LR hipocampal é > 2× a LR do `recall_mode`
- [ ] Recompensa inesperada eleva dopamina e portanto a LR cortical (córtex deve aprender mais)
- [ ] Você entende por que ACh baixo durante o sono é fundamental para a consolidação
- [ ] **Integração:** adicione `NeuromodulatorSystem` ao seu `HippocampalIndexSystem` do Vol. I — use `hippocampal_lr()` para modular a taxa de aprendizado das sinapses STDP do CA3

---

## SEMANA 21
**Tema: O Sistema Integrado — Gate do Mês 5**

### 📚 O que estudar (2h)

Esta semana não tem leitura nova. É semana de **construção e diagnóstico**.

Se quiser uma leitura leve de referência:
- Review: Kumaran, Hassabis & McClelland 2016 "What Learning Systems do Intelligent Agents Need?" — Cell; busque PubMed (resume CLS para IA moderna de forma clara)

### ⚗️ Exercício Prático — Gate (1h30 + quanto precisar)

```python
# integrated_system.py — Semana 21 (Gate do Mês 5)
import numpy as np

# Importe seus módulos dos arquivos anteriores:
#   from volume1_hippocampus import HippocampalIndexSystem
#   from spatial_pooler      import SpatialPooler
#   from temporal_memory     import TemporalMemory
#   from sleep_consolidation import MemoryConsolidationSystem
#   from neuromodulation     import NeuromodulatorSystem


class NeocorticalHippocampalSystem:
    """
    Sistema neocórtex-hipocampo integrado.

    FLUXO DE EXPERIÊNCIA NOVA:
      input → NM.encoding_mode()
            → hipocampo.store_episode()  [fast, one-shot]
            → neocortex.sp.compute(learn=False)  [forward pass]
            → NM.signal_novelty(anomaly_score)

    FLUXO DE RECALL:
      cue → NM.recall_mode()
          → hipocampo.recall_episode(cue)   [CA3 pattern completion]
          → neocortex.sp.compute(recalled)  [representação semântica]

    FLUXO DE SONO (consolidação):
      NM.sleep_mode()
      → buffer hipocampal → córtex aprende (sleep_cycle)
    """
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.nm  = NeuromodulatorSystem()

        # ── Hipocampo (rápido) ─────────────────────────────────────────────
        # Use seu HippocampalIndexSystem do Volume I.
        # Se ainda não tiver integrado, use o FastHippocampus desta semana.
        try:
            from volume1_hippocampus import HippocampalIndexSystem
            self.hippocampus = HippocampalIndexSystem(cortex_size=input_size)
            self._hippo_mode = 'full'
        except ImportError:
            from cls_demo import FastHippocampus
            self.hippocampus = FastHippocampus(max_eps=1000)
            self._hippo_mode = 'buffer'
            print("[INFO] Usando FastHippocampus (Volume I não encontrado)")

        # ── Neocórtex (lento) ──────────────────────────────────────────────
        self.sp  = SpatialPooler(input_size=input_size,
                                 column_count=2048, active_frac=0.02)
        self.tm  = TemporalMemory(n_columns=2048, cells_per_column=16,
                                  activation_threshold=8, learning_threshold=5)

        # ── Consolidação ───────────────────────────────────────────────────
        self.consolidation = MemoryConsolidationSystem(self.sp, self.tm)

        # ── Métricas ───────────────────────────────────────────────────────
        self.experience_count = 0
        self.sleep_count      = 0
        self.anomaly_history  = []

    # ── protocolo público ─────────────────────────────────────────────────────

    def experience(self, input_pattern: np.ndarray) -> dict:
        """Processa uma nova experiência. Retorna métricas."""
        self.nm.encoding_mode()
        self.experience_count += 1

        # Hipocampo: armazenamento imediato
        if self._hippo_mode == 'full':
            self.hippocampus.store_episode(input_pattern)
        else:
            self.hippocampus.store(input_pattern)

        # Buffer de consolidação
        self.consolidation.buffer.append(input_pattern.copy())
        if len(self.consolidation.buffer) > self.consolidation.max_buf:
            self.consolidation.buffer.pop(0)

        # Neocórtex: apenas forward pass
        sdr     = self.sp.compute(input_pattern, learn=False)
        _, anom = self.tm.compute(sdr, learn=False)
        self.nm.signal_novelty(anom)
        self.anomaly_history.append(anom)

        return {
            'step':           self.experience_count,
            'cortex_anomaly': anom,
            'novelty_gain':   self.nm.novelty_gain(),
            'hippo_lr':       self.nm.hippocampal_lr(),
        }

    def recall(self, cue: np.ndarray) -> dict:
        """Recupera memória dado um cue parcial."""
        self.nm.recall_mode()

        # Hipocampo: recall episódico
        if self._hippo_mode == 'full':
            recalled = self.hippocampus.recall_episode(cue)
        else:
            # Fallback: retorna o episódio mais similar do buffer
            buf = self.hippocampus.memory
            if not buf:
                recalled = cue.copy()
            else:
                sims    = [np.dot(cue, p) / (np.linalg.norm(p) + 1e-8) for p in buf]
                recalled = buf[int(np.argmax(sims))].copy()

        # Neocórtex: representação semântica do recalled
        sdr     = self.sp.compute(recalled, learn=False)
        _, anom = self.tm.compute(sdr, learn=False)

        return {
            'episodic_recall': recalled,
            'semantic_sdr':    sdr,
            'cortex_anomaly':  anom,
        }

    def sleep(self, n_replays: int = 60) -> tuple:
        """Consolida memórias. Retorna (anomaly_before, anomaly_after)."""
        self.nm.sleep_mode()
        self.sleep_count += 1
        before, after = self.consolidation.sleep_cycle(n_replays=n_replays)
        self.nm.encoding_mode()   # acorda depois
        return before, after

    def status(self):
        print("\n═══ STATUS DO SISTEMA ═══")
        print(f"Experiências totais:  {self.experience_count}")
        print(f"Ciclos de sono:       {self.sleep_count}")
        print(f"Buffer de episódios:  {len(self.consolidation.buffer)}")
        if self.anomaly_history:
            last = np.mean(self.anomaly_history[-10:])
            print(f"Anomaly recente:      {last:.4f}")
        print("Neuromodulação:")
        self.nm.summary()


# ─── Experimento do Gate ───────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)

    system = NeocorticalHippocampalSystem(input_size=100)

    # 10 "experiências de vida"
    experiences = [(np.arange(100) % 10 == i).astype(float) for i in range(10)]

    print("═══ FASE 1: Experiências novas ═══")
    for i, exp in enumerate(experiences):
        m = system.experience(exp)
        print(f"  Exp {i+1:2d} | Anomaly: {m['cortex_anomaly']:.3f} "
              f"| Novelty: {m['novelty_gain']:.2f}x")

    print("\n═══ FASE 2: Sono (consolidação) ═══")
    before, after = system.sleep(n_replays=80)
    print(f"  Anomaly: {before:.4f} → {after:.4f}  (melhora: {before-after:+.4f})")

    print("\n═══ FASE 3: Recall com ruído ═══")
    for i, exp in enumerate(experiences[:5]):
        noisy = exp.copy()
        flip  = np.random.choice(100, 30, replace=False)
        noisy[flip] = 1 - noisy[flip]

        result     = system.recall(noisy)
        similarity = float(np.corrcoef(exp, result['episodic_recall'])[0, 1])
        print(f"  Recall {i+1}: similaridade = {similarity:.3f}")

    system.status()

    # ── Critérios de aprovação no Gate ────────────────────────────────────────
    print("\n═══ CRITÉRIOS DO GATE ═══")
    recent_anom = np.mean(system.anomaly_history[-5:])
    print(f"[{'✓' if recent_anom < 0.8 else '✗'}] Córtex processou experiências"
          f" (anomaly < 0.8): {recent_anom:.3f}")
    print(f"[{'✓' if before > after else '✗'}] Sono melhorou familiaridade cortical:"
          f" {before:.3f} → {after:.3f}")
    print(f"[  ] Recall hipocampal funcionando (verifique similarities acima)")
    print(f"[  ] Você consegue explicar cada módulo sem olhar o código?")
```

### ✅ Critérios de aprovação do Gate (Mês 5)

- [ ] Sistema integrado roda ponta a ponta sem erros
- [ ] Sono melhora o anomaly cortical (before > after)
- [ ] Recall hipocampal tem similaridade > 0.5 com o original (30% de ruído)
- [ ] Você consegue explicar em 3 frases por que o sistema precisa dos dois módulos
- [ ] Você adicionou `NeuromodulatorSystem` modulando as LRs do sistema real

---

## ✅ CHECKLIST DO MÊS 5 (Integração)

- [ ] Demonstrou catástrofe catastrófica e a resolução CLS com replay
- [ ] Implementou ciclos de sono com replay hipocampal → melhora cortical mensurável
- [ ] Implementou codificação preditiva hierárquica (erro propagando para cima)
- [ ] Implementou sistema de neuromodulação (DA, ACh, NE) modulando LRs
- [ ] Integrou tudo em `NeocorticalHippocampalSystem` com protocolo experience→sleep→recall
- [ ] **Gate aprovado:** sistema integrado rodando com todos os critérios acima

**Próximo:** instrumentar cientificamente o que você construiu — benchmarks, otimização e documentação de P&D.
