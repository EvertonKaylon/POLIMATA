# 🧠 Trilha de Neurobiologia para IA Bioinspirada
## Volume II — Mês 4: O Neocórtex

> **Pré-requisito:** Volume I completo. Você deve ter o `HippocampalIndexSystem` da Semana 12 rodando.
>
> **O que muda aqui:** No Vol. I você construiu um sistema que aprende rápido e lembra episódios
> específicos — o hipocampo. Agora você constrói o parceiro evolutivamente mais recente: o **neocórtex**,
> que aprende devagar, generaliza padrões e **prediz o futuro**. Ao final do Mês 4, você terá
> os dois blocos centrais do seu projeto.

---

## O QUE O NEOCÓRTEX FAZ (que o hipocampo não faz)

```
HIPOCAMPO               NEOCÓRTEX
─────────────────       ─────────────────────────────
Aprende rápido          Aprende devagar (muitas repetições)
Lembra episódios        Extrai padrões / regras
Específico              Generaliza
Capacidade limitada     Capacidade enorme
Memória temporária      Memória de longo prazo
```

O hipocampo é o bloco de notas. O neocórtex é o livro encadernado.

---

---

# MÊS 4 — O NEOCÓRTEX

---

## SEMANA 13
**Tema: A Arquitetura do Neocórtex — Camadas e Colunas**

### 📚 O que estudar (2h)

O neocórtex é organizado em **6 camadas horizontais** (layers) e **colunas verticais** (columns).
Esta organização não é decorativa — ela define exatamente como a informação flui.

**As 6 camadas e suas funções:**

| Camada | Conteúdo principal | Recebe de | Envia para |
|--------|--------------------|-----------|------------|
| L1 | Axônios distais, poucos corpos celulares | Feedback de longe | — |
| L2/3 | Neurônios piramidais médios | L4 local + L2/3 de outras colunas | Outras colunas e áreas |
| L4 | Células estreladas | **Tálamo / áreas inferiores (feedforward)** | L2/3, L5/6 |
| L5 | Neurônios piramidais gigantes | L2/3, feedback superior | **Subcórtex, medula, córtex motor** |
| L6 | Neurônios fusiformes | L5, feedback | **Tálamo** (modula o que chega) |

**A unidade de computação — a minicoluna:**
Vernon Mountcastle (1957) descobriu que o córtex é composto de colunas verticais (~80-100 neurônios)
que respondem à mesma feature do input, independente de qual camada. Colunas adjacentes
respondem a features similares — formando um **mapa** da entrada.

**A regra que guia a sua implementação:**
- **L4** detecta o padrão atual (feedforward — "o que está chegando agora")
- **L2/3** prediz o próximo padrão (aprende sequências — "o que vai vir")
- **L5** envia para níveis superiores (abstração progressiva)
- **L6** fecha o loop de atenção com o tálamo

**Recursos gratuitos:**
- YouTube: "Two Minute Neuroscience: Cerebral Cortex" (~2min — comece por aqui)
- YouTube: "HTM School Episode 10 - Cortical Columns" (canal Numenta, ~40min — **o mais importante**)
- Scholarpedia: busque "Neocortex" — artigo revisado por especialistas, gratuito
- Paper: Mountcastle 1997 "The columnar organization of the neocortex" — ResearchGate, gratuito

### ⚗️ Exercício Prático (1h30)

```python
# cortical_column.py — Semana 13
import numpy as np

class CorticalLayer:
    """Uma camada dentro de uma coluna cortical."""
    def __init__(self, n_neurons, layer_name, learning_rate=0.01):
        self.name = layer_name
        self.n = n_neurons
        self.lr = learning_rate
        self.activity = np.zeros(n_neurons)
        self.weights_in = None  # definido ao conectar

    def activate(self, input_vec, threshold=0.0):
        """Ativa a camada dado um input."""
        if self.weights_in is not None:
            raw = self.weights_in @ input_vec
            # ReLU com threshold — modelo de condutância simplificado
            self.activity = np.maximum(0, raw - threshold)
        return self.activity

    def learn_hebbian(self, input_vec):
        """Hebb com normalização multiplicativa — igual ao Vol. I."""
        if self.weights_in is not None and self.activity.sum() > 0:
            dW = np.outer(self.activity, input_vec) * self.lr
            self.weights_in += dW
            # Normaliza para evitar explosão
            norms = np.linalg.norm(self.weights_in, axis=1, keepdims=True)
            self.weights_in /= (norms + 1e-8)


class CorticalColumn:
    """
    Coluna cortical com 4 camadas funcionais.

    Fluxo feedforward:  input → L4 → L2/3 → L5 (output)
    Fluxo preditivo:    L2/3 gera predição → erro calculado em L4
    """
    def __init__(self, col_id, input_size, n_cells=100):
        self.id = col_id

        # L4: detecta features do input atual (feedforward)
        self.L4 = CorticalLayer(n_cells, 'L4', learning_rate=0.01)
        self.L4.weights_in = np.random.randn(n_cells, input_size) * 0.1

        # L2/3: associação e predição (aprende mais devagar)
        self.L23 = CorticalLayer(n_cells, 'L2/3', learning_rate=0.005)
        self.L23.weights_in = np.random.randn(n_cells, n_cells) * 0.1

        # L5: saída para áreas superiores
        self.L5 = CorticalLayer(n_cells // 2, 'L5', learning_rate=0.001)
        self.L5.weights_in = np.random.randn(n_cells // 2, n_cells) * 0.1

        # Estado preditivo
        self.prediction = np.zeros(n_cells)       # o que L2/3 espera ver em L4
        self.prediction_error = np.zeros(n_cells) # diferença entre previsto e real

    def forward(self, feedforward_input, top_down_context=None):
        """
        Processa um input. Retorna representação de L5.

        feedforward_input : sinal de área inferior ou sensor
        top_down_context  : predição da área superior (opcional)
        """
        # L4: detecta o input atual
        l4_act = self.L4.activate(feedforward_input)

        # Erro de predição: diferença entre o que chegou e o que L2/3 esperava
        self.prediction_error = l4_act - self.prediction

        # L2/3: combina L4 com erro de predição
        l23_input = l4_act + 0.3 * self.prediction_error
        if top_down_context is not None:
            l23_input += 0.2 * top_down_context[:len(l23_input)]
        l23_act = self.L23.activate(l23_input)

        # Atualiza predição para o próximo timestep
        self.prediction = l23_act.copy()

        # L5: representação de saída
        l5_act = self.L5.activate(l23_act)
        return l5_act

    def learn(self, feedforward_input):
        """Atualiza pesos de L4 e L2/3."""
        self.L4.learn_hebbian(feedforward_input)
        self.L23.learn_hebbian(self.L4.activity)

    @property
    def output(self):
        return self.L5.activity

    @property
    def pred_error_magnitude(self):
        return np.mean(np.abs(self.prediction_error))


# ─── Script de teste ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(42)

    N_COLUMNS = 8
    INPUT_DIM  = 50

    columns = [CorticalColumn(col_id=i, input_size=INPUT_DIM) for i in range(N_COLUMNS)]

    # Simular 200 passos com dois estímulos alternados
    stim_A = (np.arange(INPUT_DIM) % 2 == 0).astype(float)
    stim_B = (np.arange(INPUT_DIM) % 2 == 1).astype(float)
    stimuli = [stim_A, stim_B] * 100

    pred_errors = []
    for t, stim in enumerate(stimuli):
        errs = []
        for col in columns:
            col.forward(stim)
            col.learn(stim)
            errs.append(col.pred_error_magnitude)
        pred_errors.append(np.mean(errs))

    # O erro de predição deve cair com o tempo (coluna aprende o padrão)
    plt.figure(figsize=(10, 4))
    plt.plot(pred_errors, alpha=0.7)
    plt.xlabel('Passo de tempo')
    plt.ylabel('Erro de predição médio')
    plt.title('CorticalColumn: Erro de Predição Caindo com Aprendizado')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('semana13_cortical_column.png')
    print("Salvo: semana13_cortical_column.png")
    print(f"Erro inicial: {pred_errors[0]:.4f}  |  Erro final: {pred_errors[-1]:.4f}")
```

### ✅ Como saber se absorveu

- [ ] O erro de predição decresce ao longo do tempo (coluna aprende o padrão alternado)
- [ ] Você consegue explicar em voz alta por que L4 recebe do tálamo enquanto L2/3 envia para outras colunas
- [ ] Você entende que a **coluna** é a unidade computacional (não o neurônio individual)
- [ ] **Conexão obrigatória:** escreva um comentário no seu projeto explicando qual parte do seu código corresponde a L4 (detectar) e qual corresponde a L2/3 (prever)

---

## SEMANA 14
**Tema: SDR no Neocórtex — A Linguagem de Alta Capacidade**

### 📚 O que estudar (2h)

No Volume I você usou esparsidade (k-WTA) como mecanismo no DG.
Agora você vai entender **por que** o cérebro inventou representações esparsas — e o que é matematicamente poderoso nelas.

**SDR — Sparse Distributed Representation:**
- **Sparse:** apenas 1-5% dos neurônios ativos em qualquer momento
- **Distributed:** cada conceito é representado por MUITOS neurônios juntos (não um único)
- **Representation:** o padrão de ativação *é* o significado

**As 4 propriedades que tornam SDR poderoso para o seu projeto:**

1. **Capacidade exponencial:**
   Com N neurônios e k ativos → C(N, k) padrões possíveis
   `N=2048, k=40` → ~10^93 padrões (mais que átomos no universo observável)
   Compare: representação local (1-hot) → apenas 2048 padrões

2. **Sobreposição semântica:**
   Padrões com significados similares têm mais bits em comum
   "gato" e "felino" têm alto overlap. "gato" e "automóvel" têm overlap próximo de zero.

3. **Robustez ao ruído:**
   20% dos bits errados → ainda é o mesmo padrão (overlap alto)
   Você já explorou isso no DG do Vol. I — agora tem o vocabulário formal.

4. **Suporte nativo a sequências:**
   SDRs de timesteps diferentes podem ser sobrepostos para representar o histórico — base do Temporal Memory (Semana 16).

**Recursos gratuitos:**
- Paper: Numenta "Properties of Sparse Distributed Representations and their Role in HTM" — **arxiv.org/abs/1503.07469** (gratuito, leitura de 40min)
- YouTube: "HTM School Episode 1 - Sparse Distributed Representations" — canal Numenta, ~25min (**ESSENCIAL**)
- YouTube: "HTM School Episode 2 - SDR Capacity and Comparison" — ~20min

### ⚗️ Exercício Prático (1h30)

```python
# sdr.py — Semana 14
import numpy as np
import math

class SDR:
    """
    Sparse Distributed Representation — tipo de dado fundamental do neocórtex.

    Conexão com Vol. I:
      No DG, k-WTA já produzia SDRs implicitamente.
      Esta classe formaliza o SDR como tipo com operações definidas —
      vai ser usado pelo SpatialPooler (Sem. 15) e TemporalMemory (Sem. 16).
    """
    def __init__(self, size: int, active_bits: np.ndarray = None):
        self.size = size
        self.active = np.array(active_bits, dtype=int) if active_bits is not None \
                      else np.array([], dtype=int)

    # ── propriedades ──────────────────────────────────────────────────────────
    @property
    def density(self) -> float:
        """Fração de bits ativos (~2% é biológico para neocórtex)."""
        return len(self.active) / self.size

    @property
    def n_active(self) -> int:
        return len(self.active)

    @property
    def as_binary(self) -> np.ndarray:
        vec = np.zeros(self.size, dtype=float)
        if len(self.active) > 0:
            vec[self.active] = 1.0
        return vec

    # ── operações ─────────────────────────────────────────────────────────────
    def overlap(self, other: 'SDR') -> int:
        """Bits em comum. Overlap alto ↔ semanticamente similar."""
        return int(np.intersect1d(self.active, other.active).size)

    def overlap_score(self, other: 'SDR') -> float:
        """Overlap normalizado pelo menor conjunto ativo (0-1)."""
        denom = min(self.n_active, other.n_active)
        return self.overlap(other) / denom if denom > 0 else 0.0

    def is_match(self, other: 'SDR', theta: float = 0.5) -> bool:
        """True se overlap_score ≥ theta → 'é o mesmo conceito'."""
        return self.overlap_score(other) >= theta

    # ── construtores estáticos ─────────────────────────────────────────────────
    @staticmethod
    def random(size: int, n_active: int, rng=None) -> 'SDR':
        if rng is None:
            rng = np.random.default_rng()
        bits = rng.choice(size, n_active, replace=False)
        return SDR(size, bits)

    def add_noise(self, noise_frac: float, rng=None) -> 'SDR':
        """Versão ruidosa: noise_frac dos bits ativos são substituídos."""
        if rng is None:
            rng = np.random.default_rng()
        n_flip = int(self.n_active * noise_frac)
        if n_flip == 0:
            return SDR(self.size, self.active.copy())
        keep = rng.choice(self.n_active, self.n_active - n_flip, replace=False)
        kept = self.active[keep]
        inactive = np.setdiff1d(np.arange(self.size), self.active)
        new_bits = rng.choice(inactive, n_flip, replace=False)
        return SDR(self.size, np.concatenate([kept, new_bits]))

    # ── estatísticas ───────────────────────────────────────────────────────────
    @staticmethod
    def capacity_log10(n_bits: int, n_active: int) -> float:
        """log10 do número de SDRs únicos possíveis = log10(C(N,k))."""
        return sum(math.log10(n_bits - i) - math.log10(i + 1) for i in range(n_active))

    @staticmethod
    def false_match_probability(n: int, k: int, theta: float = 0.5) -> float:
        """Probabilidade de dois SDRs aleatórios se confundirem (analítica)."""
        # Aproximação: P(overlap ≥ θk) com distribuição hipergeométrica
        from scipy.stats import hypergeom
        threshold = int(theta * k)
        rv = hypergeom(n, k, k)
        return float(1 - rv.cdf(threshold - 1))


# ─── Demonstração ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    N, K = 2048, 40
    rng  = np.random.default_rng(42)

    # 1. Capacidade
    cap = SDR.capacity_log10(N, K)
    print(f"SDR({N}, {K}) — capacidade: 10^{cap:.1f} padrões únicos")
    print(f"Representação local (1-hot):   10^{math.log10(N):.1f} padrões\n")

    # 2. Robustez ao ruído
    original = SDR.random(N, K, rng)
    noise_levels = np.linspace(0, 1, 20)
    overlaps     = [original.overlap_score(original.add_noise(nl, rng)) for nl in noise_levels]

    plt.figure(figsize=(8, 4))
    plt.plot(noise_levels * 100, overlaps, 'b-o', linewidth=2, markersize=4)
    plt.axhline(0.5, color='red', linestyle='--', label='Threshold θ=0.5')
    plt.fill_between(noise_levels * 100, overlaps, 0.5,
                     where=np.array(overlaps) >= 0.5, alpha=0.2, color='green',
                     label='Match (mesmo conceito)')
    plt.xlabel('Ruído adicionado (%)')
    plt.ylabel('Overlap score com original')
    plt.title('SDR: Robustez ao Ruído')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('semana14_sdr_robustez.png')
    print("Salvo: semana14_sdr_robustez.png")

    # 3. Taxa de falso-positivo entre aleatórios
    false_matches = sum(
        1 for _ in range(10_000)
        if SDR.random(N, K, rng).is_match(SDR.random(N, K, rng))
    )
    print(f"Falso-positivos entre SDRs aleatórios: {false_matches}/10000 = {false_matches/100:.2f}%")
    print("(Esperado: ~0% — SDRs aleatórios quase nunca se confundem)")
```

### ✅ Como saber se absorveu

- [ ] A capacidade de SDR(2048, 40) é astronomicamente maior que representação local — você calculou
- [ ] A curva de robustez mostra que SDRs aceitam até ~30-40% de ruído e ainda reconhecem o padrão
- [ ] A taxa de falso-positivo entre SDRs aleatórios é < 0.1%
- [ ] **Conexão obrigatória:** escreva um comentário no código do seu DG (Vol. I) explicando que o k-WTA estava criando SDRs — e que agora você entende matematicamente *por que* isso funciona

---

## SEMANA 15
**Tema: Spatial Pooler — O Córtex Aprende a Representar**

### 📚 O que estudar (2h)

O **Spatial Pooler (SP)** é a implementação computacional de L4 do neocórtex: ele transforma qualquer input em um SDR estável e esparso.

**O problema que o SP resolve:**
- Inputs similares → SDRs com alta sobreposição (mesmo conceito)
- Inputs diferentes → SDRs com baixa sobreposição
- A representação deve ser **estável** após aprendizado
- Todas as colunas devem ser usadas (sem "neurônios mortos")

**Algoritmo (simplificado):**
```
Para cada input:
  1. Para cada coluna c:
       overlap(c) = soma das permanências conectadas * bits ativos do input
       overlap(c) *= boost(c)    ← correção homeostática

  2. k-WTA → SDR: top-k colunas ficam ativas

  3. Aprende (colunas ativas):
       bit de input ativo   → permanência += Δ+
       bit de input inativo → permanência -= Δ-

  4. Atualiza boost:
       colunas usadas raramente → boost aumenta (homeostase)
```

**Por que "permanências" e não "pesos"?**
Biologicamente, uma sinapse não tem "peso" contínuo — ela existe ou não.
A permanência modela a *probabilidade de conexão*: `permanência > limiar` = sinapse conectada.
Isso é mais biológico que pesos contínuos, e mais robusto a ruído.

**Recursos gratuitos:**
- YouTube: "HTM School Episode 3 - Spatial Pooler" (Numenta, ~30min)
- YouTube: "HTM School Episode 4 - Boosting" (~25min)
- Paper: Numenta "Evaluating Real-Time Anomaly Detection Algorithms" — numenta.com/research (gratuito)

### ⚗️ Exercício Prático (1h30)

```python
# spatial_pooler.py — Semana 15
import numpy as np

class SpatialPooler:
    """
    Spatial Pooler: L4 do neocórtex.

    Converte input binário arbitrário em SDR estável e esparso.
    Versão completa do k-WTA que você implementou no DG (Vol. I),
    agora com permanências (modelo biológico) e boosting sistematizado.
    """
    def __init__(self,
                 input_size:      int,
                 column_count:    int   = 2048,
                 active_frac:     float = 0.02,   # ~2% de colunas ativas
                 potential_pct:   float = 0.5,    # cada col. conecta a 50% do input
                 init_perm:       float = 0.5,
                 perm_increment:  float = 0.05,
                 perm_decrement:  float = 0.01,
                 perm_threshold:  float = 0.5,    # abaixo = sinapse desconectada
                 boost_strength:  float = 2.0):

        self.input_size    = input_size
        self.n_cols        = column_count
        self.k             = max(1, int(column_count * active_frac))
        self.perm_inc      = perm_increment
        self.perm_dec      = perm_decrement
        self.perm_thresh   = perm_threshold
        self.boost_str     = boost_strength
        self.target_dens   = active_frac

        n_pot = max(1, int(input_size * potential_pct))

        # potential_pools[c] = índices do input aos quais a coluna c pode se conectar
        self.potential_pools = np.array([
            np.random.choice(input_size, n_pot, replace=False)
            for _ in range(column_count)
        ])  # shape: (column_count, n_pot)

        # Permanências: inicializadas próximo ao threshold + ruído pequeno
        self.permanences = np.clip(
            np.random.randn(column_count, n_pot) * 0.1 + init_perm,
            0.0, 1.0
        )  # shape: (column_count, n_pot)

        # Frequência de ativação (para boosting)
        self.activation_frequency = np.full(column_count, active_frac)
        self.iteration = 0

    # ── core ──────────────────────────────────────────────────────────────────

    def _compute_overlaps(self, input_sdr: np.ndarray) -> np.ndarray:
        """
        Overlap de cada coluna com o input.
        Vetorizado: sem loop Python explícito.
        """
        # Busca os bits de input para cada pool → shape (n_cols, n_pot)
        input_at_pools = input_sdr[self.potential_pools]
        # Sinapses conectadas (permanência >= threshold)
        connected = self.permanences >= self.perm_thresh
        return np.sum(input_at_pools * connected, axis=1)  # shape: (n_cols,)

    def _boost(self) -> np.ndarray:
        """
        Boost homeostático.
        Colunas abaixo da densidade alvo recebem boost > 1.
        Evita colunas mortas (nunca ativas).
        """
        return np.exp(self.boost_str * (self.target_dens - self.activation_frequency))

    def _learn(self, input_sdr: np.ndarray, active: np.ndarray):
        """Atualiza permanências das colunas ativas."""
        for c in np.where(active)[0]:
            pool       = self.potential_pools[c]
            inp_active = input_sdr[pool].astype(bool)
            self.permanences[c][inp_active]  += self.perm_inc
            self.permanences[c][~inp_active] -= self.perm_dec
            self.permanences[c]               = np.clip(self.permanences[c], 0.0, 1.0)

    def compute(self, input_sdr: np.ndarray, learn: bool = True) -> np.ndarray:
        """
        Processa um input e retorna SDR de colunas ativas.

        input_sdr : array binário de tamanho input_size
        learn     : se True, atualiza permanências e frequências
        Retorna   : array binário de tamanho column_count
        """
        input_sdr  = np.asarray(input_sdr, dtype=float)
        self.iteration += 1

        overlaps  = self._compute_overlaps(input_sdr)
        boosted   = overlaps * self._boost()

        # k-WTA
        active = np.zeros(self.n_cols, dtype=float)
        if boosted.max() > 0:
            top_k = np.argpartition(boosted, -self.k)[-self.k:]
            active[top_k] = 1.0

        # Atualizar frequência de ativação (média móvel)
        alpha = 0.001
        self.activation_frequency = (1 - alpha) * self.activation_frequency + alpha * active

        if learn:
            self._learn(input_sdr, active)

        return active

    # ── métricas ──────────────────────────────────────────────────────────────

    def stability_score(self, input_sdr: np.ndarray, n_trials: int = 5) -> float:
        """Overlap médio entre SDRs gerados para o mesmo input (1.0 = perfeito)."""
        sdrs = [self.compute(input_sdr, learn=False) for _ in range(n_trials)]
        overlaps = []
        for i in range(len(sdrs)):
            for j in range(i + 1, len(sdrs)):
                a, b = sdrs[i], sdrs[j]
                ov = np.sum(a * b) / (self.k + 1e-8)
                overlaps.append(ov)
        return float(np.mean(overlaps)) if overlaps else 0.0


# ─── Demonstração ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(42)

    INPUT_SIZE    = 512
    COLUMN_COUNT  = 2048

    sp = SpatialPooler(input_size=INPUT_SIZE, column_count=COLUMN_COUNT, active_frac=0.02)

    # Criar 5 padrões distintos
    patterns = {
        c: (np.arange(INPUT_SIZE) % 5 == i).astype(float)
        for i, c in enumerate('ABCDE')
    }

    # Treinar 1000 iterações
    print("Treinando Spatial Pooler...")
    stab_log = []
    for t in range(1000):
        for p in patterns.values():
            sp.compute(p, learn=True)
        if t % 100 == 99:
            stab = np.mean([sp.stability_score(p) for p in patterns.values()])
            stab_log.append((t + 1, stab))
            print(f"  iter {t+1:4d} | estabilidade: {stab:.4f}")

    # Verificar separação entre padrões
    print("\n=== Overlap entre SDRs (deve ser baixo para padrões diferentes) ===")
    sdrs = {c: sp.compute(p, learn=False) for c, p in patterns.items()}
    names = list(sdrs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            ov = np.sum(sdrs[n1] * sdrs[n2]) / sp.k
            print(f"  {n1} ↔ {n2}: overlap = {ov:.3f}")

    # Verificar uso das colunas
    dead = int(np.sum(sp.activation_frequency < 0.001))
    print(f"\nColunas mortas: {dead}/{COLUMN_COUNT}")

    iters, stabs = zip(*stab_log)
    plt.figure(figsize=(8, 4))
    plt.plot(iters, stabs, 'g-o', linewidth=2)
    plt.xlabel('Iterações de Treino')
    plt.ylabel('Estabilidade média')
    plt.title('Spatial Pooler: Convergência para SDRs Estáveis')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('semana15_sp_convergencia.png')
    print("\nSalvo: semana15_sp_convergencia.png")
    print("→ Estabilidade deve superar 0.90 após 1000 iterações")
```

### ✅ Como saber se absorveu

- [ ] Estabilidade média > 0.90 após 1000 iterações (mesmo input → mesmo SDR)
- [ ] Overlap entre padrões diferentes < 0.20
- [ ] Colunas mortas próximo de 0 (boosting funcionando)
- [ ] Você consegue explicar a diferença entre "permanência" (modelo biológico) e "peso" (redes neurais clássicas)
- [ ] **Desafio:** desligue o boosting (`boost_strength=0`) e observe o que acontece com as colunas mortas

---

## SEMANA 16
**Tema: Temporal Memory — O Córtex Prediz o Futuro**

### 📚 O que estudar (2h)

A descoberta central de Jeff Hawkins: **o neocórtex é, fundamentalmente, uma máquina de fazer previsões**.

Cada nível do córtex faz duas coisas simultaneamente:
1. **Reconhece** o padrão atual
2. **Prediz** qual padrão virá a seguir

Quando a predição está correta → ativação silenciosa, só o erro (pequeno) sobe.
Quando erra → sinal de erro sobe forte para o nível acima ("preste atenção, algo inesperado!").

**O mecanismo do Temporal Memory (TM):**

Dentro de cada minicoluna existem N células (uma por camada/contexto).
Cada célula representa o *mesmo feature* de L4, mas em um **contexto temporal diferente**.

- Célula 1 da coluna 42: feature 42 precedida pelo padrão XYZ
- Célula 2 da coluna 42: feature 42 precedida pelo padrão ABC
- ...

Quando a coluna 42 fica ativa:
- **Burst** (todas as células ativam) → contexto desconhecido = **novidade**
- **Ativação precisa** (só 1 célula) → contexto reconhecido = **previsto**

O **anomaly score** é a fração de colunas que entrou em burst — quanto maior, mais surpreendente foi o input.

**Recursos gratuitos:**
- YouTube: "HTM School Episode 7 - Temporal Memory Part 1" (canal Numenta)
- YouTube: "HTM School Episode 8 - Temporal Memory Part 2"
- Paper: Hawkins & Ahmad 2016 "Why Neurons Have Thousands of Synapses, A Theory of Sequence Memory in Neocortex" — **numenta.com/research/papers** (gratuito)

### ⚗️ Exercício Prático (1h30)

```python
# temporal_memory.py — Semana 16
import numpy as np
from collections import defaultdict

class TemporalMemory:
    """
    HTM Temporal Memory: L2/3 do neocórtex.
    Aprende e prediz sequências de SDRs.

    Trabalha com o SpatialPooler:
      input → SP (SDR de colunas) → TM (células contextuais + predições)

    A TM responde: "dado o que está acontecendo agora E o que aconteceu antes,
                    o que vai acontecer a seguir?"
    """
    def __init__(self,
                 n_columns:            int,
                 cells_per_column:     int   = 32,
                 activation_threshold: int   = 13,
                 learning_threshold:   int   = 10,
                 initial_permanence:   float = 0.21,
                 perm_increment:       float = 0.10,
                 perm_decrement:       float = 0.10,
                 max_segments_per_cell:int   = 128,
                 max_synapses_per_seg: int   = 32):

        self.n_cols       = n_columns
        self.cells_per    = cells_per_column
        self.n_cells      = n_columns * cells_per_column
        self.act_thresh   = activation_threshold
        self.lrn_thresh   = learning_threshold
        self.init_perm    = initial_permanence
        self.perm_inc     = perm_increment
        self.perm_dec     = perm_decrement
        self.max_segs     = max_segments_per_cell
        self.max_syns     = max_synapses_per_seg

        # Estado
        self.active_cells    : set = set()
        self.winner_cells    : set = set()
        self.predicted_cells : set = set()
        self._prev_active    : set = set()
        self._prev_winners   : set = set()

        # Segmentos dendríticos distais: {cell_id: [{cell_id: permanence}, ...]}
        self.segments: dict = defaultdict(list)

        self.anomaly_score: float = 1.0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _cid(self, col: int, cell: int) -> int:
        return col * self.cells_per + cell

    def _col_cells(self, col: int):
        start = col * self.cells_per
        return range(start, start + self.cells_per)

    def _seg_overlap(self, seg: dict, cells: set) -> int:
        return sum(1 for c, p in seg.items() if c in cells and p >= 0.5)

    def _best_segment(self, cell_id: int, cells: set):
        best_idx, best_ov = -1, 0
        for idx, seg in enumerate(self.segments[cell_id]):
            ov = self._seg_overlap(seg, cells)
            if ov > best_ov:
                best_ov, best_idx = ov, idx
        return best_idx, best_ov

    # ── core ──────────────────────────────────────────────────────────────────

    def compute(self, active_columns: np.ndarray, learn: bool = True):
        """
        Processa um timestep dado as colunas ativas do SP.
        Retorna: (active_cells, anomaly_score)
        """
        self._prev_active  = self.active_cells.copy()
        self._prev_winners = self.winner_cells.copy()

        new_active  : set = set()
        new_winner  : set = set()
        n_bursting  : int = 0

        for col in np.where(active_columns)[0]:
            predicted_here = [c for c in self._col_cells(col)
                              if c in self.predicted_cells]

            if predicted_here:
                # Ativação precisa — sequência reconhecida
                new_active.update(predicted_here)
                new_winner.update(predicted_here)
            else:
                # Burst — novidade!
                n_bursting += 1
                all_cells = list(self._col_cells(col))
                new_active.update(all_cells)
                # Winner = célula com melhor segmento para o contexto anterior
                best_cell, best_ov = all_cells[0], -1
                for cell in all_cells:
                    _, ov = self._best_segment(cell, self._prev_active)
                    if ov > best_ov:
                        best_ov, best_cell = ov, cell
                new_winner.add(best_cell)

        n_active = int(active_columns.sum())
        self.anomaly_score = n_bursting / n_active if n_active > 0 else 1.0

        self.active_cells = new_active
        self.winner_cells = new_winner

        if learn:
            self._learn()
        self._compute_predictions()

        return self.active_cells, self.anomaly_score

    def _learn(self):
        """Reforça e cria segmentos nas células winner."""
        for cell in self.winner_cells:
            best_idx, best_ov = self._best_segment(cell, self._prev_active)

            if best_idx >= 0 and best_ov >= self.lrn_thresh:
                # Reforçar segmento existente
                seg = self.segments[cell][best_idx]
                for c in list(seg):
                    if c in self._prev_active:
                        seg[c] = min(1.0, seg[c] + self.perm_inc)
                    else:
                        seg[c] = max(0.0, seg[c] - self.perm_dec)
            elif len(self.segments[cell]) < self.max_segs and self._prev_winners:
                # Criar novo segmento
                n_syns = min(self.max_syns, len(self._prev_winners))
                sampled = np.random.choice(list(self._prev_winners), n_syns, replace=False)
                self.segments[cell].append({int(c): self.init_perm for c in sampled})

    def _compute_predictions(self):
        """Atualiza células previstas para o próximo timestep."""
        self.predicted_cells = set()
        for cell_id, segs in self.segments.items():
            for seg in segs:
                if self._seg_overlap(seg, self.active_cells) >= self.act_thresh:
                    self.predicted_cells.add(cell_id)
                    break

    def reset(self):
        """Reseta estado entre sequências independentes."""
        self.active_cells    = set()
        self.winner_cells    = set()
        self.predicted_cells = set()
        self._prev_active    = set()
        self._prev_winners   = set()


# ─── Demonstração ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(42)

    INPUT_SIZE   = 200
    N_COLUMNS    = 512

    sp = SpatialPooler(input_size=INPUT_SIZE, column_count=N_COLUMNS, active_frac=0.02)
    tm = TemporalMemory(n_columns=N_COLUMNS, cells_per_column=16,
                        activation_threshold=8, learning_threshold=5)

    # Sequência ABCDABCD…
    seq_inputs = {
        c: (np.arange(INPUT_SIZE) % 4 == i).astype(float)
        for i, c in enumerate('ABCD')
    }
    sequence = list('ABCDABCD') * 3

    anomaly_by_epoch = []
    print("Treinando na sequência ABCD...")
    for epoch in range(40):
        tm.reset()
        epoch_anom = []
        for symbol in sequence:
            sdr = sp.compute(seq_inputs[symbol], learn=True)
            _, anom = tm.compute(sdr, learn=True)
            epoch_anom.append(anom)
        anomaly_by_epoch.append(np.mean(epoch_anom))
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | Anomaly score: {anomaly_by_epoch[-1]:.4f}")

    print(f"\nAnomaly inicial: {anomaly_by_epoch[0]:.4f}")
    print(f"Anomaly final:   {anomaly_by_epoch[-1]:.4f}")
    print("→ Score deve cair (sequência aprendida)")

    # Detecção de anomalia: inserir 'X' inesperado
    X_input = (np.random.rand(INPUT_SIZE) > 0.7).astype(float)
    seq_inputs['X'] = X_input
    tm.reset()

    print("\n=== Detectando anomalias na sequência ABCXABCD ===")
    for sym in 'ABCXABCD':
        sdr  = sp.compute(seq_inputs[sym], learn=False)
        _, a = tm.compute(sdr, learn=False)
        flag = "  ← ANOMALIA!" if a > 0.7 else ""
        print(f"  {sym}: anomaly = {a:.3f}{flag}")

    # Gráfico
    plt.figure(figsize=(8, 4))
    plt.plot(anomaly_by_epoch, 'r-o', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Anomaly score médio')
    plt.title('Temporal Memory: Aprendendo a Sequência ABCD')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('semana16_tm_aprendizado.png')
    print("\nSalvo: semana16_tm_aprendizado.png")
```

### ✅ Como saber se absorveu

- [ ] O anomaly score cai de >0.8 para <0.3 ao longo dos epochs (sequência aprendida)
- [ ] A letra 'X' gera anomaly score > 0.7 (novidade detectada)
- [ ] Você entende a diferença entre "burst" (novidade) e "ativação precisa" (previsto)
- [ ] Você consegue explicar por que são necessárias **múltiplas células por coluna** para aprender sequências (resposta: cada célula representa o mesmo feature em um *contexto* diferente)

---

## ✅ CHECKLIST DO MÊS 4 (Neocórtex)

- [ ] CorticalColumn com 4 camadas implementada — erro de predição cai com treinamento
- [ ] SDR implementado como tipo de dado com overlap, robustez e cálculo de capacidade
- [ ] SpatialPooler com permanências, k-WTA e boosting homeostático — estabilidade > 0.90
- [ ] TemporalMemory com burst, predição, anomaly score — aprende sequência ABCD
- [ ] Detecção de anomalia funcionando (input inesperado → score alto)
- [ ] Você consegue explicar a função biológica de cada módulo implementado

**Próximo:** Fazer o neocórtex e o hipocampo conversarem.
