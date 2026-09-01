# 🧠 Trilha de Neurobiologia para IA Bioinspirada
## Volume II — Mês 6: Pesquisa Aplicada

> **Pré-requisito:** Gates 1-5 passados. Sistema integrado rodando.
>
> **O que muda agora:** Você para de ser aluno e começa a ser pesquisador.
> Pesquisador não é quem sabe tudo — é quem tem **perguntas boas**, **métricas honestas**
> e **documentação suficiente para não perder o próprio trabalho**.

---

---

# MÊS 6 — PESQUISA APLICADA

---

## SEMANA 22
**Tema: Benchmarking — Comparando Seu Sistema com a Biologia**

### 📚 O que estudar (2h)

Você tem um sistema. Agora precisa saber **o quão biológico ele é** — não por vaidade,
mas porque os números vão dizer exatamente o que consertar.

**Benchmarks essenciais para o seu hipocampo:**

| Benchmark | Referência biológica | Métrica no seu código |
|---|---|---|
| Pattern separation ratio | DG reduz overlap em ~70% (Leutgeb 2007) | `overlap_CA1 / overlap_EC` |
| One-shot learning | Recall após 1 exposição | `n_epochs=1` |
| Capacidade CA3 | ~0.14N padrões (Hopfield 1982) | recall accuracy vs N_stored |
| Forgetting curve | Decaimento exponencial (Ebbinghaus) | overlap vs time since storage |

**Benchmarks essenciais para o seu neocórtex:**

| Benchmark | Referência biológica | Métrica no seu código |
|---|---|---|
| SDR stability | Mesmo input → mesmo SDR | overlap de 2 runs sem learn |
| Sequence prediction | Anomaly < 0.2 após aprender | `tm.anomaly_score` |
| Anomaly detection | Outlier → anomaly > 0.8 | inserir elemento inesperado |
| Dead columns | ~0% (boosting impede) | `sp.activation_frequency.min()` |

**Recursos gratuitos:**
- Paper: Leutgeb et al. 2007 "Pattern separation in the dentate gyrus and CA3" — Science; busque PubMed → PMC gratuito
- Paper: Numenta "Evaluating Real-Time Anomaly Detection Algorithms: The Numenta Anomaly Benchmark" — numenta.com/research

### ⚗️ Exercício Prático (1h30)

```python
# benchmark_suite.py — Semana 22
import numpy as np
from dataclasses import dataclass


@dataclass
class BenchResult:
    test_name:          str
    score:              float
    biological_target:  float
    passed:             bool
    notes:              str = ""


class NeuroBenchmarkSuite:
    """
    Suite de benchmarks comparando o sistema com dados biológicos.
    Rode depois de cada iteração de desenvolvimento para acompanhar progresso.
    """
    def __init__(self, system):
        self.sys     = system
        self.results: list[BenchResult] = []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_patterns(self, n: int, size: int = 100):
        return [(np.arange(size) % n == i).astype(float) for i in range(n)]

    def _overlap(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = min(a.sum(), b.sum())
        return float(np.sum(a * b) / denom) if denom > 0 else 0.0

    def _recall_similarity(self, original, recalled):
        if original.std() < 1e-8 or recalled.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(original, recalled)[0, 1])

    # ── benchmarks hipocampais ────────────────────────────────────────────────

    def test_pattern_separation(self, n_trials: int = 20) -> BenchResult:
        """
        DG deve reduzir overlap em pelo menos 60%.
        Biologia: Leutgeb et al. 2007 — redução ~70%.
        """
        ratios = []
        size   = self.sys.input_size

        for _ in range(n_trials):
            base    = (np.random.rand(size) > 0.75).astype(float)
            variant = base.copy()
            flip    = np.random.choice(size, int(size * 0.20), replace=False)
            variant[flip] = 1 - variant[flip]

            ec_overlap  = self._overlap(base, variant)

            # Processar pelo hipocampo
            if hasattr(self.sys, 'hippocampus') and hasattr(self.sys.hippocampus, 'store_episode'):
                out_base = self.sys.hippocampus.store_episode(base)
                out_var  = self.sys.hippocampus.store_episode(variant)
                ca1_overlap = self._overlap(
                    (out_base > 0.5).astype(float),
                    (out_var  > 0.5).astype(float)
                )
            else:
                # Fallback: mede via SP (separação cortical)
                sdr_base = self.sys.sp.compute(base,    learn=False)
                sdr_var  = self.sys.sp.compute(variant, learn=False)
                ca1_overlap = self._overlap(sdr_base, sdr_var)

            if ec_overlap > 0:
                ratios.append(ca1_overlap / ec_overlap)

        mean_ratio = float(np.mean(ratios)) if ratios else 1.0
        passed     = mean_ratio < 0.40   # saída deve ter < 40% do overlap do input

        result = BenchResult(
            test_name         = "Pattern Separation",
            score             = mean_ratio,
            biological_target = 0.30,
            passed            = passed,
            notes             = f"Ratio médio: {mean_ratio:.3f} | alvo: < 0.40"
        )
        self.results.append(result)
        return result

    def test_one_shot_learning(self, n_patterns: int = 8) -> BenchResult:
        """
        Hipocampo deve recordar após UMA única exposição.
        """
        patterns = self._make_patterns(n_patterns, self.sys.input_size)
        recalls  = []

        for p in patterns:
            if hasattr(self.sys, 'hippocampus') and hasattr(self.sys.hippocampus, 'store_episode'):
                self.sys.hippocampus.store_episode(p)
            else:
                self.sys.hippocampus.store(p)

        for p in patterns:
            noisy = p.copy()
            noisy[np.random.choice(len(p), int(len(p)*0.3), replace=False)] = \
                1 - noisy[np.random.choice(len(p), int(len(p)*0.3), replace=False)]

            result = self.sys.recall(noisy)
            sim    = self._recall_similarity(p, result['episodic_recall'])
            recalls.append(sim)

        mean_r = float(np.mean(recalls))
        passed = mean_r > 0.50

        result = BenchResult(
            test_name         = "One-Shot Learning",
            score             = mean_r,
            biological_target = 0.70,
            passed            = passed,
            notes             = f"Recall médio (30% ruído): {mean_r:.3f}"
        )
        self.results.append(result)
        return result

    # ── benchmarks corticais ──────────────────────────────────────────────────

    def test_sdr_stability(self, n_trials: int = 5) -> BenchResult:
        """
        Mesmo input → mesmo SDR após treinamento.
        """
        test_input = (np.arange(self.sys.input_size) % 3 == 0).astype(float)

        # Treinar um pouco antes
        for _ in range(200):
            self.sys.sp.compute(test_input, learn=True)

        # Verificar estabilidade
        sdrs     = [self.sys.sp.compute(test_input, learn=False) for _ in range(n_trials)]
        overlaps = []
        k        = self.sys.sp.k
        for i in range(len(sdrs)):
            for j in range(i + 1, len(sdrs)):
                ov = float(np.sum(sdrs[i] * sdrs[j])) / (k + 1e-8)
                overlaps.append(ov)

        score  = float(np.mean(overlaps)) if overlaps else 0.0
        passed = score > 0.90

        result = BenchResult(
            test_name         = "SDR Stability (Spatial Pooler)",
            score             = score,
            biological_target = 0.95,
            passed            = passed,
            notes             = f"Overlap interno: {score:.3f} | alvo: > 0.90"
        )
        self.results.append(result)
        return result

    def test_sequence_learning(self, seq_len: int = 5, n_epochs: int = 30) -> BenchResult:
        """
        TM deve aprender sequência: anomaly < 0.25 após treino.
        """
        patterns = self._make_patterns(seq_len, self.sys.input_size)

        for _ in range(n_epochs):
            self.sys.tm.reset()
            for p in patterns:
                sdr = self.sys.sp.compute(p, learn=True)
                self.sys.tm.compute(sdr, learn=True)

        # Avaliar
        anomalies = []
        self.sys.tm.reset()
        for p in patterns:
            sdr  = self.sys.sp.compute(p, learn=False)
            _, a = self.sys.tm.compute(sdr, learn=False)
            anomalies.append(a)

        mean_anom = float(np.mean(anomalies))
        score     = 1 - mean_anom   # invertido: score alto = bom
        passed    = mean_anom < 0.30

        result = BenchResult(
            test_name         = "Sequence Learning (Temporal Memory)",
            score             = score,
            biological_target = 0.80,
            passed            = passed,
            notes             = f"Anomaly médio: {mean_anom:.3f} | alvo: < 0.30"
        )
        self.results.append(result)
        return result

    def test_anomaly_detection(self) -> BenchResult:
        """
        Input inesperado deve gerar anomaly > 0.70.
        """
        # Treinar em sequência conhecida
        patterns = self._make_patterns(4, self.sys.input_size)
        for _ in range(50):
            self.sys.tm.reset()
            for p in patterns:
                sdr = self.sys.sp.compute(p, learn=True)
                self.sys.tm.compute(sdr, learn=True)

        # Input nunca visto
        outlier = (np.random.rand(self.sys.input_size) > 0.5).astype(float)
        sdr     = self.sys.sp.compute(outlier, learn=False)
        _, anom = self.sys.tm.compute(sdr, learn=False)

        passed = anom > 0.70

        result = BenchResult(
            test_name         = "Anomaly Detection",
            score             = float(anom),
            biological_target = 0.80,
            passed            = passed,
            notes             = f"Anomaly para input inédito: {anom:.3f}"
        )
        self.results.append(result)
        return result

    # ── relatório final ───────────────────────────────────────────────────────

    def run_all(self) -> dict:
        """Roda todos os benchmarks e imprime relatório."""
        print("═" * 58)
        print("  NEUROBIOLOGICAL BENCHMARK REPORT")
        print("═" * 58)

        tests = [
            self.test_pattern_separation,
            self.test_one_shot_learning,
            self.test_sdr_stability,
            self.test_sequence_learning,
            self.test_anomaly_detection,
        ]

        for test in tests:
            r      = test()
            status = "✓ PASS" if r.passed else "✗ FAIL"
            gap    = r.score - r.biological_target
            bar    = "█" * int(r.score * 20) + "░" * (20 - int(r.score * 20))
            print(f"\n[{status}] {r.test_name}")
            print(f"  Score: {r.score:.3f} [{bar}]")
            print(f"  Alvo:  {r.biological_target:.3f}  (gap: {gap:+.3f})")
            print(f"  Nota:  {r.notes}")

        passed = sum(1 for r in self.results if r.passed)
        print(f"\n{'═'*58}")
        print(f"  Resultado: {passed}/{len(self.results)} benchmarks aprovados")

        if passed < len(self.results):
            print("\n  Próximas prioridades de melhoria:")
            fails = sorted([r for r in self.results if not r.passed],
                           key=lambda x: x.biological_target - x.score, reverse=True)
            for r in fails:
                print(f"    → {r.test_name}: {r.score:.3f} (precisa de +{r.biological_target - r.score:.3f})")

        return {r.test_name: r.score for r in self.results}


# ─── Script ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)

    # Importe seu sistema integrado:
    # from integrated_system import NeocorticalHippocampalSystem
    # system = NeocorticalHippocampalSystem(input_size=100)

    # Para testar sem o sistema completo, crie um mock:
    class MockSystem:
        input_size = 100
        def __init__(self):
            from spatial_pooler import SpatialPooler
            from temporal_memory import TemporalMemory
            from cls_demo import FastHippocampus
            self.sp          = SpatialPooler(100, 512, 0.02)
            self.tm          = TemporalMemory(512, 16, 8, 5)
            self.hippocampus = FastHippocampus()
        def recall(self, cue):
            buf = self.hippocampus.memory
            if not buf:
                return {'episodic_recall': cue.copy()}
            sims = [np.dot(cue, p) / (np.linalg.norm(p) + 1e-8) for p in buf]
            return {'episodic_recall': buf[int(np.argmax(sims))].copy()}

    system = MockSystem()
    suite  = NeuroBenchmarkSuite(system)
    scores = suite.run_all()
```

### ✅ Como saber se absorveu

- [ ] Você rodou a suite e tem números concretos para cada benchmark
- [ ] Você identificou qual benchmark falha no seu sistema (toda pesquisa tem onde falha)
- [ ] **Obrigatório:** escreva 3 hipóteses sobre *por que* os benchmarks falham — cada hipótese deve ser testável com uma mudança de código específica

---

## SEMANA 23
**Tema: Otimizando Código Python — Fazendo o Sistema Rodar Rápido**

### 📚 O que estudar (1h)

Com 2048 colunas, 32 células/coluna, e loops de aprendizado, seu sistema vai ficar lento.
Quatro ferramentas para resolver isso:

**1. Encontrar o gargalo (profiling):**
```python
import cProfile
cProfile.run("system.experience(pattern)", sort="cumulative")
```
Mostra qual função consome mais tempo. Nunca otimize sem medir primeiro.

**2. Vetorizar com NumPy:**
```python
# Lento — loop Python (microsegundos por elemento)
for c in range(n_cols):
    overlaps[c] = np.dot(W[c], input_sdr)

# Rápido — operação matricial (nanosegundos)
overlaps = W @ input_sdr   # mesma operação, 10-100× mais rápido
```

**3. Tipos corretos:**
```python
# Lento para operações em massa
cells = set()                         # Python set

# Rápido para operações vetorizadas
cells = np.zeros(n_cells, dtype=bool) # NumPy bool array
```

**4. Pré-alocar:**
```python
# Ruim — cria novo array a cada passo
result = np.zeros(n)   # dentro do loop!

# Bom — pré-aloca fora e reutiliza
result = np.zeros(n)
for t in range(T):
    result[:] = 0      # reseta sem realocar
```

**Recursos gratuitos:**
- NumPy docs: "Performance Tips" — numpy.org/doc
- Python: documentação cProfile — docs.python.org/3/library/profile.html
- YouTube: "Python Performance Optimization" — Real Python canal

### ⚗️ Exercício Prático (1h30)

```python
# optimization.py — Semana 23
import numpy as np
import cProfile, pstats, io, time


# ─── Vetorização do SpatialPooler._compute_overlaps ───────────────────────────

def benchmark_sp_overlap():
    """
    Compara o compute_overlaps com loop vs vetorizado.
    Este é tipicamente o gargalo #1 do SpatialPooler.
    """
    N_IN   = 512
    N_COLS = 2048
    N_POT  = 256
    THRESH = 0.5

    rng            = np.random.default_rng(0)
    input_sdr      = (rng.random(N_IN) > 0.8).astype(float)
    potential_pools = rng.integers(0, N_IN, (N_COLS, N_POT))
    permanences     = rng.random((N_COLS, N_POT))

    # ── versão com loop ────────────────────────────────────────────────────
    def loop_version():
        out = np.zeros(N_COLS)
        for c in range(N_COLS):
            pool      = potential_pools[c]
            connected = permanences[c] >= THRESH
            out[c]    = np.sum(input_sdr[pool] * connected)
        return out

    # ── versão vetorizada ──────────────────────────────────────────────────
    def vectorized_version():
        input_at_pools = input_sdr[potential_pools]          # (N_COLS, N_POT)
        connected      = permanences >= THRESH                # (N_COLS, N_POT)
        return np.sum(input_at_pools * connected, axis=1)    # (N_COLS,)

    # Verificar equivalência
    r_loop = loop_version()
    r_vec  = vectorized_version()
    assert np.allclose(r_loop, r_vec), "Resultados diferentes!"

    # Medir tempo
    N_RUNS = 50

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        loop_version()
    t_loop = (time.perf_counter() - t0) / N_RUNS * 1000

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        vectorized_version()
    t_vec  = (time.perf_counter() - t0) / N_RUNS * 1000

    print(f"compute_overlaps com loop:      {t_loop:.2f} ms")
    print(f"compute_overlaps vetorizado:    {t_vec:.2f} ms")
    print(f"Speedup: {t_loop / t_vec:.1f}×")
    print("✓ Resultados idênticos")
    return t_loop, t_vec


# ─── Perfil do sistema completo ───────────────────────────────────────────────

def profile_system(system, n_steps: int = 200):
    """Perfila n_steps de experience() e mostra top gargalos."""
    pattern = (np.arange(system.input_size) % 3 == 0).astype(float)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n_steps):
        system.experience(pattern)
    pr.disable()

    s  = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(12)
    print("═══ TOP GARGALOS (cumulative time) ═══")
    print(s.getvalue())


# ─── Template de otimização pessoal ───────────────────────────────────────────
"""
ROTEIRO DE OTIMIZAÇÃO (faça nesta ordem):

1. Perfil: rode profile_system() — identifique as 3 funções mais lentas
2. Priorize: escolha a mais lenta que você consegue vetorizar
3. Meça antes: time.perf_counter() na função original
4. Vetorize: substitua loops Python por operações NumPy
5. Meça depois: calcule speedup
6. Verifique: resultados devem ser idênticos (assert np.allclose)
7. Repita para a próxima função

Meta: sistema processa 1000 passos em < 30 segundos no seu hardware.
"""

if __name__ == '__main__':
    print("═══ BENCHMARK DE VETORIZAÇÃO: compute_overlaps ═══\n")
    benchmark_sp_overlap()

    print("\n═══ Próximo passo ═══")
    print("Rode profile_system(seu_sistema) para encontrar o próximo gargalo.")
    print("Aplique o mesmo padrão: loop → operação matricial NumPy.")
```

### ✅ Como saber se absorveu

- [ ] Você rodou o profiler no seu sistema e identificou as 3 funções mais lentas
- [ ] Vetorizou pelo menos uma delas com NumPy — speedup documentado
- [ ] **Meta mínima:** sistema processa 1000 steps em < 30 segundos na sua máquina
- [ ] **Desafio:** aplique a mesma vetorização ao `_learn()` do SpatialPooler

---

## SEMANA 24
**Tema: Como Ler Papers Científicos**

### 📚 O que estudar (2h)

Um pesquisador passa ~40% do tempo lendo. Você precisa de um protocolo eficiente.

**O Protocolo das 3 Passes — Keshav (2007):**

**Pass 1 — 10 min (decidir se vale ler):**
1. Leia título + abstract + introdução
2. Leia apenas os títulos das seções
3. Leia a conclusão
4. Olhe as figuras (apenas captions)
→ Você sabe: (a) o que o paper faz, (b) se é relevante para você

**Pass 2 — 1 hora (entender sem detalhes matemáticos):**
1. Leia com atenção, mas pule demonstrações
2. Anote os pontos principais e figuras importantes
3. Marque referências para ler depois
→ Você consegue resumir em 2 frases

**Pass 3 — 4-5 horas (apenas para os papers centrais do seu projeto):**
1. Re-implemente o algoritmo principal do paper
2. Verifique cada claim com o seu código
3. Identifique limitações não mencionadas
→ Você "possui" o paper

**Para o seu projeto:** Pass 1 para todos, Pass 2 para relevantes, Pass 3 para os 3-5 mais importantes.

**Recursos gratuitos:**
- O próprio protocolo: Keshav 2007 "How to Read a Paper" — busque "Keshav how to read a paper PDF" (3 páginas, gratuito)

### ⚗️ Exercício Prático (1h30)

**Template de anotação (salve como `.md` para cada paper):**

```markdown
# Anotação: [Título completo]

**Autores:**
**Ano:**
**Journal/Conferência:**
**Link/PDF:**
**Pass realizado:** [ ] 1   [ ] 2   [ ] 3

---

## O que o paper faz (máx. 2 frases)


## Qual problema resolve


## Método principal (em português, sem jargão)


## Resultado principal (o número mais importante)


## Limitações reconhecidas pelos autores


## Como afeta meu projeto (2-3 frases)


## O que devo re-implementar deste paper


## Referências para ler depois
-
-

---
**Data de leitura:**
**Próxima revisão:**
```

**Exercício desta semana:**
Aplique o protocolo em 2 papers:

1. **McClelland, McNaughton & O'Reilly 1995** (Semana 17) — aplique Pass 2
   - Busque: "McClelland 1995 complementary learning systems PubMed"
   - Foco: as Figuras 1-3 explicam o modelo inteiro

2. **Um paper que você encontrou por conta própria** — aplique Pass 1
   - Busque no Google Scholar: `hippocampus neocortex artificial neural network 2020`
   - Escolha o mais relevante para o seu projeto

### ✅ Como saber se absorveu

- [ ] Você tem 2 anotações completas no template
- [ ] McClelland 1995 resumido em 2 frases por escrito
- [ ] Você identificou pelo menos 1 paper que não está nesta trilha mas é relevante para o seu projeto
- [ ] Você sabe o que é Pass 1 vs Pass 3 e quando usar cada um

---

## SEMANA 25
**Tema: Documentando Sua Pesquisa**

### 📚 O que estudar (30min)

Pesquisa não documentada é pesquisa perdida.

Não é burocracia — é auto-preservação. Você vai agradecer a si mesmo em 3 meses quando
esquecer por que tomou a decisão X.

**Mínimo viável de documentação para P&D:**

1. **README técnico** — o que é, como instalar, como rodar (outra pessoa consegue usar)
2. **Diário de pesquisa** — decisões de design e por quê; hipóteses testadas
3. **Benchmark log** — resultados por semana (você enxerga progressão)

**Estrutura do README técnico:**

```markdown
# [Nome do Projeto]

## O que é
Uma frase.

## Arquitetura
[Diagrama ASCII + descrição de cada módulo]

## Instalação
pip install numpy matplotlib

## Uso rápido
python integrated_system.py

## Módulos
| Arquivo | Biologia correspondente | Referência |
|---|---|---|
| hippocampus.py | EC→DG→CA3→CA1 | McClelland 1995 |
| spatial_pooler.py | L4 neocórtex | Hawkins 2016 |
| temporal_memory.py | L2/3 neocórtex | Hawkins 2016 |
| ... | ... | ... |

## Benchmarks atuais (dd/mm/aaaa)
| Teste | Score | Alvo biológico |
|---|---|---|
| Pattern Separation | 0.xx | 0.30 |
| ... | | |

## Limitações conhecidas
- [...]

## Próximos passos
- [ ] [...]

## Referências
- [...]
```

### ⚗️ Exercício Prático (1h30)

**Parte 1:** Escreva o README do seu projeto (estrutura acima).

**Parte 2:** Comece o diário de pesquisa. Template para cada entry:

```markdown
# Diário de Pesquisa — [Data]

## Sessão de hoje
Duração:
Objetivo:

## O que funcionou


## O que NÃO funcionou
(e hipótese de por quê)


## Próxima hipótese a testar


## Pergunta aberta que surgiu


## Benchmarks desta sessão
| Teste | Score |
|---|---|
| | |
```

### ✅ Como saber se absorveu

- [ ] README completo — outra pessoa consegue instalar e rodar seu projeto lendo apenas o README
- [ ] Primeiro entry do diário de pesquisa escrito (não precisa ser perfeito — só precisa existir)
- [ ] Benchmarks atuais documentados com data e versão do código
- [ ] Pelo menos 3 hipóteses de melhoria listadas nos próximos passos

---

## SEMANA 26
**Tema: Roadmap de Pesquisa — O Que Vem a Seguir**

### 📚 O que revisar (2h)

Esta semana é de **síntese e planejamento**. Não tem leitura nova.

Revise os checklists dos meses anteriores. Para cada item não marcado, decida:
- (A) Ignorar — com justificativa escrita
- (B) Colocar no roadmap

### ⚗️ Exercício Final (1h30)

```python
# roadmap.py — Semana 26
"""
Roadmap de Pesquisa — Próximos 6 Meses.
Substitua os exemplos pelas suas hipóteses reais.
"""

ROADMAP = {

    "hipoteses": [
        # ── Formato obrigatório ──────────────────────────────────────────────
        # Cada hipótese deve ser TESTÁVEL:
        #   "Se eu implementar X, o benchmark Y vai de Z1 para Z2"
        # ────────────────────────────────────────────────────────────────────
        {
            "hipotese":    "Substituir Hopfield simétrico por CA3 com STDP assimétrico "
                           "vai permitir aprender sequências episódicas (não só padrões estáticos)",
            "experimento": "Implementar CA3-STDP e testar recall de sequência A→B→C",
            "metrica":     "Recall de sequências com > 80% de acurácia",
            "prazo":       "3 semanas",
        },
        {
            "hipotese":    "Adicionar theta oscillations como clock discreto vai melhorar "
                           "a separação entre encoding e recall no hipocampo",
            "experimento": "ThetaClock que alterna modos: encoding (fase 0) / recall (fase 2)",
            "metrica":     "Pattern separation ratio cai para < 0.25 (atual: veja benchmark)",
            "prazo":       "4 semanas",
        },
        {
            "hipotese":    "Consolidação acelerada por schema: padrões consistentes com "
                           "representação cortical já existente consolidam em < 10 replays",
            "experimento": "Medir n_replays para consolidação de padrão similar vs. novo",
            "metrica":     "Padrão similar consolida em < 50% do tempo de padrão novo",
            "prazo":       "5 semanas",
        },
    ],

    "topicos_avancados": [
        # Estude DEPOIS desta trilha, quando os fundamentos estiverem sólidos
        "Grid cells e computação espacial (Moser, Moser & McNaughton 2008)",
        "Atenção via acetilcolina — gating de input (Hasselmo 2006)",
        "Working memory em córtex prefrontal (Goldman-Rakic 1995)",
        "Free Energy Principle — framework unificado (Friston 2010)",
        "Spike sorting e análise de registros reais (Numpy + MEArec)",
        "Neuromorphic hardware: Intel Loihi, IBM TrueNorth",
        "Transformer vs. córtex: semelhanças e diferenças estruturais",
    ],

    "limitacoes_atuais": [
        # Preencha com o que você descobriu nos benchmarks
        # Exemplo:
        "Sem representação temporal dentro do hipocampo (só padrões estáticos, não sequências)",
        "Ausência de atenção top-down modulando quais features são relevantes",
        "CA3 (Hopfield) com capacidade teórica limitada a ~0.14N — precisa de versão esparsa",
        "Nenhuma forma de meta-aprendizado: o sistema não sabe o que não sabe",
    ],

    "perguntas_abertas": [
        # Escreva as questões genuínas que surgiram durante o estudo
        # Exemplo:
        "Como representar relações abstratas (A>B, B>C → A>C) sem treino explícito?",
        "Quando e por que o sistema deve 'desaprender' algo?",
        "O neocórtex artificial precisa de hierarquia de áreas ou uma coluna é suficiente?",
    ],
}


def print_roadmap():
    print("═" * 62)
    print("  ROADMAP DE PESQUISA — PRÓXIMOS 6 MESES")
    print("═" * 62)

    print("\n📌 HIPÓTESES A TESTAR:")
    for i, h in enumerate(ROADMAP["hipoteses"], 1):
        print(f"\n  {i}. {h['hipotese']}")
        print(f"     Experimento : {h['experimento']}")
        print(f"     Métrica     : {h['metrica']}")
        print(f"     Prazo       : {h['prazo']}")

    print("\n📚 TÓPICOS AVANÇADOS (para depois dos fundamentos):")
    for t in ROADMAP["topicos_avancados"]:
        print(f"  • {t}")

    print("\n⚠️  LIMITAÇÕES ATUAIS DO SISTEMA:")
    for l in ROADMAP["limitacoes_atuais"]:
        print(f"  • {l}")

    print("\n❓ PERGUNTAS ABERTAS:")
    for q in ROADMAP["perguntas_abertas"]:
        print(f"  • {q}")

    print("\n" + "═" * 62)
    print("  PRÓXIMAS AÇÕES CONCRETAS (próximas 2 semanas):")
    if ROADMAP["hipoteses"]:
        print(f"  1. Implementar: {ROADMAP['hipoteses'][0]['experimento']}")
    print("  2. Ler (Pass 2): paper prioritário da lista de limitações")
    print("  3. Atualizar benchmarks com data")
    print("═" * 62)


if __name__ == '__main__':
    print_roadmap()
```

### ✅ Checklist Final — 6 Meses

Se você está aqui, você foi do zero em neurobiologia ao sistema integrado neocórtex-hipocampo em Python puro.

**O que você construiu:**
- [ ] Neurônio IF com parâmetros biofísicos (tau_m, V_thresh, V_reset)
- [ ] Sinapses com STDP — janela temporal assimétrica (Bi & Poo 1998)
- [ ] Circuito E/I — rede estável com equilíbrio excitação/inibição
- [ ] Circuito trisináptico EC→DG→CA3→CA1 completo
- [ ] Place cells e sistema de indexação hipocampal
- [ ] SDR como tipo de dado com capacidade e overlap
- [ ] SpatialPooler com permanências, k-WTA e boosting
- [ ] TemporalMemory com burst, predição e anomaly score
- [ ] Consolidação de memória com ciclos de sono (replay)
- [ ] Codificação preditiva hierárquica
- [ ] NeuromodulatorSystem (DA, ACh, NE) modulando LRs
- [ ] Sistema integrado com protocolo experience→sleep→recall
- [ ] Suite de benchmarks biológicos
- [ ] README técnico + Diário de pesquisa
- [ ] Roadmap de pesquisa com hipóteses testáveis

---

---

# APÊNDICE: Recursos Gratuitos Consolidados

---

## 📖 Livros Online (acesso livre e gratuito)

### 1. Theoretical Neuroscience — Dayan & Abbott
**O mais importante desta lista para o seu projeto.**
- Cobre: neurônio, sinapses, redes, aprendizado, representações
- PDF gratuito (MIT Press autoriza):
  Busque "Dayan Abbott Theoretical Neuroscience PDF site:gatsby.ucl.ac.uk"
  URL direta: `https://boulderschool.yale.edu/sites/default/files/files/DayanAbbott.pdf`
- Relevante para: Semanas 1-12 (Vol. I)

### 2. Neuroscience 4th ed. — Purves et al.
- Texto clássico, versão online gratuita no NCBI Bookshelf
- URL: `https://www.ncbi.nlm.nih.gov/books/NBK10799/`
- Relevante para: anatomia hipocampal, anatomia cortical

### 3. Principles of Neural Science — Kandel et al.
- O mais completo (mas pesado — use só o capítulo que precisa)
- Busque capítulos específicos no Google Scholar como "Kandel [assunto] chapter PDF"

---

## 🎓 Cursos Online Gratuitos

### 1. Neuromatch Academy — Computational Neuroscience
**O mais relevante para este projeto.** Python + Google Colab, 100% gratuito.
- URL: `https://compneuro.neuromatch.io`
- Relevante: W1D1 (neurônio), W1D4 (redes), W3D1 (hipocampo), W3D2 (córtex)

### 2. Coursera — Computational Neuroscience (Univ. of Washington)
- Instrutores: Rajesh Rao & Adrienne Fairhall
- Audit gratuito (sem certificado)
- Busque: "Computational Neuroscience Coursera Rao" → primeira opção
- Relevante para: Semanas 1-8

---

## 📄 Papers Essenciais (todos gratuitos)

| Semana | Autores (Ano) | Título curto | Como acessar |
|--------|--------------|--------------|--------------|
| V.I Sem.7 | Bi & Poo 1998 | "Synaptic modifications in cultured hippocampal neurons" | PubMed → PMC free |
| V.I Sem.11 | Hopfield 1982 | "Neural networks and physical systems..." | pnas.org open access |
| V.I Sem.12 | O'Keefe & Dostrovsky 1971 | "The hippocampus as a spatial map" | ResearchGate |
| Sem.14 | Ahmad & Hawkins 2015 | "Properties of Sparse Distributed Representations" | arxiv.org/abs/1503.07469 |
| Sem.15 | Hawkins et al. 2016 | "HTM Spatial Pooler" | numenta.com/research |
| Sem.16 | Hawkins & Ahmad 2016 | "Why Neurons Have Thousands of Synapses" | numenta.com/research |
| Sem.17 | McClelland et al. 1995 | "Why there are complementary learning systems" | PubMed → PMC free |
| Sem.18 | Wilson & McNaughton 1994 | "Reactivation of hippocampal memories during sleep" | ResearchGate |
| Sem.19 | Rao & Ballard 1999 | "Predictive coding in the visual cortex" | ResearchGate PDF |
| Sem.20 | Hasselmo 2006 | "The role of acetylcholine in learning and memory" | PubMed → PMC free |
| Sem.22 | Leutgeb et al. 2007 | "Pattern separation in the dentate gyrus and CA3" | PubMed → PMC free |

---

## 📺 YouTube — Canais e Séries

### HTM School (Numenta) — **canal mais importante para Semanas 13-22**
- Busque: "HTM School" no YouTube
- Episódios essenciais:
  - Ep. 1-2: SDR
  - Ep. 3-5: Spatial Pooler
  - Ep. 7-10: Temporal Memory
  - Ep. 12: Synaptic Permanence

### Neuromatch Academy (YouTube)
- Canal do YouTube com todas as aulas gravadas
- Busque: "Neuromatch Academy 2021 Computational Neuroscience"

### Two Minute Neuroscience
- Vídeos de 2-3min para introdução rápida de cada estrutura cerebral
- Ótimo para Semanas 1-4 do Vol. I

---

## 🔍 Como Encontrar Papers Gratuitos

Em ordem de preferência:

1. **PubMed Central** (`pubmed.ncbi.nlm.nih.gov`)
   — Muitos papers têm botão "PMC Free Full Text"

2. **arXiv** (`arxiv.org`)
   — Preprints de papers computacionais (neurociência computacional, HTM, etc.)

3. **ResearchGate** (`researchgate.net`)
   — Autores frequentemente postam PDFs dos próprios papers

4. **Semantic Scholar** (`semanticscholar.org`)
   — Agrega links para PDFs open access automaticamente

5. **Unpaywall** (extensão de browser gratuita)
   — Encontra automaticamente versões legalmente gratuitas enquanto você navega

6. **Google Scholar**
   — Busque `[título do paper] filetype:pdf` — muitas universidades postam versões abertas

---

## 🐍 Dependências Python

```bash
# Tudo que você precisa — nenhum framework de deep learning
pip install numpy matplotlib

# Para análise estatística dos benchmarks
pip install scipy pandas

# Para profiling (já vem com Python — sem instalação)
# import cProfile, pstats

# Para visualização de grafos neurais (opcional)
pip install networkx

# Para serialização dos pesos do sistema (salvar/carregar)
# import pickle   # já vem com Python
```

---

## 🗺️ Mapa do Conhecimento — O que esta trilha cobriu

```
FUNDAMENTOS (Vol. I Meses 1-2)
  Neurônio IF → Sinapses → STDP → Circuito E/I → SDR k-WTA

HIPOCAMPO (Vol. I Mês 3)
  EC → DG (separação) → CA3 Hopfield (completação) → CA1 (indexação) → Place Cells

NEOCÓRTEX (Vol. II Mês 4)
  CorticalColumn → SDR class → SpatialPooler → TemporalMemory

INTEGRAÇÃO (Vol. II Mês 5)
  CLS Theory → Sleep Consolidation → Predictive Coding → Neuromodulação → Sistema Integrado

PESQUISA (Vol. II Mês 6)
  Benchmarks → Otimização → Leitura de papers → Documentação → Roadmap
```

---

*Esta trilha foi desenhada para 4 horas/semana por 26 semanas.*
*A diferença entre simulação e ciência é o que você vai fazer a seguir:*
*testar hipóteses, documentar resultados e construir em cima do que aprendeu.*
