# Git Workflow para Pesquisa
## Como documentar o progresso sem perder rastreabilidade científica

---

## A diferença fundamental

Num repositório de software, commits descrevem **o que o código faz**.
Num repositório de pesquisa, commits descrevem **o que você aprendeu**.

```
Software:  "Add dropout layer to model"
Pesquisa:  "[EXP-001] LIF neuron — I_crítico empírico 18 pA vs 20 pA teórico"
```

O commit de pesquisa tem: o experimento, o resultado, a divergência.
Alguém lendo o `git log` deve entender a história intelectual do projeto.

---

## Formato de commit para pesquisa

```
[MODULO] Título de uma linha (máx 72 chars)

Hipótese testada: [o que você esperava]
Resultado:        [o que aconteceu]
Divergência:      [se diferiu, por quê]
Próximo passo:    [uma ação concreta]

Ref: paper ou equação relevante
```

**Exemplos reais para este projeto:**

```bash
# Após implementar LIF
git commit -m "[LIF] Neurônio LIF com integração de Euler — taxa de disparo correta

Hipótese: I=25pA → ~15 Hz. Resultado: 18 Hz.
Divergência: 20% acima — tau_m real pode ser >20ms em corticais.
Próximo: Spike trains Poisson (Semana 2)

Ref: Gerstner Ch.1.3, eq. 1.5"

# Após DG funcionar
git commit -m "[DG] k-WTA com k=2% separa padrões 90% similares para <25% overlap

Hipótese: overlap_saída/overlap_entrada < 0.33 (Leutgeb 2007).
Resultado: ratio = 0.21 — abaixo do alvo (melhor que esperado).
Divergência: nenhuma. Sistema biologicamente plausível.
Próximo: CA3 Hopfield (Semana 9)"

# Quando algo NÃO funciona
git commit -m "[CA3] Hopfield entra em estado espúrio com N>25 padrões (esperado)

Hipótese: capacidade = 0.14*200 = 28 padrões.
Resultado: colapso em N=22 (abaixo do teórico).
Divergência: pesos não normalizados aumentam interferência.
Próximo: normalizar W após cada store() e retestar"
```

---

## Branches: quando usar

```
main          → código estável, documentado, passa todos os testes
               (commits só quando o experimento termina com resultado)

exp/lif       → exploração do LIF, commits frequentes e messy
exp/dg        → exploração do DG
exp/ca3       → exploração do CA3
```

**Workflow por experimento:**

```bash
# Começar experimento
git checkout -b exp/lif-neuron

# Durante a exploração: commits frequentes, mensagens simples
git commit -m "WIP: testando tau_m=20ms vs 10ms"
git commit -m "WIP: reset brando não converge"
git commit -m "WIP: reset duro funciona melhor"

# Quando terminar e tiver resultado:
git checkout main
git merge --squash exp/lif-neuron   # une todos os WIPs em um commit limpo
git commit -m "[LIF] Neurônio LIF — resultado final + análise"

# Deletar branch de exploração
git branch -d exp/lif-neuron
```

O `--squash` é a chave: a exploração fica na branch, o `main` só vê o resultado final.

---

## Tags como marcos de pesquisa

```bash
# Após passar cada Gate da trilha:
git tag -a "gate-1-hipocampo" -m "Gate 1: Hipocampo completo DG+CA3+CA1
Pattern separation: ratio=0.21 (alvo <0.33) ✓
One-shot learning: recall 78% com 30% ruído ✓
Data: 2025-08-15"

git tag -a "gate-2-neocortex" -m "Gate 2: Neocórtex com SP+TM
SP stability: 0.94 (alvo >0.90) ✓
TM anomaly learned: 0.18 (alvo <0.30) ✓"

git tag -a "gate-3-integracao" -m "Gate 3: Sistema integrado CLS
Catastrophic forgetting mitigado ✓"

# Ver todos os marcos
git tag -l
```

---

## O que commitar vs o que NÃO commitar

**✓ Commitar:**
```
src/hippocampus/lif_neuron.py          → código fonte
experiments/exp_001/hypothesis.md      → escrito ANTES de rodar
experiments/exp_001/run.py             → o experimento
experiments/exp_001/results.md         → resultados em texto
RESEARCH_LOG.md                        → diário atualizado
tests/test_equivalence.py              → testes
docs/*.md                              → documentação
pure_python/*.py                       → versão pura
```

**✗ NÃO commitar (já no .gitignore):**
```
experiments/*/figures/*.png            → plots binários grandes
*.npy / *.pkl                          → arrays salvos
hardware_profile.json                  → específico de cada máquina
__pycache__/                           → bytecode
```

**Para salvar plots importantes no GitHub:**
```bash
# Converter para SVG (texto, versionável) em vez de PNG
# No matplotlib:
plt.savefig('experiments/exp_001/figures/fi_curve.svg')
# SVG é texto, pode ser commitado — PNG não
```

---

## Primeira semana de commits — sequência real

```bash
# Dia 1: setup
bash setup_research_repo.sh neocortex-hippocampus-py
cd neocortex-hippocampus-py
# → já fez o primeiro commit automaticamente

# Dia 2: implementar LIF
# Abrir hypothesis.md, confirmar que concorda com a previsão
nvim experiments/exp_001_lif_neuron/hypothesis.md

# Criar a implementação
nvim src/hippocampus/lif_neuron.py

# Rodar experimento
python experiments/exp_001_lif_neuron/run.py

# Preencher results.md com a análise manual
nvim experiments/exp_001_lif_neuron/results.md

# Atualizar o diário
nvim RESEARCH_LOG.md

# Commit do experimento completo
git add src/hippocampus/lif_neuron.py
git add experiments/exp_001_lif_neuron/
git add RESEARCH_LOG.md
git commit -m "[EXP-001] LIF Neuron — I_crítico e curva f-I

Hipótese: I_crítico = 20 pA.
Resultado: [preencher com o número real]
Divergência: [preencher]
Próximo: Spike trains Poisson (Semana 2)"

# Publicar
git push origin main
```

---

## Como documentar a migração NumPy → Python puro

Esta é a parte mais valiosa do repositório para outros pesquisadores.

**Estrutura de cada módulo migrado:**

```python
# pure_python/hippocampus/lif_neuron_pure.py

"""
LIF Neuron — Implementação Python Puro
=======================================
Equivalente a: src/hippocampus/lif_neuron.py (versão NumPy)
Verificada em: tests/test_equivalence.py

Por que Python puro:
  Cada linha corresponde a uma equação do paper.
  Nenhuma operação é "mágica" ou oculta por uma biblioteca.
  Serve como documentação matemática do modelo.
"""

class LIFNeuron_Pure:
    def step(self, I_ext, dt=0.001):
        # NumPy equiv: dU = (-(U - U_rest) + R_m * I) * dt / tau_m
        # → escalar puro, sem diferença aqui
        dU = (-(self.U - self.U_rest) + self.R_m * I_ext) * (dt / self.tau_m)
        self.U += dU
        ...
```

**Commit da migração:**

```bash
git commit -m "[PURE] LIF Neuron migrado para Python puro

Equivalência verificada: assert |numpy_out - pure_out| < 1e-6 ✓
Linha a linha correspondente às equações do paper.
Ver pure_python/hippocampus/lif_neuron_pure.py"
```

---

## README do GitHub: o que escrever no início

Não espere ter tudo pronto para fazer o repositório público.
A pesquisa em progresso é honesta e valiosa.

**Template para o topo do README:**

```markdown
> ⚠️ Pesquisa ativa em progresso.
> Estado atual: implementando Semana 3 — Sinapses com STDP.
> Última atualização: [data do último commit]
```

Repositórios de pesquisa em progresso recebem mais stars do que
repositórios "terminados" que ficaram parados por meses.
A linha de progresso é o produto.

---

## Comandos úteis no Archcraft + Zsh

```bash
# Ver história da pesquisa de forma legível
git log --oneline --graph

# Ver o que mudou em um experimento
git diff HEAD~1 experiments/exp_001/results.md

# Ver todos os experimentos e seus estados
ls experiments/*/results.md | xargs grep -l "Resultado:" 2>/dev/null

# Alias úteis para zsh (~/.zshrc):
alias glog='git log --oneline --graph --all'
alias gst='git status'
alias gcm='git commit -m'
alias gaa='git add -A'

# Commitar tudo com uma mensagem (útil para WIP):
alias gwip='git add -A && git commit -m "WIP: $(date +%H:%M)"'
```
