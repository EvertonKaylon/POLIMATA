# PROGRESS.md — Projeto POLIMATA

# Estado do Projeto

> **Última atualização:** 2026-09-01

---

## Status Atual

```text
FASE: TRILHA A — FUNDAMENTOS COMPUTACIONAIS
ETAPA: Subetapa A1 (Diagnóstico Prático de Python e Estruturas Básicas)
PRÓXIMO: Avaliação das respostas do usuário para os 3 desafios da Subetapa A1
```

---

## O Que Foi Feito

### 2026-08-31 / 2026-09-01 — Fundação do Harness & Infraestrutura

- [x] Auditoria completa do repositório e isolamento do material anterior.
- [x] Decisão: todo código prévio é vibecoding → movido para `archive/`.
- [x] Nome do projeto: **POLIMATA** (*P.O.L.I.M.A.T.A* — Acrônimo em definição).
- [x] Git inicializado localmente e `.gitignore` configurado.
- [x] Estrutura de diretórios limpa criada (`src/`, `tests/`, `experiments/`, `plans/`, `docs/`, `knowledge-gates/`, `archive/`).
- [x] [`AGENTS.md`](AGENTS.md) refinado como constituição enxuta.
- [x] [`.agents/skills/anti-vibecoding/SKILL.md`](.agents/skills/anti-vibecoding/SKILL.md) criado com regras estritas.
- [x] [`LEARNING.md`](LEARNING.md) criado (ponto de partida nivelado em 0).
- [x] [`PROJECT.md`](PROJECT.md) formalizado com a visão do autor, hardware e cronograma.
- [x] [`README.md`](README.md) criado com apresentação trilíngue (Inglês, Português, Coreano).
- [x] Repositório remoto configurado e sincronizado: [GitHub POLIMATA](https://github.com/EvertonKaylon/POLIMATA).
- [x] Regra de Invariante de Sequência Estrita e Perfil de Hardware persistida e commitada.
- [x] [`knowledge-gates/00-fundamentos-computacionais.md`](knowledge-gates/00-fundamentos-computacionais.md) criado (Trilha A).
- [x] [`plans/active/trilha-a-fundamentos.md`](plans/active/trilha-a-fundamentos.md) criado como plano ativo.

---

## Foco Ativo: Trilha A (Fundamentos Computacionais)

- [ ] **Knowledge Gate 00:** Fundamentos Computacionais e Álgebra Vetorial em Python Puro
  - [ ] **A1. Diagnóstico de sintaxe e manipulação de listas** *(Em andamento)*
  - [ ] A2. Operações de soma vetorial e multiplicação por escalar
  - [ ] A3. Implementação e compreensão do Produto Escalar (Dot Product)
  - [ ] A4. Criação do módulo `src/fundamentos.py` e testes em `tests/test_fundamentos.py`
- [ ] **Próximo:** Knowledge Gate 01 (Trilha B: Neurônio Artificial e Limites da Linearidade)

---

## Decisões Tomadas

| # | Decisão | Data | Contexto |
|---|---------|------|----------|
| 1 | Todo código existente é vibecoding — arquivar, não usar | 2026-08-31 | Declaração de transparência do autor |
| 2 | AGENTS.md = constituição mínima, regras detalhadas no skill | 2026-08-31 | Opção B da auditoria |
| 3 | Código vibecoding → `archive/scripts/` com nota | 2026-08-31 | Isolamento de artefatos não autorais |
| 4 | Capítulos teóricos → `archive/research/` | 2026-08-31 | Avaliação crítica em `archive/README.md` |
| 5 | Nome do projeto: POLIMATA (P.O.L.I.M.A.T.A) | 2026-08-31 | Acrônimo a definir ao longo do projeto |
| 6 | Invariante de Sequência Estrita (Trilha A $\to$ B $\to$ C...) | 2026-09-01 | Correção do usuário sobre iniciar pela Trilha A |
| 7 | Repositório público no GitHub com README trilíngue (EN/PT/KO) | 2026-09-01 | Foco em portfólio acadêmico internacional |

---

## Riscos Ativos & Mitigações

| Risco | Severidade | Mitigação |
|-------|-----------|-----------|
| Amnésia entre sessões | ALTO | Persistência obrigatória em `LEARNING.md`, `PROGRESS.md` e commits Git |
| Ilusão de competência | ALTO | Todo código prévio arquivado; progressão bloqueada sem testes e explicações |
| Sobrecarga do hardware | MÉDIO | Desenvolvimento restrito a Python puro / NumPy leve, sem GPUs pesadas |

---

## Open Questions

- Qual será o significado definitivo das letras do acrônimo P.O.L.I.M.A.T.A?
- Qual é o nível de partida prático do autor na sintaxe de Python? *(sendo avaliado na Subetapa A1)*
