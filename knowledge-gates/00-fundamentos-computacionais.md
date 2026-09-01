# KNOWLEDGE GATE 00 — Fundamentos Computacionais e Álgebra Vetorial em Python

> **Domínio:** Trilha A (Fundamentos Computacionais)  
> **Status:** ATIVO  
> **Alvo para Avanço:** Nível 4 (Fundamento) / Nível 5 (Implementação em Código)

---

## 1. Objetivo

Dominar as estruturas de dados fundamentais do Python e as operações algébricas elementares sobre vetores e matrizes usando **apenas Python puro** (sem importar NumPy ou qualquer biblioteca externa).

---

## 2. Critérios por Nível de Domínio

### NÍVEL 1 — RECONHECE
- [ ] Identifica e diferencia tipos primitivos (`int`, `float`, `bool`, `str`) e coleções (`list`, `tuple`, `dict`).
- [ ] Reconhece a representação de um vetor matemático como uma lista homogênea de números em Python.

### NÍVEL 2 — EXPLICA
- [ ] Explica a diferença entre iterar por índice (`range(len(...))`) e iterar diretamente sobre elementos ou pares (`zip`, `enumerate`).
- [ ] Explica o que é dimensionalidade de um vetor e por que operações vetoriais exigem compatibilidade dimensional.
- [ ] Explica geometricamente e algebricamente o que é o **Produto Escalar (Dot Product)**.

### NÍVEL 3 — APLICA
- [ ] Escreve funções para soma de dois vetores elemento a elemento.
- [ ] Escreve funções para multiplicação de vetor por escalar.
- [ ] Escreve a função de produto escalar entre dois vetores usando loops manuais.

### NÍVEL 4 — TRANSFERE (Alvo Mínimo Teórico)
- [ ] Prevê o resultado algébrico e geométrico de produtos escalares com vetores ortogonais, paralelos e opostos.
- [ ] Consegue reescrever operações vetoriais utilizando *list comprehensions* e explicar as diferenças de legibilidade e performance.

### NÍVEL 5 — RECONSTRÓI (Alvo de Engenharia)
- [ ] Cria um módulo `src/fundamentos.py` contendo funções vetoriais testadas e documentadas com *type hints*.
- [ ] Cria suíte de testes unitários em `tests/test_fundamentos.py` validando todas as operações e tratamento de erros (ex: vetores de tamanhos diferentes).

---

## 3. Registro de Aprovação

| Data | Nível Atingido | Evidência (Teste / Explicação / Código) | Avaliador |
|------|----------------|-----------------------------------------|-----------|
| 2026-09-01 | Nível 0 (Inicial) | Ponto de partida declarado | Harness |
