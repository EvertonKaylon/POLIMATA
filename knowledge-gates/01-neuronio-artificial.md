# KNOWLEDGE GATE 01 — O Neurônio Artificial e Limites da Linearidade

> **Domínio:** Trilha B (Fundamentos de Redes Neurais)  
> **Status:** PENDENTE  
> **Alvo para Avanço:** Nível 4 (Fundamento) / Nível 5 (Implementação)

---

## 1. O Que É Este Gate?

Este documento estabelece os critérios objetivos que você deve cumprir para comprovar domínio sobre o primeiro tijolo computacional: o modelo de neurônio artificial com função degrau (McCulloch-Pitts / Rosenblatt) e suas propriedades geométricas.

---

## 2. Critérios por Nível de Domínio

### NÍVEL 1 — RECONHECE
- [ ] Identifica e nomeia os componentes: entradas ($x_i$), pesos ($w_i$), bias ($b$), soma ponderada ($z$), função de ativação ($\phi$), saída ($y$).

### NÍVEL 2 — EXPLICA
- [ ] Explica com as próprias palavras por que multiplicamos entradas por pesos.
- [ ] Explica qual é o papel biológico e matemático do bias ($b$) ou limiar ($\theta$).
- [ ] Explica a diferença entre soma linear e ativação não linear.

### NÍVEL 3 — APLICA
- [ ] Calcula manualmente a saída de um neurônio dadas entradas, pesos e bias arbitrários.
- [ ] Configura pesos e bias na mão para implementar as portas lógicas **AND** e **OR**.

### NÍVEL 4 — TRANSFERE (Alvo Mínimo para Teoria)
- [ ] Prevê geometricamente onde a reta de decisão corta os eixos $x_1$ e $x_2$ dada a equação $w_1 x_1 + w_2 x_2 + b = 0$.
- [ ] Demonstra e explica formalmente por que um único neurônio linear é **incapaz** de resolver a função **XOR**.

### NÍVEL 5 — RECONSTRÓI (Alvo para Implementação)
- [ ] Implementa em Python puro (sem bibliotecas externas) uma classe ou função de neurônio artificial que realize o forward pass e testes de portas booleanas.
- [ ] Escreve testes que comprovem o funcionamento correto de sua implementação.

### NÍVEL 6 — CRITICA (Visão Científica)
- [ ] Explica as diferenças fundamentais entre a abstração computacional (McCulloch-Pitts / Rosenblatt) e o neurônio biológico real (potencial contínuo, dinâmica temporal, canais iônicos, spikes).

---

## 3. Registro de Aprovação

| Data | Nível Atingido | Evidência (Teste / Explicação / Código) | Avaliador |
|------|----------------|-----------------------------------------|-----------|
| — | Nível 0 (Inicial) | Ponto de partida declarado | Harness |
