# PROJECT.md — Projeto POLIMATA

# P.O.L.I.M.A.T.A
*(Acrônimo em Definição ao Longo do Projeto)*

---

## 1. Visão e Missão

O **Projeto POLIMATA** é uma iniciativa de pesquisa independente e engenharia de primeiros princípios, conduzida por **Everton Kaylon**. 

### Objetivos Centrais:
1. **Ambiente de Pesquisa e Aprendizagem:** Construir e compreender, do zero absoluto, um modelo computacional biologicamente inspirado de Inteligência Artificial Neuromórfica.
2. **Portfólio Acadêmico de Alto Nível:** Estruturar uma base sólida de publicações, código auditável e comprovação de domínio teórico/prático para aplicação em programas de Mestrado/Pós-Graduação (com foco em iniciativas internacionais como BK21/GKS na Coreia do Sul).
3. **Acoplamento Físico Futuro:** Servir como base de controle neural para eventual aplicação em sistemas robóticos com atuadores/músculos artificiais (pesquisas de materiais macios / DEAs / ferrofluidos).

---

## 2. Entregáveis e Resultados Tangíveis

* **Sistema Computacional Integrado:** Arquitetura funcional contendo rede neural spiking (SNN) + dinâmica de memória hipocampal (replay/consolidação) + neocórtex preditivo (*Predictive Coding*), implementada em **Python Puro e NumPy**.
* **Publicações & Divulgação Científica:**
  * Preprint / Artigo técnico formal documentando a arquitetura e os resultados experimentais.
  * Registro e divulgação do processo de aprendizagem e pesquisa no **GitHub**, **Medium**, **YouTube** e **Instagram**.
* **Transparência e Autoria:** Garantir que cada linha de código, formulação matemática e teste seja implementada, depurada e explicada pelo autor, eliminando qualquer dependência de geração cega por IA.

---

## 3. Filosofia: Primeiros Princípios e Evolução Histórica

Todo o desenvolvimento segue rigorosamente a abordagem de **Primeiros Princípios** e contextualização histórica:
* **Matemática & Física:** Compreender a derivação, o significado geométrico e a física subjacente antes de escrever código.
* **Computação & Infraestrutura:** Dominar a transformação de estados, complexidade algorítmica e estruturas de dados elementares antes de utilizar qualquer biblioteca de alto nível.
* **Biologia vs. Modelo:** Manter distinção clara entre evidência neurobiológica real e abstração matemática de engenharia.

---

## 4. Restrições Operacionais e Perfil de Hardware

### Hardware do Ambiente de Desenvolvimento:
* **Processador:** Intel(R) Core(TM) i7-3770 CPU @ 3.40 GHz
* **Memória RAM:** 8,00 GB
* **Armazenamento:** SSD 120 GB (Netac)
* **Placa de Vídeo:** NVIDIA GeForce GT 730 (4 GB)
* **Diretriz de Engenharia decorrente:** O código deve ser extremamente leve, eficiente e determinístico. Foco em simulações numéricas diretas, evitando frameworks pesados de Deep Learning que demandem GPUs modernas.

### Recursos de IA e Software:
* Planos gratuitos e ferramentas locais. A IA atua exclusivamente como tutor e harness pedagógico, nunca como executora de código.

### Cronograma de Dedicação:
* **Segundas-feiras** (bloco dedicado)
* **Terças-feiras** (período noturno)
* **Quintas-feiras** (período noturno)
* **Sextas-feiras** (bloco dedicado)

---

## 5. Trilha Arquitetural do Sistema

```text
[Trilha A: Fundamentos]   Python Puro, Vetores, Matrizes, Cálculo Elementar
        ↓
[Trilha B: Redes Neurais] Neurônio Artificial, Perceptron, Limites do Linear, MLP, Backprop
        ↓
[Trilha C: Neurociência]  Biofísica de Membrana, Potencial de Ação, LIF, STDP, SNNs
        ↓
[Trilha D: Memória]       Hipocampo Computacional, Replay, Consolidação de Memória
        ↓
[Trilha E: Neocórtex]     Codificação Preditiva (Predictive Coding), Hierarquia Cortical
        ↓
[Trilha F: POLIMATA]      Sistema Fechado: SNN + Memória + Neocórtex (+ Interface Muscular)
```
