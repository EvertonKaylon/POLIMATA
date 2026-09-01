# 🧠 Trilha de Neurobiologia para IA Bioinspirada
## Volume II — Meses 4 a 6: Abertura e Mapa de Navegação

> **Este documento:** Leia antes de abrir qualquer arquivo do Vol. II.
> É o mapa que conecta onde você chegou no Vol. I e onde vai chegar ao final do Vol. II.

---

## O que você trouxe do Volume I

| ✓ | Módulo | Arquivo | Biologia |
|---|--------|---------|---------|
| ✓ | Neurônio Integrate-and-Fire | `neuron.py` | Neurônio cortical |
| ✓ | Sinapses com STDP | `stdp.py` | Plasticidade Bi & Poo 1998 |
| ✓ | Circuito E/I | `ei_circuit.py` | Equilíbrio excitatório/inibitório |
| ✓ | k-WTA (esparsidade) | `kwta.py` | Inibição lateral no DG |
| ✓ | Dentate Gyrus | `dg.py` | Separação de padrões |
| ✓ | CA3 — Hopfield | `ca3.py` | Completação de padrões |
| ✓ | CA1 | `ca1.py` | Comparador / indexador |
| ✓ | Place Cells | `place_cells.py` | Representação espacial |
| ✓ | HippocampalIndexSystem | `hippocampus.py` | Sistema completo |

---

## O que você vai construir no Volume II

| # | Semana | Módulo novo | Arquivo sugerido |
|---|--------|------------|-----------------|
| 13 | Mês 4 | CorticalColumn (4 camadas) | `cortical_column.py` |
| 14 | Mês 4 | SDR class (tipo de dado formal) | `sdr.py` |
| 15 | Mês 4 | SpatialPooler (L4) | `spatial_pooler.py` |
| 16 | Mês 4 | TemporalMemory (L2/3) | `temporal_memory.py` |
| 17 | Mês 5 | Demonstração CLS | `cls_demo.py` |
| 18 | Mês 5 | MemoryConsolidationSystem | `sleep_consolidation.py` |
| 19 | Mês 5 | HierarchicalPredictiveCortex | `predictive_coding.py` |
| 20 | Mês 5 | NeuromodulatorSystem | `neuromodulation.py` |
| 21 | Mês 5 | NeocorticalHippocampalSystem | `integrated_system.py` |
| 22 | Mês 6 | NeuroBenchmarkSuite | `benchmark_suite.py` |
| 23 | Mês 6 | Vetorização + profiling | `optimization.py` |
| 24 | Mês 6 | Template de anotação de papers | `paper_notes/` |
| 25 | Mês 6 | README + Diário de pesquisa | `README.md`, `diary.md` |
| 26 | Mês 6 | Roadmap de pesquisa | `roadmap.py` |

---

## Arquitetura Final do Sistema

```
┌─────────────────────────────────────────────────────┐
│            NeocorticalHippocampalSystem             │
│                                                     │
│   ┌──────────────────┐   ┌──────────────────────┐   │
│   │    HIPOCAMPO     │   │     NEOCÓRTEX        │   │
│   │  (Vol. I)        │   │  (Vol. II)           │   │
│   │                  │   │                      │   │
│   │  EC → DG         │   │  Input               │   │
│   │     → CA3        │   │   → SpatialPooler    │   │
│   │     → CA1        │   │   → TemporalMemory   │   │
│   │                  │   │   → CorticalColumn   │   │
│   │  One-shot learn  │   │  Slow learning       │   │
│   │  Episodic memory │   │  Semantic memory     │   │
│   └────────┬─────────┘   └──────────┬───────────┘   │
│            │                        │               │
│            └──────────┬─────────────┘               │
│                       │                             │
│            ┌──────────▼──────────┐                  │
│            │ NeuromodulatorSystem│                  │
│            │  DA  ACh  NE  5HT   │                  │
│            └─────────────────────┘                  │
│                                                     │
│   Protocolo:                                        │
│   experience() → sleep() → recall()                 │
└─────────────────────────────────────────────────────┘
```

---

## Os 5 Gates do Volume II

| Gate | Semana | Critério de aprovação |
|------|--------|-----------------------|
| G-IV-4  | Sem. 16 | SP estável (stab>0.90) + TM aprende sequência (anomaly<0.30) |
| G-V-5   | Sem. 21 | Sistema integrado: experience→sleep→recall funcionando |
| G-VI-5  | Sem. 22 | Suite de benchmarks: pelo menos 3/5 aprovados |
| G-VI-6  | Sem. 23 | Sistema processa 1000 steps em < 30s após otimização |
| G-VI-7  | Sem. 26 | README + diário + roadmap com 3 hipóteses testáveis |

---

## Como usar os documentos do Volume II

| Documento | Quando abrir |
|-----------|-------------|
| `trilha_vol2_abertura.md` (este) | Agora e no início de cada mês |
| `trilha_vol2_mes4_neocortex.md` | Semanas 13-16 |
| `trilha_vol2_mes5_integracao.md` | Após passar G-IV-4 |
| `trilha_vol2_mes6_pesquisa.md` | Após passar G-V-5 (contém também o Apêndice de Recursos) |

---

## Ritual das 4 Horas Semanais (mantido do Vol. I)

```
Bloco 1 (1h30)  → Recurso da semana: vídeo / paper / capítulo
Bloco 2 (2h00)  → Implementação do exercício Python
Bloco 3 (0h30)  → Responder por escrito as 3 perguntas de verificação
                   + uma frase: "Isso mudou meu projeto como?"
```

**Antes de cada semana:**
> *"Como o que aprendi semana passada aparece no meu código hoje?"*

**Ao final do Volume II:**
> *"O que meu sistema ainda não consegue fazer que o biológico consegue?"*
> (Esta é a pergunta que vai guiar os próximos 6 meses de pesquisa.)
