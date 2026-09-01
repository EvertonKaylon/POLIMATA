# 💰 Análise de Custos da IA — Comparação com Imóveis na Zona Leste de SP

> Documento complementar: análise financeira detalhada para treinar IA, viabilidade no VivoBook 15, e comparação com o mercado imobiliário de Rodolfo Pirani, São Mateus

---

## 1. Quanto Custa Treinar IA — Dados Detalhados

### 1.1 Custos Reais de Modelos Famosos

| Modelo | Ano | Parâmetros | Custo de Treino (USD) | Custo (BRL*) | Duração |
|---|---|---|---|---|---|
| **GPT-2** | 2019 | 1.5B | ~$50K | ~R$ 250K | Semanas |
| **GPT-3** | 2020 | 175B | ~$4.6M | ~R$ 23M | Meses |
| **LLaMA 1** | 2023 | 65B | ~$2.4M | ~R$ 12M | ~21 dias (2048 A100s) |
| **GPT-4** | 2023 | ~1.7T (estimado) | ~$78-100M | ~R$ 400-500M | Meses |
| **Gemini Ultra** | 2024 | Não revelado | ~$100M+ | ~R$ 500M+ | Meses |
| **GPT-5** (estimado) | 2025 | Não revelado | ~$200M-1B | ~R$ 1-5B | Meses |

*Câmbio estimado: 1 USD = 5 BRL

### 1.2 Decomposição dos Custos

O custo de treinar um LLM não é só GPU. Aqui está a decomposição:

| Componente | % do Custo Total | Detalhes |
|---|---|---|
| **Computação (GPUs)** | 40-60% | Milhares de GPUs por meses |
| **Dados (coleta e limpeza)** | 15-25% | Equipes de anotadores, licenças de dados |
| **Engenharia** | 15-20% | Salários de PhDs em ML ($200K-500K/ano cada) |
| **Infraestrutura** | 5-10% | Eletricidade, refrigeração, rede |
| **Falhas e reruns** | 5-15% | Treinos que falham a 80% do progresso = dinheiro perdido |

### 1.3 Custo de GPUs (Aluguel em Cloud)

| GPU | VRAM | Preço/hora (USD) | Preço/hora (BRL) | Para que serve |
|---|---|---|---|---|
| NVIDIA T4 | 16 GB | $0.50 | ~R$ 2.50 | Fine-tuning leve, inferência |
| NVIDIA A10G | 24 GB | $1.00 | ~R$ 5.00 | Fine-tuning moderado |
| NVIDIA RTX 4090 | 24 GB | $0.40-0.70 | ~R$ 2-3.50 | Fine-tuning local/cloud |
| NVIDIA A100 | 40/80 GB | $1.10-3.00 | ~R$ 5.50-15 | Treino sério |
| NVIDIA H100 | 80 GB | $2.00-4.00 | ~R$ 10-20 | Treino profissional |
| NVIDIA H200 | 141 GB | $4.00-8.00 | ~R$ 20-40 | Treino de fronteira |

### 1.4 Custo de Hardware (Compra no Brasil)

| Hardware | Preço Brasil (2025/2026) | VRAM | Viabilidade para IA |
|---|---|---|---|
| RTX 3060 12GB | R$ 2.000-3.000 | 12 GB | Fine-tuning básico (QLoRA 7B) |
| RTX 3090 24GB | R$ 8.000-12.000 | 24 GB | Fine-tuning sério (QLoRA até 27B) |
| RTX 4070 Ti 12GB | R$ 4.500-6.000 | 12 GB | Fine-tuning moderado |
| RTX 4090 24GB | R$ 22.000-30.000 | 24 GB | O "padrão ouro" para desenvolvedores |
| RTX 5090 32GB | R$ 15.000-20.000 | 32 GB | Nova geração, excelente para IA |
| Desktop completo (i7 + RTX 3060 + 32GB RAM) | R$ 6.000-9.000 | — | Estação de trabalho inicial |
| Desktop completo (i9 + RTX 4090 + 64GB RAM) | R$ 35.000-50.000 | — | Estação profissional |

---

## 2. Imóveis em Rodolfo Pirani, São Mateus — Dados Detalhados

### 2.1 Perfil da Região

- **Localização:** Jardim Rodolfo Pirani, subdistrito de São Mateus, Zona Leste de São Paulo
- **Acesso:** Próximo ao monotrilho da linha 15-Prata (em expansão)
- **Perfil:** Bairro predominantemente residencial/popular com presença de conjuntos COHAB
- **Valor médio do m²:** R$ 6.900 a R$ 7.000

### 2.2 Faixas de Preço por Tipo

| Tipo | Metragem Típica | Faixa de Preço | Preço Médio |
|---|---|---|---|
| **Apartamento COHAB (2 dorm)** | ~48-52 m² | R$ 120.000 — R$ 150.000 | **R$ 135.000** |
| **Apartamento padrão (2 dorm)** | ~45-60 m² | R$ 180.000 — R$ 280.000 | **R$ 230.000** |
| **Apartamento novo/lançamento** | ~40-55 m² | R$ 250.000 — R$ 350.000 | **R$ 300.000** |
| **Sobrado** | ~80-120 m² | R$ 290.000 — R$ 450.000 | **R$ 370.000** |
| **Casa** | ~100-200 m² | R$ 350.000 — R$ 600.000 | **R$ 475.000** |

### 2.3 Contexto do Financiamento

Para referência de viabilidade financeira:
- Financiamento MCMV (Minha Casa Minha Vida): até R$ 350K, parcelas de R$ 800-2.000/mês
- Renda necessária: ~3x o valor da parcela
- Entrada típica: 10-20% do valor

---

## 3. A Grande Comparação: IA vs Imóvel

### 3.1 Tabela de Equivalências

Usando como referência um **apartamento COHAB em Rodolfo Pirani (R$ 135.000)**:

| O que você treinaria | Custo (BRL) | Equivalência em Apartamentos |
|---|---|---|
| Rede neural XOR (NumPy) | R$ 0 | 0 |
| Fine-tune Gemma 2B (Colab grátis) | R$ 0 | 0 |
| Fine-tune Gemma 7B (Colab Pro, 1 mês) | R$ 55 | 0.0004 apto (4 dez-milésimos!) |
| Fine-tune Llama 70B (RunPod, 8h) | R$ 300 | 0.002 apto |
| Comprar RTX 4090 | R$ 26.000 | 0.19 apto (19% de um apto) |
| Desktop completo para IA | R$ 40.000 | 0.30 apto (30%) |
| Treinar modelo 1B do zero | R$ 50.000-375.000 | **0.4 a 2.8 aptos** |
| Treinar modelo 7B do zero | R$ 1.250.000 | **9.3 apartamentos** |
| Treinar GPT-3 (175B) | R$ 23.000.000 | **170 apartamentos** |
| Treinar GPT-4 (~1.7T) | R$ 500.000.000 | **3.700 apartamentos** |
| Treinar GPT-5 (estimado) | R$ 2.500.000.000 | **18.500 apartamentos** |

### 3.2 Visualização em Escala

```
Fine-tune Colab:     R$ 0           [*] (grátis)
Fine-tune RunPod:    R$ 300         [**]
RTX 4090:            R$ 26K         [****]
Desktop IA:          R$ 40K         [******]
1 Apto COHAB:        R$ 135K        [********************]
Treino modelo 7B:    R$ 1.25M       [=====================================================...]
Treino GPT-3:        R$ 23M         (sai da tela --- 170 apartamentos)
Treino GPT-4:        R$ 500M        (sai do bairro inteiro --- 3.700 aptos)
```

### 3.3 O que você compra com o preço de 1 apartamento COHAB (R$ 135K)

Se em vez de comprar um apartamento em Rodolfo Pirani, você investisse R$ 135K em IA:

| Item | Custo | O que ganha |
|---|---|---|
| Desktop top (i9 + RTX 4090 + 64GB) | R$ 45.000 | Estação de trabalho profissional para IA |
| 2 anos de Colab Pro+ | R$ 6.000 | Acesso a A100 no Google |
| 2 anos de RunPod (uso moderado) | R$ 12.000 | ~3.000 horas de GPU A100 |
| Curso de ML/Deep Learning (online) | R$ 2.000 | Formação profissional |
| Dataset profissional licenciado | R$ 5.000 | Dados de qualidade |
| **Total** | **R$ 70.000** | Estação + cloud + formação |
| **Sobra** | **R$ 65.000** | Para viver enquanto estuda |

### 3.4 Análise de Retorno sobre Investimento (ROI)

| Investimento | Custo | Retorno Potencial | Prazo |
|---|---|---|---|
| **Apartamento em São Mateus** | R$ 135K | Valorização ~5-8%/ano, aluguel ~R$ 800/mês | Longo prazo (10+ anos) |
| **Formação em IA + equipamento** | R$ 70K | Salário júnior ML: R$ 5-10K/mês; Sênior: R$ 15-30K/mês | Médio prazo (1-3 anos) |
| **Fine-tuning + serviço de IA** | R$ 0-5K | Venda de serviços personalizados de IA | Curto prazo (6-12 meses) |

---

## 4. Diagnóstico Completo do VivoBook 15

### 4.1 Análise Componente por Componente

#### CPU: Intel Core i3-1115G4

```
Cores:          2 físicos / 4 threads
Clock:          3.0 GHz (base) / 4.1 GHz (boost)
Cache:          6 MB L3
TDP:            28W
Arquitetura:    Tiger Lake (11ª geração)
Fabricação:     10nm SuperFin

Veredicto para IA:
- Adequado para: scripts Python, NumPy, scikit-learn, inferência de modelos tiny
- Inadequado para: treino de deep learning, qualquer coisa com PyTorch/TensorFlow pesado
- Ponto positivo: tem instruções AVX-512 (ajuda o llama.cpp)
- Ponto negativo: apenas 2 cores limita paralelismo
```

#### RAM: 4 GB DDR4

```
Capacidade:     4 GB (3.70 GB utilizável)
Tipo:           DDR4 (provavelmente 3200 MHz)
Slots:          1 ocupado (verificar se tem 2° slot para upgrade)

Uso típico do Windows 11:
- Sistema:      ~2.0-2.5 GB
- Antivírus:    ~0.3 GB
- Explorer:     ~0.2 GB
- Disponível:   ~1.0-1.2 GB para aplicativos

Veredicto para IA:
- Com 1 GB livre, pode rodar modelos quantizados de no máximo ~0.5-1.5B parâmetros
- Fine-tuning local: IMPOSSÍVEL (precisa mínimo 8-16 GB)
- UPGRADE PARA 16 GB É A PRIORIDADE #1 (~R$ 200-350)
```

#### GPU: Intel UHD Graphics (Gen 12)

```
Tipo:           Integrada (compartilha RAM do sistema)
VRAM dedicada:  0 MB (128 MB é da RAM do sistema)
Compute Units:  48 EU (Execution Units)
CUDA cores:     0 (não é NVIDIA)
OpenCL:         Suportado
Vulkan:         Suportado

Veredicto para IA:
- CUDA: NÃO SUPORTADO (PyTorch/TensorFlow requerem NVIDIA)
- OpenVINO: Possível inferência otimizada para Intel (opção alternativa)
- Treino: IMPOSSÍVEL via GPU
- Toda computação de IA será via CPU
```

#### SSD: 238 GB

```
Modelo:         SM2P32A8-256GC1 (Silicon Motion controller)
Interface:      NVMe PCIe (provavelmente Gen 3)
Capacidade:     238 GB (usável)

Veredicto para IA:
- Modelos GGUF pequenos: 0.5-3 GB cada (cabe tranquilamente)
- Datasets de texto: geralmente < 1 GB (sem problema)
- Limitação: se quiser vários modelos grandes, espaço fica apertado
- Dica: usar HDD externo ou cloud para armazenar modelos/datasets
```

### 4.2 Tabela de Capacidades (O que PODE e NÃO PODE)

| Tarefa | Pode? | Velocidade | Observações |
|---|---|---|---|
| Programar em Python | ✅ Sim | Normal | VS Code + terminal funciona bem |
| Aprender NumPy/Pandas | ✅ Sim | Normal | Datasets até ~100 MB |
| Rede neural do zero (NumPy) | ✅ Sim | Rápido | XOR, classificação simples |
| scikit-learn (ML clássico) | ✅ Sim | Normal | Datasets pequenos-médios |
| Jupyter Notebook local | ✅ Sim | Normal | Fechar browser, usar nb apenas |
| Google Colab (via browser) | ✅ Sim | Depende internet | O MELHOR caminho para deep learning |
| Ollama + Qwen 0.5B (Q2) | ⚠️ Limitado | Muito lento (1-2 tok/s) | Fechar TUDO, contexto curto |
| PyTorch local (treino) | ❌ Não | — | Sem VRAM, RAM insuficiente |
| Fine-tuning local | ❌ Não | — | Mínimo 16GB RAM + GPU NVIDIA |
| Stable Diffusion | ❌ Não | — | Sem GPU dedicada |
| Rodar modelo 7B+ local | ❌ Não | — | Precisa 8-16 GB RAM livre |

### 4.3 Upgrades Recomendados (Prioridade)

| Prioridade | Upgrade | Custo | Impacto | Dificuldade |
|---|---|---|---|---|
| 🔴 **1** | **RAM 8GB adicional (total 12-16GB)** | R$ 200-350 | ENORME: desbloqueia modelos 3-7B locais | Fácil (1 parafuso) |
| 🟡 2 | SSD externo 1TB para modelos | R$ 300-500 | Moderado: mais espaço para modelos | Trivial (USB) |
| 🟢 3 | Monitor externo | R$ 500-800 | Conforto: mais espaço de tela para código | Trivial (HDMI) |
| 🔵 4 | Desktop dedicado com RTX 3060 | R$ 5.000-7.000 | TRANSFORMADOR: fine-tuning local | Investimento |

---

## 5. Plataformas Gratuitas Detalhadas

### 5.1 Google Colab — Guia Completo

| Aspecto | Detalhes |
|---|---|
| **URL** | colab.research.google.com |
| **GPU Grátis** | NVIDIA T4 (16GB VRAM) — variável por disponibilidade |
| **RAM Grátis** | ~12 GB |
| **Disco** | ~78 GB (temporário) |
| **Timeout idle** | ~90 minutos sem atividade |
| **Duração máxima** | ~12 horas por sessão |
| **Salvar trabalho** | Google Drive (monte com `from google.colab import drive; drive.mount('/content/drive')`) |
| **Versão Pro** | R$ 50-55/mês — GPU melhor, mais RAM, sessões mais longas |
| **Versão Pro+** | R$ 250-280/mês — A100, 52GB RAM, sessões em background |

### 5.2 Kaggle Notebooks

| Aspecto | Detalhes |
|---|---|
| **URL** | kaggle.com |
| **GPU Grátis** | NVIDIA T4 ou P100 (16GB VRAM) |
| **Quota** | 30 horas de GPU por semana |
| **RAM** | ~13 GB |
| **Vantagem** | Datasets públicos integrados, competições |
| **Salvamento** | Automático, com versionamento |

### 5.3 Lightning AI Studio

| Aspecto | Detalhes |
|---|---|
| **URL** | lightning.ai |
| **GPU Grátis** | Acesso limitado a GPUs |
| **Vantagem** | Ambiente persistente (não perde progresso), muito profissional |
| **Ideal para** | Quem quer ambiente mais estável que Colab |

---

## 6. Resumo Executivo

### Para o perfil da pesquisadora (VivoBook 15, Zona Leste SP):

| Pergunta | Resposta Direta |
|---|---|
| **Posso estudar IA no meu notebook?** | SIM — Python, NumPy, scikit-learn, Colab |
| **Posso treinar um LLM do zero?** | NÃO — custaria de R$ 50K a R$ 500M |
| **Posso personalizar um LLM?** | SIM — fine-tuning gratuito no Google Colab |
| **Quanto custa o caminho viável?** | R$ 0 a R$ 350 (upgrade RAM + Colab grátis) |
| **Vale mais que um apartamento?** | O CONHECIMENTO vale mais. O apartamento é segurança patrimonial. São investimentos diferentes. |
| **Melhor primeiro investimento?** | Upgrade RAM (R$ 300) > Colab grátis > Kaggle > Cursos online |
| **Quanto tempo para ficar produtivo?** | 6-12 meses com estudo consistente |
| **É possível ganhar dinheiro com IA?** | SIM — salários de R$ 5K-30K/mês em ML. Freelance de fine-tuning: R$ 2-10K por projeto |
