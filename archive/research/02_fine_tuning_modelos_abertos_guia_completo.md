# 🔧 Fine-Tuning de Modelos Abertos — Guia Completo e Aprofundado

> Documento complementar: detalhamento técnico de como personalizar Gemma, Llama e outros LLMs para propósitos pessoais

---

## 1. O que é Fine-Tuning (em profundidade)

### 1.1 Analogia Detalhada

Imagine que treinar um LLM do zero é como **educar uma criança desde o nascimento** — leva décadas e custa milhões. Fine-tuning é como **contratar um médico formado e dar um curso de especialização** em dermatologia. Ele já sabe medicina; você só precisa ensinar a especialidade.

O modelo base (Gemma, Llama, etc.) já "leu" trilhões de palavras e aprendeu:
- Gramática e sintaxe de dezenas de idiomas
- Conhecimento factual geral
- Capacidade de raciocínio lógico
- Seguir instruções

Com fine-tuning, você **ajusta** esse conhecimento para:
- Responder num tom específico (formal, informal, técnico)
- Seguir um formato de saída (JSON, tabelas, relatórios)
- Ser especialista em um domínio (direito, medicina, culinária)
- Falar como uma persona específica

### 1.2 Tipos de Fine-Tuning

| Tipo | O que faz | Custo Computacional | Quando usar |
|---|---|---|---|
| **Full Fine-Tuning** | Atualiza TODOS os parâmetros do modelo | Altíssimo (precisa de muita VRAM) | Quando tem hardware robusto e muitos dados |
| **LoRA** | Congela o modelo, treina pequenos adaptadores de rank baixo | Moderado | Quando quer qualidade com menos VRAM |
| **QLoRA** | LoRA + modelo base quantizado em 4 bits | Baixo | Quando tem pouca VRAM (Colab grátis) |
| **Prefix Tuning** | Treina vetores de "prefixo" prepended às camadas | Baixo | Quando quer múltiplas tarefas no mesmo modelo |
| **Adapter Tuning** | Insere módulos treináveis entre camadas congeladas | Moderado | Alternativa ao LoRA |

---

## 2. Como Funciona o LoRA — Matemática Simplificada

### 2.1 O Problema Original

Um modelo como Gemma 7B tem ~7 bilhões de parâmetros. Para fine-tuning completo, você precisaria:
- Armazenar 7B de parâmetros (em float16 = ~14GB)
- Armazenar os gradientes (~14GB)
- Armazenar os estados do otimizador Adam (~28GB)
- **Total: ~56GB de VRAM** — impossível em GPUs consumidor

### 2.2 A Solução LoRA

LoRA decompõe as atualizações de peso em duas matrizes menores:

```
Peso Original (W): Dimensão d × d (ex: 4096 × 4096)
  → Congelado, não muda

Adaptador LoRA:
  A: Dimensão d × r (ex: 4096 × 16)
  B: Dimensão r × d (ex: 16 × 4096)

  Peso Efetivo = W + A × B

  Parâmetros treináveis: 2 × d × r = 2 × 4096 × 16 = 131.072
  vs. original: d × d = 4096 × 4096 = 16.777.216

  Redução: ~128x menos parâmetros!
```

### 2.3 Hiperparâmetros do LoRA

| Parâmetro | O que controla | Valores recomendados | Impacto |
|---|---|---|---|
| **r (rank)** | Tamanho dos adaptadores | 8, 16, 32, 64 | Maior = mais capacidade, mais VRAM |
| **lora_alpha** | Escala da contribuição LoRA | 2× o rank (ex: r=16, alpha=32) | Maior = LoRA influencia mais |
| **lora_dropout** | Regularização dos adaptadores | 0.05 a 0.1 | Previne overfitting |
| **target_modules** | Quais camadas recebem LoRA | q_proj, k_proj, v_proj, o_proj | Mais módulos = mais flexibilidade |

---

## 3. Pipeline Completo de Fine-Tuning (Passo a Passo)

### 3.1 Passo 1 — Preparar o Dataset

O formato mais comum é **instrução-resposta** em JSONL:

```json
{"instruction": "Traduza para coreano: Bom dia", "output": "좋은 아침이에요 (joeun achimieyo)"}
{"instruction": "Qual é o prato principal do Brasil?", "output": "O prato mais emblemático do Brasil é a feijoada, feita com feijão preto e diversas carnes de porco."}
{"instruction": "Explique fotossíntese para uma criança de 8 anos", "output": "As plantas comem luz do sol! Elas usam a luz, a água e o ar para fazer sua própria comida. É como se elas tivessem uma cozinha mágica dentro das folhas verdes."}
```

**Regras de ouro para datasets:**
- **Qualidade > Quantidade:** 500 exemplos excelentes > 5.000 exemplos mediocres
- **Diversidade:** Varie os tipos de pergunta, tom e complexidade
- **Consistência:** Mantenha o mesmo formato e estilo em todas as respostas
- **Sem erros:** Revise manualmente — o modelo aprende TUDO, inclusive erros

### 3.2 Passo 2 — Escolher o Modelo Base

**Para seu hardware (via Colab gratuito):**

| Modelo | Por que escolher | VRAM (QLoRA) |
|---|---|---|
| **Gemma 2 2B** | Excelente qualidade para o tamanho, feito pelo Google | ~6 GB |
| **Phi-3 mini (3.8B)** | Muito capaz para o tamanho, raciocínio forte | ~7 GB |
| **Qwen 2.5 1.5B** | Muito leve, bom em múltiplos idiomas | ~4 GB |
| **Llama 3.2 3B** | Boa qualidade geral, comunidade enorme | ~7 GB |
| **Gemma 2 9B** | Qualidade superior, mas precisa Colab Pro ou T4 com cuidado | ~12 GB |

### 3.3 Passo 3 — Configurar o Ambiente (Google Colab)

```python
# Célula 1 — Instalar dependências
!pip install -q unsloth transformers trl peft bitsandbytes datasets accelerate

# Célula 2 — Verificar GPU disponível
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### 3.4 Passo 4 — Carregar Modelo com QLoRA

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-2b-bnb-4bit",  # versão já quantizada
    max_seq_length=2048,
    load_in_4bit=True,      # QLoRA: modelo base em 4 bits
    dtype=None,              # auto-detect (float16 ou bfloat16)
)

# Aplicar LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # rank — 16 é bom equilíbrio
    lora_alpha=32,           # alpha = 2x rank
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",  # economiza VRAM
)
```

### 3.5 Passo 5 — Preparar os Dados

```python
from datasets import load_dataset

# Carregar seu dataset (pode ser de um arquivo local ou Hugging Face)
dataset = load_dataset("json", data_files="meu_dataset.jsonl", split="train")

# Template de prompt (específico para cada modelo)
prompt_template = """<bos><start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>"""

def format_prompt(example):
    return {"text": prompt_template.format(**example)}

dataset = dataset.map(format_prompt)
```

### 3.6 Passo 6 — Treinar

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="./resultados",
        num_train_epochs=3,                # 2-5 épocas é o ideal
        per_device_train_batch_size=2,     # ajuste conforme VRAM
        gradient_accumulation_steps=4,     # simula batch maior
        learning_rate=2e-4,                # taxa de aprendizado
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,                         # precisão mista
    ),
)

trainer.train()
```

### 3.7 Passo 7 — Testar e Salvar

```python
# Testar o modelo ajustado
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    "<bos><start_of_turn>user\nQual a capital da Coreia do Sul?<end_of_turn>\n<start_of_turn>model\n",
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Salvar adaptadores LoRA
model.save_pretrained("meu_modelo_coreano")
tokenizer.save_pretrained("meu_modelo_coreano")

# OU mesclar LoRA no modelo base e salvar versão completa
model.save_pretrained_merged("modelo_final", tokenizer, save_method="merged_16bit")
```

---

## 4. Exemplos Práticos de Propósitos Pessoais

### 4.1 Assistente de Coreano (para o app Sejong)

```json
{"instruction": "Como se diz 'obrigado' em coreano?", "output": "감사합니다 (gamsahamnida) — formal\n고마워 (gomawo) — informal\n\nUse 감사합니다 com professores, mais velhos e desconhecidos. Use 고마워 com amigos próximos da mesma idade."}
{"instruction": "Conjugue o verbo 가다 (ir) no presente formal", "output": "가다 → 갑니다 (gamnida)\n\nRegra: Para verbos terminados em 'ㅏ' ou 'ㅗ', remove o 다 e adiciona ㅂ니다.\nExemplo: 저는 학교에 갑니다 (Eu vou para a escola)."}
```

### 4.2 Assistente de Vendas (para a loja Sejong)

```json
{"instruction": "Cliente perguntou sobre o Sejong VIP Box", "output": "O Sejong VIP Box é nosso combo especial que inclui [produtos]. O preço é [valor] e a vantagem é que o cliente economiza [X]% em comparação com comprar separado. Posso embalar para presente também!"}
{"instruction": "Como lidar com cliente que reclama do preço", "output": "Entendo sua preocupação com o valor. Nossos produtos são artesanais/importados com qualidade premium. Temos opções em diferentes faixas de preço. Posso mostrar alternativas que se encaixam melhor no seu orçamento?"}
```

### 4.3 Tutor Pessoal de Estudos

```json
{"instruction": "Explique derivadas", "output": "Derivada é a taxa de variação instantânea de uma função. Imagine que você está dirigindo: o velocímetro mostra sua velocidade NAQUELE instante — isso é a derivada da posição em relação ao tempo.\n\nFórmula: f'(x) = lim(h→0) [f(x+h) - f(x)] / h\n\nExemplo simples: f(x) = x² → f'(x) = 2x\nIsso significa que em x=3, a taxa de variação é 2×3 = 6."}
```

---

## 5. Dicas Avançadas

### 5.1 Evitando Problemas Comuns

| Problema | Causa | Solução |
|---|---|---|
| **Overfitting** | Modelo memoriza dados em vez de aprender | Reduzir épocas (2-3), usar dropout, mais dados diversos |
| **Catastrophic Forgetting** | Modelo perde habilidades gerais | Usar LoRA (não muda pesos originais), incluir dados gerais no dataset |
| **Out of Memory (OOM)** | VRAM insuficiente | Reduzir batch_size, usar gradient_checkpointing, reduzir rank LoRA |
| **Respostas genéricas** | Dataset muito pequeno ou pouco diverso | Aumentar dataset (mínimo 500 exemplos), melhorar qualidade |
| **Modelo "alucinando"** | Dados conflitantes ou insuficientes | Limpar dataset, adicionar exemplos de "não sei" |

### 5.2 Quanto Custa de Verdade (Cenários Práticos)

| Cenário | Plataforma | Modelo | Dataset | Tempo | Custo Total |
|---|---|---|---|---|---|
| Estudante iniciante | Colab Grátis | Gemma 2B | 500 exemplos | ~30 min | **R$ 0** |
| Projeto pessoal sério | Colab Pro | Gemma 9B | 2.000 exemplos | ~2h | **~R$ 50/mês** |
| Startup | RunPod A100 | Llama 70B | 10.000 exemplos | ~8h | **~R$ 60** |
| Empresa | Cluster próprio | Custom 7B | 100K+ exemplos | ~dias | **R$ 5K-50K** |

### 5.3 Onde Encontrar Datasets

- **Hugging Face Datasets:** huggingface.co/datasets (milhares de datasets gratuitos)
- **Criar manualmente:** Para propósitos pessoais, curar 500-2000 exemplos à mão é o mais eficaz
- **Sintéticos:** Usar GPT-4/Claude para gerar exemplos de treino (depois revisar manualmente)
- **Kaggle:** kaggle.com/datasets (excelente para dados tabulares e competições)

---

## 6. Comparação Final: Modelos Abertos vs APIs Fechadas

### Para o seu caso específico (pesquisador com VivoBook 15):

| Critério | Fine-Tuning Gemma/Llama (Colab) | Usar API do ChatGPT/Claude |
|---|---|---|
| **Custo mensal** | R$ 0 (Colab grátis) | R$ 0-100 (free tier / Plus) |
| **Privacidade** | Total (dados ficam com você) | Dados vão para OpenAI/Anthropic |
| **Personalização** | Alta (treina no SEU estilo) | Média (prompt engineering) |
| **Qualidade geral** | Boa (modelos menores) | Excelente (modelos enormes) |
| **Funciona offline** | Sim (se exportar modelo) | Não |
| **Aprendizado** | Enorme (entende como IA funciona) | Baixo (usa como ferramenta) |
| **Recomendação** | Para APRENDER e projetos específicos | Para USAR no dia a dia |

> **Conclusão:** Ambos se complementam. Use APIs para produtividade diária e fine-tuning para aprender e criar ferramentas personalizadas.
