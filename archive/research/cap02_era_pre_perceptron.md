# 📜 Capítulo 2 — Era Pré-Perceptron: Das Raízes Filosóficas à Conferência de Dartmouth

> Documento expandido do índice principal `fundamentos_ia_pesquisa.md`

---

## 1. As Raízes Filosóficas da Inteligência Artificial

### 1.1 Aristóteles e a Lógica Formal (~350 a.C.)

Aristóteles é considerado o pai da lógica formal ocidental. Sua obra *Organon* estabeleceu os **silogismos** — o primeiro sistema formal para dedução lógica.

**Exemplo de silogismo:**
```
Premissa 1: Todos os humanos são mortais
Premissa 2: Sócrates é humano
Conclusão:  Sócrates é mortal
```

**Por que importa para IA:**
- Aristóteles mostrou que o raciocínio pode ser **mecanizado** — reduzido a regras formais
- Isso plantou a semente: se o raciocínio segue regras, uma máquina poderia seguir essas mesmas regras
- Os **sistemas especialistas** dos anos 1970-80 eram essencialmente silogismos computadorizados
- A lógica proposicional de Aristóteles está na base dos circuitos lógicos digitais

**Outras contribuições:**
- **Categorias:** Sistema de classificação que antecipa taxonomias e ontologias computacionais
- **Teoria das Formas:** Abstração de conceitos — precursor da representação de conhecimento
- **De Anima ("Sobre a Alma"):** Tentativa de explicar a percepção, memória e imaginação como processos naturais

### 1.2 Ramon Llull e a Ars Magna (~1305)

**Ramon Llull** foi um filósofo, teólogo e místico catalão que criou a **Ars Magna** (Grande Arte) — um sistema de discos concêntricos rotativos que combinavam conceitos de forma mecânica para gerar novas proposições e argumentos.

**Como funcionava:**
- Discos de papel ou metal com conceitos escritos (Bondade, Grandeza, Eternidade, etc.)
- Girando os discos, novas combinações surgiam
- Llull acreditava que TODAS as verdades poderiam ser derivadas mecanicamente

**Por que importa:** É considerada a primeira tentativa de criar uma **"máquina de raciocínio"** — um dispositivo físico que mecaniza o pensamento. Leibniz citou Llull como inspiração.

### 1.3 Thomas Hobbes — "Razão como Cálculo" (1651)

No *Leviatã*, Hobbes escreveu:

> *"Por RAZÃO... não entendo outra coisa senão CÁLCULO (isto é, adição e subtração) das consequências de nomes gerais."*

**Contribuição radical:**
- Igualou **pensar** a **calcular** — removendo o misticismo do raciocínio
- Se pensar é calcular, e máquinas podem calcular, então máquinas podem (em teoria) pensar
- Esta é a **tese filosófica fundamental** por trás de toda IA simbólica

### 1.4 René Descartes — O Dualismo e os Autômatos (1637)

Descartes, no *Discurso do Método*, argumentou que:
- O corpo humano funciona como uma **máquina** (automaton)
- MAS a mente/alma é separada e não-mecânica (dualismo mente-corpo)
- Animais são "máquinas sem alma" — operam puramente por mecanismos

**Para a IA, Descartes levantou a questão que persiste até hoje:** Uma máquina pode *realmente* pensar, ou apenas simular pensamento? Este é o precursor direto do **Problema da Consciência Artificial** e do argumento da "Chinese Room" de John Searle (1980).

### 1.5 Gottfried Leibniz — O Sonho da Computação Universal (~1670-1714)

Leibniz foi talvez o maior visionário pré-computacional:

**Contribuições:**
1. **Characteristica Universalis:** Uma linguagem formal universal que representaria todos os conceitos humanos como símbolos manipuláveis
2. **Calculus Ratiocinator:** Uma máquina que operaria sobre esses símbolos para resolver qualquer disputa — "Calculemos!" seria seu lema
3. **Sistema Binário:** Leibniz formalizou o sistema numérico binário (0 e 1) — a base de TODA a computação digital moderna
4. **Stepped Reckoner (1694):** Construiu uma calculadora mecânica capaz de multiplicar e dividir

**A visão completa:** Leibniz imaginou que todo conhecimento humano poderia ser codificado em símbolos, e disputas filosóficas, jurídicas e científicas seriam resolvidas por *cálculo*, não por debate. Isso é essencialmente o programa da **IA simbólica** — 300 anos antes de existir.

### 1.6 George Boole — A Álgebra do Pensamento (1854)

Em *"An Investigation of the Laws of Thought"*, Boole criou a **álgebra booleana**:

**Operações fundamentais:**

| Operação | Símbolo | Significado | Exemplo |
|---|---|---|---|
| **AND (E)** | ∧ | Ambos verdadeiros | Chovendo ∧ Frio = Chovendo e Frio |
| **OR (OU)** | ∨ | Pelo menos um verdadeiro | Sol ∨ Nublado = Sol ou Nublado |
| **NOT (NÃO)** | ¬ | Inversão | ¬Chovendo = Não está chovendo |

**Impacto devastador:**
- Boole reduziu a lógica formal a operações algébricas sobre 0 e 1
- Em 1937, **Claude Shannon** demonstrou que circuitos elétricos com relés podiam implementar álgebra booleana
- TODOS os computadores digitais modernos são, em última instância, máquinas booleanas
- As **portas lógicas** (AND, OR, NOT, XOR) são implementações físicas da álgebra de Boole

### 1.7 Gottlob Frege — Lógica de Predicados (1879)

Frege publicou *Begriffsschrift* ("Escrita Conceitual"), estendendo a lógica de Boole com:
- **Quantificadores** (∀ = para todo, ∃ = existe)
- **Predicados** (propriedades e relações)
- **Variáveis** ligadas por quantificadores

Isso permitiu expressar afirmações muito mais complexas e é a base da **lógica de primeira ordem**, usada em IA simbólica e bases de conhecimento.

---

## 2. Os Precursores Mecânicos

### 2.1 Charles Babbage e Ada Lovelace (1830s-1840s)

**Charles Babbage** projetou duas máquinas revolucionárias:

**Máquina Diferencial (1822):**
- Calculadora mecânica para tabelas de logaritmos
- Funcionava por engrenagens e alavancas
- Nunca completada por falta de financiamento e precisão mecânica

**Máquina Analítica (1837):**
- O primeiro **computador de propósito geral** da história (projetado, não construído)
- Tinha: entrada (cartões perfurados), processamento ("mill"), memória ("store"), saída (impressora)
- Seria programável — poderia executar diferentes algoritmos

**Ada Lovelace:**
- Filha do poeta Lord Byron
- Escreveu o que é considerado o **primeiro programa de computador** — um algoritmo para calcular números de Bernoulli na Máquina Analítica
- Fez a observação visionária: *"A Máquina Analítica não tem nenhuma pretensão de originar nada. Pode fazer qualquer coisa que nós soubemos mandá-la fazer."*
- Esta observação antecipa o debate sobre criatividade e consciência artificial

### 2.2 Leonardo Torres y Quevedo — O Autômato de Xadrez (1912-1914)

O engenheiro espanhol construiu **El Ajedrecista** — um autômato eletromecânico capaz de jogar o final de xadrez Rei+Torre vs Rei sem intervenção humana.

**Importância:**
- Um dos primeiros dispositivos a demonstrar **tomada de decisão autônoma**
- Usava sensores eletromagnéticos para detectar as peças no tabuleiro
- Seguia uma árvore de decisão fixa — mas resolvia um problema real
- Podia dar xeque-mate em qualquer posição válida (com jogo imperfeito, mas funcional)

### 2.3 Karel Čapek — R.U.R. e a Palavra "Robô" (1921)

A peça teatral *R.U.R.* (*Rossum's Universal Robots*) introduziu a palavra **"robô"** no vocabulário mundial.

- Do tcheco **robota** = trabalho forçado/servidão
- Na peça, robôs são seres artificiais biológicos (não metálicos) criados para servir humanos
- **Spoiler histórico:** Os robôs se revoltam e exterminam a humanidade — estabelecendo o tropo cultural do "robô rebelde" que influencia o medo de IA até hoje

---

## 3. Alan Turing — O Arquiteto Teórico

### 3.1 "On Computable Numbers" (1936) — A Máquina de Turing

Aos 24 anos, Turing publicou um dos papers mais importantes da história da ciência.

**O problema original:** O *Entscheidungsproblem* (problema da decisão) de Hilbert — é possível criar um procedimento mecânico que determine se qualquer afirmação matemática é verdadeira ou falsa?

**A solução de Turing:**
1. Definiu formalmente o conceito de **computação** usando uma máquina abstrata
2. A **Máquina de Turing** consiste em:
   - Uma **fita** infinita dividida em células (cada uma com um símbolo)
   - Uma **cabeça** de leitura/escrita que move pela fita
   - Uma **tabela de estados** que define as regras (se no estado X e lendo símbolo Y, escreva Z, mova para a direita e vá para o estado W)
3. Provou que existem problemas **não-computáveis** (o *Halting Problem*)
4. Demonstrou que uma **Máquina de Turing Universal** pode simular qualquer outra Máquina de Turing

**Impacto:** Estabeleceu os limites teóricos da computação. Todo computador moderno é, essencialmente, uma implementação física da Máquina de Turing Universal.

### 3.2 Trabalho em Bletchley Park (1939-1945)

Durante a Segunda Guerra Mundial, Turing liderou a equipe que quebrou o código **Enigma** dos nazistas:
- Construiu a máquina **Bombe** para decifrar mensagens criptografadas
- Estima-se que o trabalho de Turing encurtou a guerra em **2-3 anos**, salvando milhões de vidas
- Aplicou princípios estatísticos e probabilísticos à criptoanálise — precursores de técnicas de machine learning

### 3.3 "Computing Machinery and Intelligence" (1950) — O Teste de Turing

Este paper na revista *Mind* é considerado o **documento fundador da IA como campo filosófico**.

**A pergunta central:** "Máquinas podem pensar?"

Turing considerou a pergunta mal formulada e propôs substituí-la pelo **Jogo da Imitação**:

```
Arranjo:
- Interrogador (C) em uma sala
- Humano (A) em outra sala  
- Máquina (B) em outra sala
- Comunicação apenas por texto

Teste: Se C não consegue distinguir consistentemente
       A de B, então B "pensa" no sentido operacional.
```

**As 9 Objeções que Turing refutou:**

| Objeção | Argumento | Refutação de Turing |
|---|---|---|
| 1. Teológica | "Pensar requer alma" | A onipotência de Deus não exclui conceder almas a máquinas |
| 2. "Cabeça na areia" | "É assustador demais" | Não é argumento, é medo |
| 3. Matemática | Gödel/incompletude limita máquinas | Humanos também cometem erros e têm limitações |
| 4. Consciência | Máquina não "sente" | Solipsismo — como provar que outros humanos sentem? |
| 5. Incapacidades | "Não pode amar, ser criativo..." | Falácia — lista arbitrária de capacidades humanas |
| 6. Lady Lovelace | "Máquinas só fazem o que mandamos" | Máquinas podem surpreender seus criadores |
| 7. Continuidade do SN | Cérebro não é discreto | Aproximação discreta pode ser suficiente |
| 8. Informalidade | Comportamento não é regras | Não há prova de que humanos não seguem regras |
| 9. Percepção Extra-Sensorial | Telepatia existe | Turing levou a sério (!), sugeriu blindagem |

**Proposta de "Máquinas-Criança":**
Turing sugeriu que, em vez de programar toda a inteligência, seria melhor criar máquinas que **aprendem como crianças** — através de experiência, educação e interação. Esta é a ideia central do **machine learning** e do **aprendizado por reforço**.

---

## 4. O Neurônio de McCulloch-Pitts (1943) — Em Profundidade

### 4.1 Os Autores

**Warren McCulloch (1898-1969):**
- Neurofisiologista e filósofo
- Médico formado em Columbia
- Interessado em como o cérebro implementa a lógica
- Trabalhou em Yale e depois no MIT

**Walter Pitts (1923-1969):**
- Prodígio matemático autodidata
- Fugiu de casa aos 12 anos
- Encontrou McCulloch aos 18 — uma das parcerias mais frutíferas da história da ciência
- Vida trágica: morreu aos 46 de alcoolismo, em isolamento

### 4.2 O Paper: "A Logical Calculus of the Ideas Immanent in Nervous Activity"

**Tese central:** As atividades do sistema nervoso podem ser tratadas como **proposições lógicas**, e redes de neurônios podem implementar qualquer computação possível.

**O modelo formal:**

```
Neurônio no tempo t+1:
  
  Se (soma das entradas excitatórias ≥ θ) E (nenhuma entrada inibitória ativa):
      Estado = 1 (dispara)
  Senão:
      Estado = 0 (repouso)

Onde θ = limiar (threshold)
```

**Implementação de portas lógicas:**

```
AND (θ = 2):           OR (θ = 1):            NOT (θ = 0, com inibição):
  x1  x2  Saída          x1  x2  Saída          x1(inibitório)  Saída
   0   0   0               0   0   0                  0            1
   0   1   0               0   1   1                  1            0
   1   0   0               1   0   1
   1   1   1               1   1   1
```

### 4.3 Significado Histórico

- **Primeiro modelo matemático de computação neural** — ponte entre neurociência e matemática
- Provaram que redes de neurônios são **Turing-completas** — podem computar qualquer função computável
- Inspiraram diretamente John von Neumann no design de computadores
- Abriram o caminho para Rosenblatt criar o perceptron (adicionando aprendizado)

---

## 5. Cibernética — Norbert Wiener (1948)

### 5.1 O Livro e o Movimento

*"Cybernetics: or Control and Communication in the Animal and the Machine"* fundou um movimento interdisciplinar que reuniu:
- Matemáticos (Wiener, Shannon)
- Neurocientistas (McCulloch)
- Engenheiros (Bigelow)
- Antropólogos (Margaret Mead, Gregory Bateson)
- Psicólogos

### 5.2 Conceitos Fundamentais

**Feedback Negativo (Retroalimentação):**
```
Objetivo → Ação → Resultado → Comparação com Objetivo → Correção → Nova Ação
              ↑                                              ↓
              ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

Exemplos:
- **Termostato:** Mede temperatura → compara com setpoint → liga/desliga aquecedor
- **Cerebelo:** Mede posição do braço → compara com intenção → corrige movimento
- **Backpropagation:** Mede erro → compara com saída desejada → ajusta pesos

**Homeostase:** Manutenção de estabilidade por autorregulação — conceito emprestado da fisiologia de Walter Cannon.

### 5.3 As Conferências Macy (1946-1953)

Série de conferências interdisciplinares onde McCulloch, Wiener, von Neumann, Shannon, Mead e outros discutiram a convergência entre cérebros e máquinas. Estas conferências foram o cadinho intelectual de onde emergiram tanto a IA quanto a ciência cognitiva.

---

## 6. A Conferência de Dartmouth (1956)

### 6.1 A Proposta Original

A proposta para o "Dartmouth Summer Research Project on Artificial Intelligence" foi escrita em agosto de 1955 por:
- **John McCarthy** (Dartmouth) — cunhou o termo "Inteligência Artificial"
- **Marvin Minsky** (Harvard/MIT) — depois co-fundador do MIT AI Lab
- **Nathaniel Rochester** (IBM) — designer do IBM 701
- **Claude Shannon** (Bell Labs) — pai da Teoria da Informação

### 6.2 A Ambição Original

A proposta afirmava que o projeto levaria **2 meses com 10 pesquisadores** e que avanços significativos seriam feitos em:
- Computadores usando linguagem natural
- Redes neurais
- Auto-melhoria de máquinas
- Abstração e criatividade

**Realidade:** O workshop durou 8 semanas no verão de 1956, com ~20 participantes. Nenhum avanço definitivo ocorreu, mas o campo foi formalmente fundado.

### 6.3 Participantes e seu Legado

| Participante | Contribuição Posterior |
|---|---|
| **John McCarthy** | Criou LISP; fundou o Stanford AI Lab |
| **Marvin Minsky** | Co-fundou MIT AI Lab; escreveu *Perceptrons* (1969) |
| **Allen Newell & Herbert Simon** | Logic Theorist (primeiro programa de IA); Nobel de Economia (Simon, 1978) |
| **Arthur Samuel** | Programa de damas que aprendia jogando (precursor do RL) |
| **Claude Shannon** | Teoria da Informação; analisou xadrez computacionalmente |
| **Ray Solomonoff** | Fundamentos da inferência algorítmica |

### 6.4 O Otimismo Inicial (e sua queda)

Após Dartmouth, o campo estava tomado por otimismo extremo:

- **Herbert Simon (1957):** "Dentro de dez anos, um computador será campeão mundial de xadrez" (levou 40 anos — Deep Blue, 1997)
- **Marvin Minsky (1967):** "Dentro de uma geração, o problema de criar 'inteligência artificial' será substancialmente resolvido" (ainda não foi)
- **ARPA/DARPA** começou a financiar pesquisa em IA generosamente

Esse otimismo desmedido plantou as sementes para as decepções que viriam — e os cortes de financiamento que causaram o primeiro Inverno da IA.

---

> **Referências:** Aristóteles "Organon", Llull "Ars Magna" (1305), Hobbes "Leviathan" (1651), Leibniz "Dissertatio de Arte Combinatoria" (1666), Boole "Laws of Thought" (1854), Turing "On Computable Numbers" (1936), Turing "Computing Machinery and Intelligence" (1950), McCulloch & Pitts (1943), Wiener "Cybernetics" (1948), McCarthy et al. proposta de Dartmouth (1955).
