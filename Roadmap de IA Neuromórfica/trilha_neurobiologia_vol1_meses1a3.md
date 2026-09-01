# 🧠 Trilha de Neurobiologia para IA Bioinspirada
## Volume I — Meses 1 a 3: Das Bases ao Hipocampo

> **Objetivo desta trilha:** Sair do zero em neurobiologia e construir o alicerce científico para implementar um hipocampo e neocórtex artificiais em Python puro.
>
> **Perfil:** Iniciante em neurobiologia, programador com projeto em andamento.
> **Carga:** 4h/semana → 2h estudo + 1h30 prática + 0h30 revisão/verificação.

---

## O MAPA DO QUE VOCÊ VAI CONSTRUIR

Antes de começar, você precisa ver o alvo. Seu sistema final:

```
[Entrada Sensorial / Contexto]
           ↓
   [Neocórtex Sensorial]
    • processa padrões
    • aprende lentamente
    • generaliza
           ↓
   [Córtex Entorrinal (EC)]     ← interface hipocampo ↔ córtex
           ↓
      [HIPOCAMPO]
   ┌──────────────────────┐
   │  Giro Dentado (DG)   │  ← separa padrões similares
   │         ↓            │
   │         CA3          │  ← completa padrões (memória associativa)
   │         ↓            │
   │         CA1          │  ← saída / indexação cortical
   └──────────────────────┘
           ↓
   [Neocórtex Associativo]
    • recebe replay hipocampal
    • consolida em memória de longo prazo
```

Cada semana você estuda e codifica um módulo dessa arquitetura. Ao fim de 6 meses, você terá o sistema inteiro.

---

## ⚠️ COMO USAR ESTA TRILHA

1. **Leia o exercício prático ANTES de estudar** — assim você sabe onde está indo
2. **Não pule semanas** — cada semana é dependência da próxima
3. **Verificação é obrigatória** — se não passou nos critérios, repita antes de avançar
4. **Conecte ao seu código imediatamente** — após cada semana, escreva um comentário longo no seu projeto explicando como o conceito que você aprendeu aparece nele
5. **Anote as analogias** — neurobiologia em um lado, implementação Python no outro

---

---

# ═══════════════════════════════════
# MESES 1–2: O NEURÔNIO E A PLASTICIDADE
# ═══════════════════════════════════

---

# MÊS 1 — O NEURÔNIO

---

## SEMANA 1
**Tema: Anatomia do Neurônio**

### 📚 O que estudar (2h)

O que importa para o seu projeto — e só isso:

- **Soma** (corpo celular): integra entradas, decide se dispara
- **Dendrites**: recebem sinais de outros neurônios (entradas)
- **Axônio**: transmite o sinal de saída para outros neurônios
- **Botão sináptico**: ponto de contato com o próximo neurônio
- **Bainha de mielina**: acelera a transmissão (você não vai implementar isso agora)

Anote estas 3 perguntas e responda após o estudo:
1. Por que o neurônio precisa de dendrito E axônio separados?
2. O que torna um neurônio diferente de uma célula comum?
3. Como o "não disparo" também pode ser informação?

**Recursos gratuitos:**
- YouTube: "Neuroscientifically Challenged - Neuron Basics" (busca exata, ~10min)
- Khan Academy: busca "neuron structure" no site ou YouTube deles
- BrainFacts.org → "Cells of the Nervous System" (gratuito, inglês simples)

### ⚗️ Exercício Prático (1h30)

Implemente a classe base do seu neurônio. Esta classe vai evoluir por toda a trilha:

```python
# neuron.py — classe base, semana 1
import numpy as np

class Neuron:
    def __init__(self, neuron_id, neuron_type='excitatory'):
        self.id = neuron_id
        self.type = neuron_type      # 'excitatory' ou 'inhibitory'

        # Propriedades do soma (biofísica real em mV)
        self.membrane_potential = -70.0   # potencial de membrana atual
        self.threshold = -55.0            # limiar de disparo
        self.resting_potential = -70.0    # potencial de repouso

        # Estado de atividade
        self.is_firing = False
        self.spike_history = []           # registro de todos os spikes

    def receive_input(self, current):
        """Dendritos recebem corrente e somam ao potencial"""
        self.membrane_potential += current

    def check_threshold(self, timestep):
        """Soma decide se dispara"""
        if self.membrane_potential >= self.threshold:
            self.is_firing = True
            self.spike_history.append(timestep)
            self.reset()
        else:
            self.is_firing = False
            self.decay()

    def reset(self):
        """Após disparo: retorna ao repouso (período refratário simples)"""
        self.membrane_potential = self.resting_potential

    def decay(self):
        """Sem disparo: potencial decai lentamente (bombas iônicas)"""
        self.membrane_potential += (self.resting_potential - self.membrane_potential) * 0.1

    def __repr__(self):
        return f"Neuron({self.id}, V={self.membrane_potential:.1f}mV, firing={self.is_firing})"


# Script de teste — rode isso separado
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    neurons = [Neuron(f'N{i}') for i in range(5)]
    currents = [0, 5, 10, 15, 20]  # correntes diferentes para cada neurônio
    T = 100  # passos de tempo

    history = {n.id: [] for n in neurons}

    for t in range(T):
        for n, I in zip(neurons, currents):
            n.receive_input(I * 0.5)
            n.check_threshold(t)
            history[n.id].append(n.membrane_potential)

    plt.figure(figsize=(12, 6))
    for n_id, vals in history.items():
        plt.plot(vals, label=n_id)
    plt.axhline(-55, color='red', linestyle='--', label='threshold (-55mV)')
    plt.xlabel('Passo de Tempo')
    plt.ylabel('Potencial de Membrana (mV)')
    plt.title('5 Neurônios com Correntes Diferentes')
    plt.legend()
    plt.savefig('semana1_neurons.png')
    print("Gráfico salvo: semana1_neurons.png")
```

### ✅ Como saber se absorveu

Critério mínimo para avançar para a Semana 2:

- [ ] Seu código cria um neurônio que dispara quando a corrente é suficiente e não dispara quando não é
- [ ] Você consegue explicar em voz alta o que `membrane_potential`, `threshold` e `resting_potential` representam biologicamente
- [ ] Você sabe por que o potencial decai quando não há entrada (resposta correta: bombas Na⁺/K⁺ restauram o equilíbrio iônico)
- [ ] Você respondeu as 3 perguntas de absorção por escrito

---

## SEMANA 2
**Tema: Potencial de Ação — a linguagem do cérebro**

### 📚 O que estudar (2h)

- **Potencial de repouso (~-70mV):** equilíbrio iônico com mais K⁺ dentro e Na⁺ fora
- **Limiar (~-55mV):** ponto de não-retorno — se passar, o spike acontece completo
- **Despolarização:** canais de Na⁺ se abrem em cascata, potencial sobe para ~+40mV
- **Repolarização:** canais de K⁺ se abrem, potencial retorna
- **Hiperpolarização:** potencial cai abaixo do repouso brevemente (período refratário)
- **Lei do tudo-ou-nada:** o neurônio dispara com força total ou não dispara

**O que isso significa para o seu código:**
O spike é um evento binário. A informação está na *taxa de spikes* (quantos por segundo) e no *timing* (quando exatamente). Esses são os dois canais de comunicação neural.

**O modelo Integrate-and-Fire (I&F):**
Você vai implementar este modelo. A equação é:
```
τ × dV/dt = -(V - V_rest) + R × I
```
Onde τ é a constante de tempo (~10ms), R é resistência de membrana, I é a corrente de entrada.

**Recursos gratuitos:**
- YouTube: "Neuroscientifically Challenged - Action Potential" (~12min)
- YouTube: "Ninja Nerd Science - Action Potential" (mais detalhado, ~20min)
- Khan Academy: "Electrotonic and action potentials"

### ⚗️ Exercício Prático (1h30)

Atualize sua classe para um Integrate-and-Fire biofísico real:

```python
# if_neuron.py — Semana 2
import numpy as np
import matplotlib.pyplot as plt

class IFNeuron:
    """
    Integrate-and-Fire Neuron
    Modelo biofísico simplificado mas computacionalmente honesto.
    Ignora canais iônicos específicos porque para efeitos de rede
    só precisamos do comportamento de threshold.
    """
    def __init__(self, neuron_id, tau=10.0, R=1.0):
        self.id = neuron_id

        # Parâmetros biofísicos
        self.tau = tau          # constante de tempo (ms)
        self.R = R              # resistência de membrana (MΩ)
        self.V_rest = -70.0     # mV
        self.V_threshold = -55.0
        self.V_reset = -80.0    # hiperpolarização pós-spike
        self.t_refract = 2.0    # período refratário absoluto (ms)

        # Estado
        self.V = self.V_rest
        self.last_spike_time = -999.0
        self.spike_times = []

    def step(self, I_input, t, dt=0.1):
        """
        Um passo de tempo (dt em ms).
        Retorna True se disparou neste passo.
        """
        # Período refratário: neurônio não responde
        if (t - self.last_spike_time) < self.t_refract:
            self.V = self.V_reset
            return False

        # Equação diferencial da membrana
        dV = (-(self.V - self.V_rest) + self.R * I_input) / self.tau
        self.V += dV * dt

        # Verificação de threshold
        if self.V >= self.V_threshold:
            self.V = self.V_reset
            self.last_spike_time = t
            self.spike_times.append(t)
            return True

        return False

    def firing_rate(self):
        """Taxa de disparo em Hz (spikes/segundo)"""
        if len(self.spike_times) < 2:
            return 0.0
        duration = (self.spike_times[-1] - self.spike_times[0]) / 1000.0  # ms → s
        return len(self.spike_times) / max(duration, 0.001)


# Experimento 1: Corrente constante
def experiment_constant_current():
    dt = 0.1    # ms
    T = 200.0   # ms de simulação
    times = np.arange(0, T, dt)

    neuron = IFNeuron('N1')
    V_trace = []
    I = 15.0    # corrente constante

    for t in times:
        fired = neuron.step(I, t, dt)
        V_trace.append(neuron.V)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(times, V_trace, 'b-', linewidth=0.8)
    ax1.axhline(neuron.V_threshold, color='red', linestyle='--', label='threshold')
    ax1.set_ylabel('Potencial (mV)')
    ax1.set_title(f'I&F com I={I}nA — Taxa: {neuron.firing_rate():.1f} Hz')
    ax1.legend()

    # Raster dos spikes
    ax2.eventplot([neuron.spike_times], lineoffsets=0, linelengths=0.8, color='black')
    ax2.set_xlabel('Tempo (ms)')
    ax2.set_ylabel('Spikes')
    ax2.set_xlim(0, T)

    plt.tight_layout()
    plt.savefig('semana2_if_neuron.png')
    print(f"Taxa de disparo: {neuron.firing_rate():.1f} Hz")


# Experimento 2: Curva F-I (firing rate vs input current)
def experiment_fi_curve():
    currents = np.linspace(0, 30, 50)
    rates = []
    dt = 0.1
    T = 500.0

    for I in currents:
        neuron = IFNeuron('test')
        times = np.arange(0, T, dt)
        for t in times:
            neuron.step(I, t, dt)
        rates.append(neuron.firing_rate())

    plt.figure(figsize=(8, 5))
    plt.plot(currents, rates, 'b-o', markersize=3)
    plt.xlabel('Corrente de Entrada (nA)')
    plt.ylabel('Taxa de Disparo (Hz)')
    plt.title('Curva F-I do Neurônio Integrate-and-Fire')
    plt.grid(True, alpha=0.3)
    plt.savefig('semana2_fi_curve.png')


if __name__ == '__main__':
    experiment_constant_current()
    experiment_fi_curve()
    print("Gráficos salvos.")
```

### ✅ Como saber se absorveu

- [ ] Seu neurônio mostra hiperpolarização após o spike (V vai para -80mV, abaixo do repouso -70mV)
- [ ] Você consegue mudar `tau` e observar como isso muda a velocidade de integração
- [ ] Você gerou a curva F-I e vê que há uma corrente mínima para o neurônio disparar (reobase)
- [ ] Você consegue explicar: "Por que o modelo I&F ignora os canais de Na⁺ e K⁺?" → porque para comportamento de rede, só precisamos da threshold decision, não da dinâmica iônica completa

---

## SEMANA 3
**Tema: Sinapses — como neurônios se comunicam**

### 📚 O que estudar (2h)

- **Sinapse química:** pré-sináptico libera neurotransmissor → pós-sináptico recebe e muda de potencial
- **Glutamato:** excitatório — aumenta V do neurônio pós-sináptico (traz para o threshold)
- **GABA:** inibitório — diminui V do pós-sináptico (afasta do threshold)
- **Peso sináptico:** eficácia da conexão (será aprendido com STDP/Hebbian mais tarde)
- **EPSP / IPSP:** potencial pós-sináptico excitatório vs inibitório
- **Potencial de reversão:** GABA tem E_rev ~ -80mV; mesmo quando V já está baixo, ele ainda inibe

**A distinção crítica para seu projeto:**
Sinapses excitatórias (E) e inibitórias (I) em equilíbrio são o que dá estabilidade às suas redes. Sem GABA, redes explodem em atividade sincrônica (como uma convulsão epiléptica computacional).

**Recursos gratuitos:**
- YouTube: "Neuroscientifically Challenged - Synaptic Transmission" (busca exata)
- Khan Academy: "Synaptic vesicles and neurotransmitters"
- Wikipedia: "Excitatory postsynaptic potential" — seção técnica surpreendentemente boa

### ⚗️ Exercício Prático (1h30)

```python
# synapse.py — Semana 3
import numpy as np
import matplotlib.pyplot as plt

class Synapse:
    """
    Sinapse química entre dois neurônios.

    Modela a condutância sináptica com decaimento exponencial.
    E_rev (potencial de reversão) determina se é excitatório ou inibitório:
      - Glutamato (excitatório): E_rev = 0 mV
      - GABA (inibitório): E_rev = -80 mV
    A corrente flui de E_rev em direção a V_post, portanto:
      - Se V_post < E_rev: corrente positiva (excita)
      - Se V_post > E_rev: corrente negativa (inibe)
    """
    def __init__(self, pre_neuron, post_neuron, weight=1.0, syn_type='excitatory'):
        self.pre = pre_neuron
        self.post = post_neuron
        self.weight = weight
        self.syn_type = syn_type

        # Constante de tempo sináptica (ms)
        self.tau_syn = 5.0
        # Condutância atual
        self.g = 0.0
        # Potencial de reversão
        self.E_rev = 0.0 if syn_type == 'excitatory' else -80.0

    def update(self, pre_fired, dt=0.1):
        """Atualiza a condutância sináptica"""
        if pre_fired:
            self.g += self.weight    # spike eleva a condutância
        # Decaimento exponencial
        self.g -= (self.g / self.tau_syn) * dt
        self.g = max(0.0, self.g)

    def get_current(self, V_post):
        """Corrente sináptica entregue ao pós-sináptico"""
        return self.g * (self.E_rev - V_post)


# Experimento: circuito 3 neurônios E → I → E
def experiment_ei_circuit():
    from if_neuron import IFNeuron  # importa da semana anterior

    dt = 0.1
    T = 500.0
    times = np.arange(0, T, dt)

    # Neurônios
    N1 = IFNeuron('N1_excitatory')   # recebe corrente externa
    N2 = IFNeuron('N2_inhibitory')   # interneurônio inibitório
    N3 = IFNeuron('N3_target')       # alvo que será suprimido

    # Sinapses
    syn_N1_N2 = Synapse(N1, N2, weight=3.0, syn_type='excitatory')
    syn_N2_N3 = Synapse(N2, N3, weight=4.0, syn_type='inhibitory')

    V1, V2, V3 = [], [], []
    I_external = 20.0  # corrente externa apenas em N1

    for t in times:
        # Calcular correntes sinápticas
        I_to_N2 = syn_N1_N2.get_current(N2.V)
        I_to_N3 = syn_N2_N3.get_current(N3.V)

        # Simular neurônios
        fired_N1 = N1.step(I_external, t, dt)
        fired_N2 = N2.step(I_to_N2, t, dt)
        fired_N3 = N3.step(I_to_N3, t, dt)

        # Atualizar sinapses
        syn_N1_N2.update(fired_N1, dt)
        syn_N2_N3.update(fired_N2, dt)

        V1.append(N1.V)
        V2.append(N2.V)
        V3.append(N3.V)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, V, label, color in zip(axes,
                                    [V1, V2, V3],
                                    ['N1 (excit. externo)', 'N2 (interneurônio inibitório)', 'N3 (alvo suprimido)'],
                                    ['blue', 'orange', 'green']):
        ax.plot(times, V, color=color, linewidth=0.7)
        ax.axhline(-55, color='red', linestyle='--', alpha=0.5, label='threshold')
        ax.set_ylabel('V (mV)')
        ax.set_title(label)

    axes[-1].set_xlabel('Tempo (ms)')
    plt.tight_layout()
    plt.savefig('semana3_ei_circuit.png')
    print("Circuito E→I→E salvo.")
    print(f"N1 disparou {len(N1.spike_times)} vezes")
    print(f"N2 disparou {len(N2.spike_times)} vezes")
    print(f"N3 disparou {len(N3.spike_times)} vezes")


if __name__ == '__main__':
    experiment_ei_circuit()
```

### ✅ Como saber se absorveu

- [ ] Você observa que N3 dispara pouco ou nada quando N2 está ativo (inibição feedforward)
- [ ] Você entende por que `E_rev = -80mV` para GABA inibe mesmo quando o neurônio já está em -70mV
- [ ] Se você remover a sinapse N2→N3 e dar corrente a N3 diretamente, ele dispara livremente — demonstre isso
- [ ] Você consegue responder: "O que acontece com uma rede se você aumentar o peso de todas as sinapses excitatórias?"

---

## SEMANA 4
**Tema: Redes Pequenas e Balanço E/I**

### 📚 O que estudar (2h)

- **Balanço E/I:** neocórtex tem ~80% neurônios excitatórios e ~20% inibitórios — essa proporção é crítica
- **Estado Assíncrono-Irregular (AI state):** o modo de operação "normal" do córtex — neurônios disparam de forma não-sincronizada, à taxas baixas. Computacionalmente riquíssimo.
- **Estado Síncrono:** todos disparam juntos — perda de informação, computacionalmente inútil (epilepsia)
- **Recorrência:** neurônios se reconectam de volta ao conjunto — cria dinâmica complexa e memória de curto prazo
- **Conectividade esparsa:** cada neurônio cortical conecta com ~0.1-1% dos outros neurônios na vizinhança

**A distinção crucial:**
No neocórtex, GABA não apenas "para" a atividade — ele **esculpe** precisamente quais neurônios estão ativos. A inibição cria padrões, não silêncio.

**Recursos gratuitos:**
- Wikipedia: "Balanced network theory" (sim, Wikipedia é suficiente aqui)
- YouTube: busque "E/I balance cortex Vogels" — há uma palestra técnica boa
- Scholarpedia: "Conductance-based models" (leia só a introdução, ~1 página)

### ⚗️ Exercício Prático (1h30)

```python
# network.py — Semana 4
import numpy as np
import matplotlib.pyplot as plt

# Assume que IFNeuron e Synapse estão importados dos arquivos anteriores

class SmallCorticalNetwork:
    """
    Rede cortical pequena com balanço E/I.
    Proporção biológica: 80% excitatório, 20% inibitório.
    """
    def __init__(self, n_excitatory=80, n_inhibitory=20, p_connect=0.15):
        self.neurons_E = [IFNeuron(f'E{i}') for i in range(n_excitatory)]
        self.neurons_I = [IFNeuron(f'I{i}') for i in range(n_inhibitory)]
        self.all_neurons = self.neurons_E + self.neurons_I
        self.synapses = []
        self._build_connectivity(p_connect)

    def _build_connectivity(self, p_connect):
        """Conecta neurônios aleatoriamente com probabilidade p_connect"""
        for pre in self.all_neurons:
            for post in self.all_neurons:
                if pre is post:
                    continue
                if np.random.random() < p_connect:
                    is_exc = pre.id.startswith('E')
                    syn_type = 'excitatory' if is_exc else 'inhibitory'
                    # Sinapses inibitórias são mais fortes para manter equilíbrio
                    weight = 1.5 if is_exc else 4.0
                    self.synapses.append(
                        Synapse(pre, post, weight=weight, syn_type=syn_type)
                    )

    def simulate(self, T=1000.0, dt=0.1, I_background=10.0, noise_std=2.0):
        """
        Simula a rede.
        Retorna: spike_trains — dict {neuron_id: [spike_times]}
        """
        times = np.arange(0, T, dt)
        spike_trains = {n.id: [] for n in self.all_neurons}

        for t in times:
            # Acumular correntes sinápticas para cada neurônio
            currents = {n.id: 0.0 for n in self.all_neurons}
            fired_last = {n.id: n.is_firing for n in self.all_neurons}

            for syn in self.synapses:
                syn.update(fired_last[syn.pre.id], dt)
                currents[syn.post.id] += syn.get_current(syn.post.V)

            # Simular cada neurônio
            for n in self.all_neurons:
                I_total = currents[n.id] + I_background + np.random.randn() * noise_std
                fired = n.step(I_total, t, dt)
                if fired:
                    spike_trains[n.id].append(t)

        return spike_trains

    def raster_plot(self, spike_trains, T=1000.0):
        """Raster plot: X=tempo, Y=índice do neurônio, pontos=spikes"""
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, neuron in enumerate(self.all_neurons):
            spikes = spike_trains.get(neuron.id, [])
            if spikes:
                ax.scatter(spikes, [i] * len(spikes),
                          s=1.5,
                          color='blue' if neuron.id.startswith('E') else 'red',
                          alpha=0.7)

        ax.axhline(80, color='gray', linestyle='--', alpha=0.5, label='E/I boundary')
        ax.set_xlabel('Tempo (ms)')
        ax.set_ylabel('Índice do Neurônio (azul=E, vermelho=I)')
        ax.set_title('Raster Plot — Rede Cortical com Balanço E/I')
        ax.set_xlim(0, T)
        ax.legend()
        plt.tight_layout()
        plt.savefig('semana4_raster.png')

        # Calcular taxa média de disparo
        total_spikes = sum(len(v) for v in spike_trains.values())
        n_neurons = len(self.all_neurons)
        mean_rate = (total_spikes / n_neurons) / (T / 1000.0)
        print(f"Taxa média: {mean_rate:.1f} Hz")
        print(f"Total de spikes: {total_spikes}")


if __name__ == '__main__':
    np.random.seed(42)
    net = SmallCorticalNetwork(n_excitatory=80, n_inhibitory=20, p_connect=0.15)
    spikes = net.simulate(T=500.0, dt=0.1)
    net.raster_plot(spikes, T=500.0)
    print("Raster salvo: semana4_raster.png")
```

### ✅ Como saber se absorveu

- [ ] Você tem um raster plot funcionando — a atividade não deve ser toda sincronizada (cada neurônio dispara em tempos diferentes)
- [ ] Você testou o que acontece quando você remove todos os neurônios inibitórios: a rede deve explodir ou ficar máxima
- [ ] Você entende por que taxa média de ~5-30 Hz no raster é saudável, e >100 Hz sincronizado é patológico
- [ ] Você consegue explicar o balanço 80/20 E/I como parâmetro de design do seu projeto

---
---

# MÊS 2 — PLASTICIDADE SINÁPTICA

---

## SEMANA 5
**Tema: Aprendizado Hebbiano**
*"Neurons that fire together, wire together"*

### 📚 O que estudar (2h)

- **Regra de Hebb (1949):** quando neurônio A ativa B consistentemente, a sinapse A→B se fortalece
- **LTP (Long-Term Potentiation):** fortalecimento sináptico duradouro — base molecular da memória
- **LTD (Long-Term Depression):** enfraquecimento sináptico duradouro
- **BCM (Bienenstock-Cooper-Munro):** versão mais robusta da regra de Hebb com normalização
- **O problema da explosão hebbiana:** sem limite, pesos crescem indefinidamente — precisa de normalização

**Fórmula base:**
```
Δw = η × pre_activity × post_activity
```
Onde η é a taxa de aprendizado (tipicamente muito pequeno: 0.001 a 0.01).

**Conexão com o seu projeto:**
O aprendizado Hebbiano é o mecanismo primário de *armazenamento* no seu hipocampo e neocórtex. Sem isso, suas redes não aprendem.

**Recursos gratuitos:**
- Wikipedia: "Hebbian theory" — surpreendentemente boa, inclui equações
- YouTube: busque "Hebbian learning explained" (~8-10min — há vários)
- Scholarpedia: "BCM theory" — gratuito, peer-reviewed, ~3 páginas

### ⚗️ Exercício Prático (1h30)

```python
# plastic_synapse.py — Semana 5
import numpy as np
import matplotlib.pyplot as plt

class PlasticSynapse(Synapse):
    """
    Sinapse com plasticidade Hebbiana.
    Herda de Synapse (semana 3) e adiciona regra de aprendizado.
    """
    def __init__(self, pre, post, weight=0.5, syn_type='excitatory',
                 learning_rate=0.005, w_max=3.0, w_min=0.0):
        super().__init__(pre, post, weight, syn_type)
        self.lr = learning_rate
        self.w_max = w_max
        self.w_min = w_min
        self.weight_history = [weight]

    def hebbian_update(self, pre_fired, post_fired):
        """
        Regra de Hebb com normalização multiplicativa (evita explosão).
        A normalização (w_max - weight) garante saturação natural.
        """
        if pre_fired and post_fired:
            # Potenciação (LTP) — ambos co-ativos
            dw = self.lr * (self.w_max - self.weight)  # satura em w_max
            self.weight += dw
        elif pre_fired and not post_fired:
            # Depressão (LTD) — pre ativo, post inativo
            dw = self.lr * 0.2 * self.weight
            self.weight -= dw

        # Clamp nos limites
        self.weight = np.clip(self.weight, self.w_min, self.w_max)
        self.weight_history.append(self.weight)


# Experimento: como o peso evolui com co-ativação consistente
def experiment_hebbian_learning():
    from if_neuron import IFNeuron

    N_pre = IFNeuron('pre')
    N_post = IFNeuron('post')
    syn = PlasticSynapse(N_pre, N_post, weight=0.1)

    dt = 0.1
    T = 300.0
    times = np.arange(0, T, dt)

    # Fase 1 (0-150ms): pares co-ativos — apresentar sinal pareado
    # Fase 2 (150-300ms): sem co-ativação — só pre dispara
    for t in times:
        if t < 150.0:
            I_pre = 20.0     # pre dispara
            I_post = 20.0    # post dispara junto
        else:
            I_pre = 20.0     # só pre dispara
            I_post = 0.0     # post silencioso

        fired_pre = N_pre.step(I_pre, t, dt)
        fired_post = N_post.step(I_post, t, dt)
        syn.hebbian_update(fired_pre, fired_post)

    plt.figure(figsize=(10, 5))
    plt.plot(syn.weight_history, 'b-', linewidth=1.5)
    plt.axvline(int(150/dt), color='red', linestyle='--', label='Fim da co-ativação')
    plt.xlabel('Passo de Tempo')
    plt.ylabel('Peso Sináptico')
    plt.title('Evolução do Peso Hebbiano: LTP → LTD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('semana5_hebbian.png')
    print(f"Peso final: {syn.weight:.3f}")
    print(f"Peso máximo atingido: {max(syn.weight_history):.3f}")


if __name__ == '__main__':
    experiment_hebbian_learning()
```

### ✅ Como saber se absorveu

- [ ] Você vê o peso subindo durante co-ativação e caindo quando só o pre está ativo
- [ ] Você entende por que `(w_max - weight)` na fórmula evita a explosão hebbiana
- [ ] Você consegue explicar: "O que aconteceria se você tivesse só LTP sem LTD?"
- [ ] Você sabe o que é o critério BCM: neurônios muito ativos deprimem facilmente, neurônios pouco ativos potenciam facilmente — isso cria homeostase

---

## SEMANA 6
**Tema: STDP — Spike-Timing Dependent Plasticity**

### 📚 O que estudar (2h)

O STDP é a versão refinada e temporal do Hebb. A diferença é o **timing**:

- Se pre dispara **antes** de post (Δt = t_post - t_pre > 0): LTP — causalidade confirmada
- Se pre dispara **depois** de post (Δt < 0): LTD — causalidade invertida, sinapse enfraquece
- A janela temporal é ~20-50ms
- Isso biologicamente codifica *causalidade*: se A causa B, A→B se fortalece

**A curva STDP clássica:**
```
ΔW = A+ × exp(-Δt/τ+)   se Δt > 0 (LTP)
ΔW = -A- × exp(Δt/τ-)   se Δt < 0 (LTD)
```

**Por que isso é vital para o seu hipocampo:**
O CA3 usa STDP para aprender *sequências*. A ordem de disparo codifica a ordem dos eventos. Seu hipocampo precisa disso para memória episódica (evento A foi antes de B).

**Recursos gratuitos:**
- Paper original: Bi & Poo (1998) "Synaptic Modifications in Cultured Hippocampal Neurons" — busque no Google Scholar, 2 páginas de resultado, gratuito
- Scholarpedia: "Spike-timing dependent plasticity" — a referência definitiva e gratuita
- YouTube: "STDP animation" — há várias visualizações boas (~5min)

### ⚗️ Exercício Prático (1h30)

```python
# stdp_synapse.py — Semana 6
import numpy as np
import matplotlib.pyplot as plt

class STDPSynapse(PlasticSynapse):
    """
    Sinapse com STDP (Spike-Timing Dependent Plasticity).
    A assimetria temporal é o mecanismo de aprendizado de sequências.
    """
    def __init__(self, pre, post, **kwargs):
        super().__init__(pre, post, **kwargs)
        self.A_plus = 0.005    # amplitude LTP
        self.A_minus = 0.0055  # amplitude LTD (ligeiramente maior — estabilidade)
        self.tau_plus = 20.0   # ms — janela temporal LTP
        self.tau_minus = 20.0  # ms — janela temporal LTD

    def stdp_update(self, t):
        """
        Atualiza peso baseado no timing dos últimos spikes.
        Chamado apenas quando há spike pré ou pós-sináptico.
        """
        if not self.pre.spike_times or not self.post.spike_times:
            return

        t_pre = self.pre.spike_times[-1]
        t_post = self.post.spike_times[-1]
        delta_t = t_post - t_pre  # positivo: pre antes de post

        if abs(delta_t) > 100.0:  # fora da janela de plasticidade
            return

        if delta_t > 0:
            # Pre disparou antes de post → LTP (causalidade)
            dw = self.A_plus * np.exp(-delta_t / self.tau_plus)
        else:
            # Post disparou antes de pre → LTD (anticausalidade)
            dw = -self.A_minus * np.exp(delta_t / self.tau_minus)

        self.weight += dw
        self.weight = np.clip(self.weight, self.w_min, self.w_max)
        self.weight_history.append(self.weight)


# Experimento: Reproduzir a curva clássica de Bi & Poo (1998)
def experiment_stdp_curve():
    """
    Para cada Δt entre -100ms e +100ms,
    mede a mudança média no peso após 30 pares de spikes.
    """
    from if_neuron import IFNeuron

    delta_ts = np.linspace(-100, 100, 41)
    weight_changes = []

    for delta_t in delta_ts:
        N_pre = IFNeuron('pre')
        N_post = IFNeuron('post')
        syn = STDPSynapse(N_pre, N_post, weight=1.0, w_max=3.0, w_min=0.0)

        # Forçar pares de spikes com Δt controlado
        for _ in range(30):
            t_pre_spike = 100.0
            t_post_spike = t_pre_spike + delta_t

            N_pre.spike_times = [t_pre_spike]
            N_post.spike_times = [t_post_spike]
            syn.stdp_update(t_pre_spike)

        weight_changes.append(syn.weight - 1.0)  # mudança a partir do peso inicial

    plt.figure(figsize=(10, 6))
    plt.plot(delta_ts, weight_changes, 'b-o', markersize=4, linewidth=1.5)
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.fill_between(delta_ts,
                     [max(0, dw) for dw in weight_changes],
                     0, alpha=0.3, color='green', label='LTP (Δt>0)')
    plt.fill_between(delta_ts,
                     [min(0, dw) for dw in weight_changes],
                     0, alpha=0.3, color='red', label='LTD (Δt<0)')
    plt.xlabel('Δt = t_post - t_pre (ms)')
    plt.ylabel('Mudança no Peso (ΔW)')
    plt.title('Curva STDP — Reprodução de Bi & Poo (1998)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('semana6_stdp_curve.png')
    print("Curva STDP salva: semana6_stdp_curve.png")


if __name__ == '__main__':
    experiment_stdp_curve()
```

### ✅ Como saber se absorveu

- [ ] Sua curva STDP tem o formato clássico: positivo para Δt>0, negativo para Δt<0
- [ ] LTD é ligeiramente mais forte que LTP (|A-| > |A+|) — você entende por quê? (estabilidade homeostática)
- [ ] Você consegue explicar a diferença: "Hebb aprende co-ativações; STDP aprende sequências temporais causais"
- [ ] Você consegue dizer qual das duas regras é mais adequada para o hipocampo (que codifica episódios sequenciais) — resposta: STDP

---

## SEMANA 7
**Tema: Representações Esparsas**

### 📚 O que estudar (2h)

- **Esparsidade no cérebro:** apenas 1-5% dos neurônios corticais estão ativos para qualquer representação
- **Por quê?** Esparsidade aumenta a capacidade de armazenamento drasticamente e reduz interferência entre memórias
- **SDR (Sparse Distributed Representation):** representação distribuída por uma população, mas esparsa
- **k-WTA (k-Winners Take All):** mecanismo de seleção que impõe esparsidade — os k neurônios mais ativos "ganham", o resto silencia
- **Giro Dentado:** ~2% de esparsidade — o mais esparso do hipocampo

**Capacidade de memória:**
```
Rede densa (f=0.5): capacidade ~0.14N padrões para N neurônios
Rede esparsa (f=0.02): capacidade ~5× maior que densa
```

**Recursos gratuitos:**
- Paper: Kanerva (1988) "Sparse Distributed Memory" — PDF gratuito em busca Google Scholar
- Numenta Research: "Why Neurons Have Thousands of Synapses" — paper de Hawkins, gratuito em numenta.com
- YouTube: Jeff Hawkins "Sparse Distributed Representations" — busca exata (~20min, muito bom)

### ⚗️ Exercício Prático (1h30)

```python
# sparse_layer.py — Semana 7
import numpy as np
import matplotlib.pyplot as plt

class SparseLayer:
    """
    Camada neural com enforcement de esparsidade via k-WTA.
    
    k-WTA (k-Winners Take All): dados N neurônios e ativações brutas,
    apenas os k mais ativos permanecem ativos. O resto é zerado.
    Isso simula a inibição lateral que o Giro Dentado usa.
    """
    def __init__(self, n_neurons, sparsity=0.05):
        self.n_neurons = n_neurons
        self.sparsity = sparsity
        self.k = max(1, int(n_neurons * sparsity))

    def activate(self, inputs):
        """
        Aplica k-WTA ao vetor de inputs.
        Retorna representação esparsa.
        """
        raw = np.array(inputs, dtype=float)

        # Encontrar o k-ésimo maior valor
        if self.k >= len(raw):
            return raw

        threshold_val = np.partition(raw, -self.k)[-self.k]
        sparse = np.where(raw >= threshold_val, raw, 0.0)
        return sparse

    def measure_sparsity(self, activations):
        """Fração de unidades inativas"""
        return 1.0 - (np.sum(activations > 0) / len(activations))

    def measure_overlap(self, pattern_a, pattern_b):
        """
        Overlap entre dois padrões SDR.
        Alto overlap = padrões similares.
        Baixo overlap = padrões distintos.
        """
        active_a = set(np.where(pattern_a > 0)[0])
        active_b = set(np.where(pattern_b > 0)[0])
        if not active_a or not active_b:
            return 0.0
        return len(active_a & active_b) / len(active_a | active_b)


# Experimento: capacidade de memória — esparso vs denso
def experiment_capacity_comparison():
    """
    Compara a capacidade de armazenar padrões distintos
    em uma camada esparsa vs densa.
    """
    n_neurons = 500
    n_patterns_list = list(range(5, 100, 5))

    sparse_overlaps = []
    dense_overlaps = []

    sparse_layer = SparseLayer(n_neurons=n_neurons, sparsity=0.02)
    dense_layer = SparseLayer(n_neurons=n_neurons, sparsity=0.50)

    for n_patterns in n_patterns_list:
        # Gerar padrões aleatórios
        patterns_raw = [np.random.randn(n_neurons) for _ in range(n_patterns)]

        # Aplicar esparsidade
        sparse_patterns = [sparse_layer.activate(p) for p in patterns_raw]
        dense_patterns = [dense_layer.activate(p) for p in patterns_raw]

        # Medir overlap médio entre todos os pares
        sparse_overlap_avg = []
        dense_overlap_avg = []

        for i in range(n_patterns):
            for j in range(i+1, n_patterns):
                sparse_overlap_avg.append(
                    sparse_layer.measure_overlap(sparse_patterns[i], sparse_patterns[j])
                )
                dense_overlap_avg.append(
                    dense_layer.measure_overlap(dense_patterns[i], dense_patterns[j])
                )

        sparse_overlaps.append(np.mean(sparse_overlap_avg))
        dense_overlaps.append(np.mean(dense_overlap_avg))

    plt.figure(figsize=(10, 6))
    plt.plot(n_patterns_list, sparse_overlaps, 'b-o', label=f'Esparso (f=2%)', markersize=4)
    plt.plot(n_patterns_list, dense_overlaps, 'r-o', label=f'Denso (f=50%)', markersize=4)
    plt.xlabel('Número de Padrões Armazenados')
    plt.ylabel('Overlap Médio entre Padrões')
    plt.title('Interferência entre Padrões: Esparso vs Denso')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('semana7_sparsity.png')
    print("Gráfico de esparsidade salvo.")


if __name__ == '__main__':
    experiment_capacity_comparison()
```

### ✅ Como saber se absorveu

- [ ] Você demonstrou que representações esparsas têm overlap médio muito menor do que densas
- [ ] Você sabe implementar k-WTA sem frameworks
- [ ] Você consegue explicar por que o Giro Dentado precisa ser esparso: "Porque CA3 precisa receber inputs distintos para não confundir memórias similares"
- [ ] Você consegue calcular: para 500 neurônios com f=2%, quantos neurônios estão ativos? (resposta: 10)

---

## SEMANA 8
**Tema: Introdução ao Hipocampo**
*(semana mais conceitual — base para o mês inteiro seguinte)*

### 📚 O que estudar (2h — foco em compreensão, não código)

**Os dois pilares desta semana:**

**1. O caso H.M. (Henry Molaison):**
- Cirurgia em 1953 removeu o hipocampo bilateral para tratar epilepsia
- Resultado: incapaz de formar novas memórias declarativas (amnésia anterógrada)
- Manteve memória procedural (como andar de bicicleta) e memórias antigas
- Implicação: hipocampo é essencial para *codificar* novos episódios, não para armazená-los permanentemente

**2. Teoria de Complementary Learning Systems (CLS) — McClelland et al. 1995:**

| Sistema | Córtex | Hipocampo |
|---------|--------|-----------|
| Velocidade de aprendizado | Lento (décadas) | Rápido (1 tentativa) |
| Tipo de memória | Semântica / geral | Episódica / específica |
| Mecanismo | Backprop-like gradual | Armazenagem one-shot |
| Capacidade | Grande (pesos distribuídos) | Limitada (mas indexada) |

**O conceito-chave para o seu projeto:**
Seu sistema não pode ter só o hipocampo ou só o córtex. Os dois se complementam. Hipocampo aprende rápido, armazena o índice, consolida durante o "sono" para o córtex.

**Catastrophic Forgetting:**
Se o córtex aprende diretamente de tudo sem mediação do hipocampo, ele esquece as coisas antigas quando aprende coisas novas (catastrophic forgetting). O hipocampo resolve isso via *replay* durante o sono.

**Recursos gratuitos:**
- Paper: McClelland, McNaughton & O'Reilly (1995) — "Why there are complementary learning systems in the hippocampus and neocortex" — PDF gratuito no Google Scholar (busca exata do título)
- YouTube: Robert Sapolsky "Behavioral Biology — Memory" — aula da Stanford, gratuita no YouTube (~1h, assista a primeira metade)
- YouTube: "HM patient memory" — há vários vídeos curtos (~10min) sobre o caso

**Exercício de absorção (não é código — é conceitual):**

Escreva em texto (caderno, arquivo .txt, qualquer lugar) — sem olhar as notas — as respostas para:
1. Por que o córtex não pode simplesmente aprender tudo diretamente em alta velocidade?
2. O que é catastrophic forgetting e como o hipocampo resolve isso?
3. O que "indexação" significa: o hipocampo armazena a memória ou armazena o *endereço* da memória?
4. Como você vai implementar o "replay de sono" no seu sistema Python?

### ✅ Como saber se absorveu

- [ ] Você escreveu respostas às 4 perguntas sem olhar as notas
- [ ] Você consegue explicar a teoria CLS para alguém em 3 minutos
- [ ] Você sabe a diferença entre memória episódica, semântica e procedural — e qual o hipocampo cuida (resposta: episódica e semântica na fase inicial; procedural fica no cerebelo/gânglios basais)
- [ ] Você anotou no seu projeto: onde vai entrar a consolidação por replay? O que ela vai fazer?

---
---

# ═══════════════════════════════════
# MÊS 3 — O HIPOCAMPO
# ═══════════════════════════════════

---

## SEMANA 9
**Tema: Anatomia Funcional do Hipocampo**

### 📚 O que estudar (2h)

**As 4 regiões que você vai implementar:**

| Região | Papel Computacional | Parâmetro Biológico |
|--------|---------------------|---------------------|
| Córtex Entorrinal (EC) | Interface input/output | ~10% esparsidade |
| Giro Dentado (DG) | Separação de padrões | ~2% esparsidade, ~5× maior que EC |
| CA3 | Completamento de padrões (recorrente) | ~10% esparsidade |
| CA1 | Saída / comparação / indexação | ~10% esparsidade |

**O Circuito Trisináptico:**
```
EC → DG         (via Perforant Path — fraco, muitos-para-muitos)
DG → CA3        (via Mossy Fibers — FORTE, um-para-poucos, não-plástico inicialmente)
CA3 → CA3       (Colaterais Recorrentes — plásticas, Hopfield-like)
CA3 → CA1       (Colaterais de Schaffer — plásticas)
EC → CA1        (Perforant Path direto — comparação com CA3)
CA1 → EC        (retroprojeção para saída)
```

**Por que a expansão DG é enorme?**
EC tem ~100 neurônios (no modelo), DG tem ~500+. Essa expansão é necessária para separação de padrões: mais espaço → menos sobreposição → padrões similares ficam mais distintos.

**Recursos gratuitos:**
- Wikipedia: "Hippocampal formation" — seção "Connections" — boa visão geral
- YouTube: busque "Trisynaptic circuit hippocampus" — há boas animações (~5-10min)
- Paper: O'Reilly & McClelland (1994) "Hippocampal conjunctive encoding, storage, and recall" — Google Scholar

### ⚗️ Exercício Prático (1h30)

```python
# hippocampus.py — Semana 9
import numpy as np

class HippocampalFormation:
    """
    Arquitetura do hipocampo artificial com circuito trisináptico.
    
    Tamanhos biológicamente proporcionais:
      EC (100) → DG (500) → CA3 (200) → CA1 (150)
    
    A expansão EC→DG é intencional: mais dimensões = menos overlap
    entre padrões similares = melhor separação.
    """

    def __init__(self,
                 ec_size=100,
                 dg_size=500,
                 ca3_size=200,
                 ca1_size=150):

        self.ec_size = ec_size
        self.dg_size = dg_size
        self.ca3_size = ca3_size
        self.ca1_size = ca1_size

        # Camadas com esparsidade biológica
        self.ec_layer = SparseLayer(ec_size, sparsity=0.10)
        self.dg_layer = SparseLayer(dg_size, sparsity=0.02)   # altamente esparso
        self.ca3_layer = SparseLayer(ca3_size, sparsity=0.10)
        self.ca1_layer = SparseLayer(ca1_size, sparsity=0.10)

        # Pesos sinápticos — inicialização aleatória pequena
        np.random.seed(42)
        # Perforant Path: EC → DG
        self.W_ec_dg = np.random.randn(dg_size, ec_size) * 0.1
        # Mossy Fibers: DG → CA3 (fortes, inicialmente não-plásticas)
        self.W_dg_ca3 = np.random.randn(ca3_size, dg_size) * 0.5
        # Recorrentes: CA3 → CA3
        self.W_ca3_ca3 = np.random.randn(ca3_size, ca3_size) * 0.1
        np.fill_diagonal(self.W_ca3_ca3, 0)  # sem auto-sinapses
        # Schaffer Collaterals: CA3 → CA1
        self.W_ca3_ca1 = np.random.randn(ca1_size, ca3_size) * 0.1
        # Perforant Path direto: EC → CA1 (para comparação)
        self.W_ec_ca1 = np.random.randn(ca1_size, ec_size) * 0.1

    def encode(self, input_pattern):
        """
        Processa padrão através do circuito: EC → DG → CA3 → CA1
        Retorna ativações de todas as regiões.
        """
        assert len(input_pattern) == self.ec_size, \
            f"Input deve ter {self.ec_size} dimensões, recebeu {len(input_pattern)}"

        ec_act = self.ec_layer.activate(input_pattern)
        dg_act = self.dg_layer.activate(self.W_ec_dg @ ec_act)
        ca3_act = self.ca3_layer.activate(
            self.W_dg_ca3 @ dg_act + self.W_ca3_ca3 @ np.zeros(self.ca3_size)
        )
        ca1_act = self.ca1_layer.activate(
            self.W_ca3_ca1 @ ca3_act + self.W_ec_ca1 @ ec_act
        )

        return {
            'ec': ec_act,
            'dg': dg_act,
            'ca3': ca3_act,
            'ca1': ca1_act
        }

    def measure_region_sparsity(self, activations):
        """Mede a esparsidade real de cada região"""
        return {
            region: self.ec_layer.measure_sparsity(act)
            for region, act in activations.items()
        }


# Teste: verificar esparsidade por região
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    hippo = HippocampalFormation()

    # Passar 20 padrões aleatórios
    sparsities = {r: [] for r in ['ec', 'dg', 'ca3', 'ca1']}

    for _ in range(20):
        pattern = np.random.randn(100)
        acts = hippo.encode(pattern)
        sp = hippo.measure_region_sparsity(acts)
        for region, val in sp.items():
            sparsities[region].append(val)

    print("Esparsidade média por região:")
    for region, vals in sparsities.items():
        print(f"  {region.upper()}: {np.mean(vals)*100:.1f}% inativo "
              f"({(1-np.mean(vals))*100:.1f}% ativo)")

    # Visualizar: overlap entre padrões similares vs dissimilares
    p1 = np.random.randn(100)
    p2 = p1 + np.random.randn(100) * 0.1  # muito similar a p1
    p3 = np.random.randn(100)              # não relacionado

    acts1 = hippo.encode(p1)
    acts2 = hippo.encode(p2)
    acts3 = hippo.encode(p3)

    sparse_layer = SparseLayer(100)
    print("\nOverlap nos padrões de entrada (EC):")
    print(f"  Similar (p1 vs p2): {sparse_layer.measure_overlap(acts1['ec'], acts2['ec']):.3f}")
    print(f"  Diferente (p1 vs p3): {sparse_layer.measure_overlap(acts1['ec'], acts3['ec']):.3f}")

    print("\nOverlap após DG (separação de padrões):")
    print(f"  Similar (p1 vs p2): {sparse_layer.measure_overlap(acts1['dg'], acts2['dg']):.3f}")
    print(f"  Diferente (p1 vs p3): {sparse_layer.measure_overlap(acts1['dg'], acts3['dg']):.3f}")
    print("\n→ DG reduz o overlap de padrões similares (separação)")
```

### ✅ Como saber se absorveu

- [ ] DG tem esparsidade próxima de 2% (~98% dos neurônios inativos), EC próxima de 10%
- [ ] Você demonstrou que DG reduz o overlap entre padrões similares (separação de padrões funcionando)
- [ ] Você consegue explicar por que DG tem 5× mais neurônios que EC: expansão para criar representações mais distintas no espaço de alta dimensão
- [ ] O código instancia sem erros com os tamanhos biológicos

---

## SEMANA 10
**Tema: Células de Lugar e Codificação Distribuída**

### 📚 O que estudar (2h)

- **Place cells:** neurônios em CA1/CA3 que disparam quando o animal está em um local específico (O'Keefe, 1971)
- **Campo de lugar:** região do espaço onde a célula é ativa — forma gaussiana
- **Representação populacional:** a posição é codificada pela *população* de células ativas, não por uma célula única
- **Grid cells:** no córtex entorrinal, padrão hexagonal que computa posição relativa (Moser, Nobel 2014)
- **Contexto além do espaço:** o hipocampo usa a mesma arquitetura para representar episódios, conceitos, contextos — não só localização física

**O modelo computacional:**
```
Ativação_i(x) = exp( -||x - x_i||² / (2σ²) )
```
Campo gaussiano centrado na localização preferida x_i com largura σ.

**Recursos gratuitos:**
- Nobel Lecture de O'Keefe: nobelpize.org (texto gratuito)
- Paper original: O'Keefe & Dostrovsky (1971) — apenas 2 páginas, leia no PubMed (gratuito)
- YouTube: "Place cells grid cells explained" — há excelentes visualizações (~10min)

### ⚗️ Exercício Prático (1h30)

```python
# place_cells.py — Semana 10
import numpy as np
import matplotlib.pyplot as plt

class PlaceCellLayer:
    """
    Camada de células de lugar para um ambiente 2D.
    
    Cada célula tem uma localização preferida no espaço.
    Ativa-se com função gaussiana quando o agente está próximo.
    
    IMPORTANTE: O princípio não é só espacial.
    A mesma arquitetura representa contextos, episódios, conceitos.
    "Lugar" é uma metáfora — o hipocampo generaliza isso.
    """

    def __init__(self, n_cells=100, env_size=(10.0, 10.0), sigma=1.0):
        self.n_cells = n_cells
        self.env_size = env_size
        self.sigma = sigma  # largura do campo de lugar

        # Distribuir centros aleatoriamente no ambiente
        np.random.seed(42)
        self.preferred_locs = np.column_stack([
            np.random.uniform(0, env_size[0], n_cells),
            np.random.uniform(0, env_size[1], n_cells)
        ])

    def activate(self, position):
        """
        Retorna a ativação de todas as células para a posição dada.
        """
        pos = np.array(position)
        distances = np.linalg.norm(self.preferred_locs - pos, axis=1)
        return np.exp(-distances**2 / (2 * self.sigma**2))

    def decode_position(self, activations):
        """
        Decodifica a posição a partir das ativações (population vector decoding).
        Média ponderada pelas ativações.
        """
        total_activation = np.sum(activations)
        if total_activation < 1e-10:
            return np.array([0.0, 0.0])
        weights = activations / total_activation
        decoded = np.sum(self.preferred_locs * weights[:, np.newaxis], axis=0)
        return decoded

    def visualize_field(self, cell_idx, resolution=50):
        """Plota o campo de lugar de uma célula específica"""
        x = np.linspace(0, self.env_size[0], resolution)
        y = np.linspace(0, self.env_size[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for xi in range(resolution):
            for yi in range(resolution):
                act = self.activate([x[xi], y[yi]])
                Z[yi, xi] = act[cell_idx]

        return X, Y, Z


# Experimento: agente em trajetória circular
def experiment_trajectory():
    layer = PlaceCellLayer(n_cells=100, env_size=(10, 10), sigma=1.5)

    # Trajetória circular
    T = 200
    angles = np.linspace(0, 2 * np.pi, T)
    trajectory = np.column_stack([
        5 + 3 * np.cos(angles),
        5 + 3 * np.sin(angles)
    ])

    # Coletar ativações ao longo da trajetória
    all_activations = [layer.activate(pos) for pos in trajectory]
    all_activations = np.array(all_activations)  # (T, n_cells)

    # Decodificar posição
    decoded_positions = np.array([
        layer.decode_position(act) for act in all_activations
    ])

    # Visualização
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Trajetória original vs decodificada
    axes[0].plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Real', linewidth=2)
    axes[0].plot(decoded_positions[:, 0], decoded_positions[:, 1],
                 'r--', label='Decodificada', linewidth=1.5, alpha=0.8)
    axes[0].set_title('Trajetória: Real vs Decodificada')
    axes[0].legend()
    axes[0].set_aspect('equal')

    # 2. Atividade de 8 células ao longo do tempo
    cells_to_plot = np.random.choice(100, 8, replace=False)
    for ci in cells_to_plot:
        axes[1].plot(all_activations[:, ci], alpha=0.7, linewidth=1)
    axes[1].set_xlabel('Passo de Tempo')
    axes[1].set_ylabel('Ativação')
    axes[1].set_title('Ativação de 8 Células de Lugar')

    # 3. Mapa de atividade (heat map)
    activity_map = np.zeros((50, 50))
    x_bins = np.linspace(0, 10, 50)
    y_bins = np.linspace(0, 10, 50)
    for t in range(T):
        xi = np.argmin(np.abs(x_bins - trajectory[t, 0]))
        yi = np.argmin(np.abs(y_bins - trajectory[t, 1]))
        activity_map[yi, xi] = np.mean(all_activations[t])

    axes[2].imshow(activity_map, origin='lower', cmap='hot',
                   extent=[0, 10, 0, 10])
    axes[2].set_title('Mapa de Atividade')

    plt.tight_layout()
    plt.savefig('semana10_place_cells.png')
    print("Visualização de células de lugar salva.")


if __name__ == '__main__':
    experiment_trajectory()
```

### ✅ Como saber se absorveu

- [ ] Você tem um mapa de campos de lugar visualmente correto (padrão gaussiano)
- [ ] O decodificador de posição funciona: posição decodificada ≈ posição real
- [ ] Posições próximas têm overlap alto, posições distantes têm overlap baixo (verifique com `measure_overlap`)
- [ ] Você consegue explicar como esse princípio se aplica à codificação de *conceitos*: conceitos similares têm representações com mais overlap, conceitos distintos têm menos

---

## SEMANA 11
**Tema: Completamento e Separação de Padrões**

### 📚 O que estudar (2h)

**Dois processos opostos no hipocampo:**

**Separação de Padrões (Pattern Separation) — Giro Dentado:**
- Torna representações de inputs similares mais distintas
- Necessário para não confundir memórias parecidas
- Alta esparsidade + expansão dimensional = orthogonalização

**Completamento de Padrões (Pattern Completion) — CA3:**
- Recupera memória completa a partir de pista parcial (cue)
- CA3 tem muitas sinapses recorrentes — funciona como rede de Hopfield
- Você vê 30% de uma memória e lembra o todo

**A rede de Hopfield como modelo de CA3:**
```
Armazenamento:  W += (1/N) × p × pᵀ    (para cada padrão p)
Recuperação:    state[i] = sign(W[i,:] · state)    (iterado até convergir)
```
- Capacidade: ~0.14 × N padrões para N neurônios
- Os padrões armazenados são *atratores* — o sistema converge para o mais próximo

**Recursos gratuitos:**
- Paper: Hopfield (1982) — 4 páginas, altamente legível — Google Scholar, PDF gratuito
- YouTube: "Hopfield Networks explained" — versões de 2020+ são mais acessíveis
- Scholarpedia: "Hopfield network" — gratuito, peer-reviewed

### ⚗️ Exercício Prático (1h30)

```python
# hopfield_ca3.py — Semana 11
import numpy as np
import matplotlib.pyplot as plt

class HopfieldCA3:
    """
    Rede de Hopfield como modelo de CA3.
    
    CA3 é uma rede recorrente: cada neurônio conecta com ~4% dos outros.
    Isso cria uma dinâmica de atratores: inputs parciais convergem
    para o padrão armazenado mais próximo (completamento de padrão).
    
    Padrões em {-1, +1}: biologicamente, -1 = inativo, +1 = ativo.
    """

    def __init__(self, n_neurons=200):
        self.N = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))
        self.stored_patterns = []

    def store(self, pattern):
        """Armazena padrão via regra de Hebb"""
        p = np.array(pattern, dtype=float)
        assert len(p) == self.N
        assert set(np.unique(p)).issubset({-1.0, 1.0}), "Padrão deve ser {-1, +1}"

        # Regra de Hebb: ΔW = p × pᵀ / N
        self.W += np.outer(p, p) / self.N
        np.fill_diagonal(self.W, 0)  # sem auto-sinapses
        self.stored_patterns.append(p.copy())

    def retrieve(self, cue, n_steps=30):
        """
        Recupera padrão armazenado a partir de pista parcial.
        Retorna: estado final, histórico de overlap
        """
        state = np.array(cue, dtype=float)
        overlap_history = []

        for step in range(n_steps):
            # Atualização assíncrona (um neurônio por vez — mais estável)
            order = np.random.permutation(self.N)
            for i in order:
                h = np.dot(self.W[i], state)
                state[i] = 1.0 if h >= 0 else -1.0

            # Medir overlap com padrões armazenados
            overlaps = [self.overlap(state, p) for p in self.stored_patterns]
            overlap_history.append(max(overlaps) if overlaps else 0.0)

        return state, overlap_history

    def overlap(self, state_a, state_b):
        """Overlap normalizado entre dois estados [-1, +1]"""
        return np.dot(state_a, state_b) / self.N

    def corrupt_pattern(self, pattern, corruption_rate):
        """Corrompe uma fração do padrão (inverte bits)"""
        corrupted = pattern.copy()
        n_corrupt = int(len(pattern) * corruption_rate)
        idx = np.random.choice(len(pattern), n_corrupt, replace=False)
        corrupted[idx] *= -1
        return corrupted


# Experimento principal: curva de recuperação vs corrupção
def experiment_pattern_completion():
    np.random.seed(42)
    N = 200
    net = HopfieldCA3(n_neurons=N)

    # Armazenar 15 padrões aleatórios
    n_patterns = 15
    patterns = [np.random.choice([-1, 1], size=N) for _ in range(n_patterns)]
    for p in patterns:
        net.store(p)

    # Testar recuperação com diferentes graus de corrupção
    corruption_rates = np.arange(0, 0.6, 0.05)
    recovery_rates = []

    n_trials = 30
    for cr in corruption_rates:
        successes = 0
        for trial in range(n_trials):
            # Escolher padrão aleatório e corrompê-lo
            p_idx = np.random.randint(n_patterns)
            original = patterns[p_idx]
            cue = net.corrupt_pattern(original, cr)

            # Recuperar
            retrieved, _ = net.retrieve(cue)

            # Verificar se recuperou o padrão original (overlap > 0.9)
            ov = net.overlap(retrieved, original)
            if ov > 0.9:
                successes += 1

        recovery_rates.append(successes / n_trials)

    plt.figure(figsize=(10, 6))
    plt.plot(corruption_rates * 100, [r * 100 for r in recovery_rates],
             'b-o', markersize=6, linewidth=2)
    plt.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% recuperação')
    plt.xlabel('Corrupção do Cue (%)')
    plt.ylabel('Taxa de Recuperação Correta (%)')
    plt.title(f'Completamento de Padrões no CA3 — {n_patterns} padrões em {N} neurônios')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('semana11_pattern_completion.png')

    # Ponto de degradação
    for i, (cr, rr) in enumerate(zip(corruption_rates, recovery_rates)):
        if rr < 0.5:
            print(f"Recuperação cai abaixo de 50% com >{cr*100:.0f}% de corrupção")
            break


if __name__ == '__main__':
    experiment_pattern_completion()
```

### ✅ Como saber se absorveu

- [ ] Sua rede Hopfield recupera padrões com até ~20-30% de corrupção com alta taxa
- [ ] Você observou "atratores espúrios" (recuperação de um padrão que não foi armazenado) — isso é esperado em redes de Hopfield
- [ ] Você consegue conectar DG→CA3: o DG torna os inputs menos sobrepostos → CA3 tem menos ambiguidade → melhor completamento
- [ ] Você consegue explicar o tradeoff: mais padrões armazenados = mais interferência = pior recuperação

---

## SEMANA 12
**Tema: Teoria de Indexação Hipocampal**

### 📚 O que estudar (2h)

**O modelo de Teyler & DiScenna (1986):**
- Hipocampo NÃO armazena a memória em si — armazena o *índice* (ponteiro)
- Cada episódio = ativação de regiões corticais distribuídas (visual, auditiva, emocional, etc.)
- Hipocampo registra quais regiões corticais estavam co-ativas (o índice)
- Recordação = hipocampo reativa os índices → córtex reconstrói a memória distribuída

**O sistema completo:**
```
ENCODING:
  Experiência → Córtex ativa representações distribuídas
              → EC codifica o padrão cortical
              → EC → DG → CA3 → CA1 cria índice compacto
              → CA1 armazena índice e aprende mapeamento CA1→EC→Córtex

RECALL:
  Cue parcial → Córtex ativa representação parcial
              → EC → CA1 (input ruidoso)
              → CA3 completa o índice (pattern completion)
              → CA1 → EC → Córtex reativa a memória completa
```

**Recursos gratuitos:**
- Paper: Teyler & DiScenna (1986) "The hippocampal memory indexing theory" — Google Scholar
- Paper: Teyler & Rudy (2007) atualização da teoria — gratuito
- YouTube: "Hippocampal indexing theory" — há aulas de neurociência computacional

### ⚗️ Exercício Prático (1h30)

```python
# indexing_system.py — Semana 12
import numpy as np
import matplotlib.pyplot as plt

class HippocampalIndexSystem:
    """
    Sistema de indexação hipocampal.
    
    O hipocampo não armazena a memória — armazena o ENDEREÇO.
    O córtex armazena o conteúdo.
    
    Analogia de programação:
      hipocampo = tabela de hash (chave → ponteiro)
      córtex = heap (onde o dado real fica)
    """

    def __init__(self,
                 cortex_size=1000,
                 ec_size=100,
                 dg_size=500,
                 ca3_size=200,
                 ca1_size=150):

        self.cortex_size = cortex_size
        self.hippocampus = HippocampalFormation(ec_size, dg_size, ca3_size, ca1_size)
        self.ca3 = HopfieldCA3(n_neurons=ca1_size)

        # Mapeamentos aprendidos
        # Córtex → EC (compressão)
        self.W_ctx_ec = np.random.randn(ec_size, cortex_size) * 0.01
        # CA1 → Córtex (reconstrução)
        self.W_ca1_ctx = np.random.randn(cortex_size, ca1_size) * 0.01

        # Memória episódica
        self.episodes = {}
        self.n_stored = 0

    def store_episode(self, cortical_pattern, learning_rate=0.1):
        """
        Armazena um episódio cortical no índice hipocampal.
        
        1. Córtex → EC: comprime o padrão cortical
        2. EC → DG → CA3 → CA1: cria índice esparso
        3. CA3 Hopfield armazena o índice CA1
        4. Aprende mapeamento CA1 → Córtex para reconstrução
        """
        cortical = np.array(cortical_pattern)

        # Passo 1: Comprime córtex em EC
        ec_input = self.W_ctx_ec @ cortical
        ec_sparse = self.hippocampus.ec_layer.activate(ec_input)

        # Passo 2: Circuito completo
        activations = self.hippocampus.encode(ec_sparse)
        ca1_act = activations['ca1']

        # Passo 3: Armazena índice no CA3 Hopfield
        # Binariza CA1 para {-1, +1}
        ca1_binary = np.where(ca1_act > 0, 1.0, -1.0)
        self.ca3.store(ca1_binary)

        # Passo 4: Aprende mapeamento CA1 → Córtex (Hebbiano)
        self.W_ca1_ctx += learning_rate * np.outer(cortical, ca1_act)

        # Guarda para avaliação
        self.episodes[self.n_stored] = {
            'cortical': cortical.copy(),
            'ca1_index': ca1_act.copy()
        }
        self.n_stored += 1
        return ca1_act

    def recall_episode(self, partial_cortical_cue):
        """
        Recupera episódio a partir de pista cortical parcial.
        """
        cue = np.array(partial_cortical_cue)

        # Cue → EC (compressão, pode ser ruidosa)
        ec_from_cue = self.hippocampus.ec_layer.activate(self.W_ctx_ec @ cue)

        # EC → CA1 (caminho direto, ruidoso)
        ca1_noisy_input = self.hippocampus.ca1_layer.activate(
            self.hippocampus.W_ec_ca1 @ ec_from_cue
        )

        # CA3 completa o índice (pattern completion)
        ca1_binary_cue = np.where(ca1_noisy_input > 0, 1.0, -1.0)
        ca1_completed, _ = self.ca3.retrieve(ca1_binary_cue, n_steps=20)
        ca1_completed = (ca1_completed + 1) / 2  # volta para [0,1]

        # CA1 → Córtex (reconstrução)
        cortical_recalled = self.W_ca1_ctx @ ca1_completed

        return cortical_recalled

    def evaluate(self, noise_level=0.5):
        """Avalia a qualidade do recall para todos os episódios armazenados"""
        overlaps = []
        for ep_id, ep_data in self.episodes.items():
            original = ep_data['cortical']

            # Gerar cue parcial com ruído
            noise = np.random.randn(len(original)) * np.std(original) * noise_level
            cue = original * (1 - noise_level) + noise

            recalled = self.recall_episode(cue)

            # Overlap (correlação de Pearson)
            overlap = np.corrcoef(original, recalled)[0, 1]
            overlaps.append(overlap)

        return np.mean(overlaps), np.std(overlaps)


# Teste completo do sistema de indexação
if __name__ == '__main__':
    np.random.seed(42)
    system = HippocampalIndexSystem()

    print("Armazenando 20 episódios...")
    for i in range(20):
        cortical = np.random.randn(1000)
        system.store_episode(cortical)

    print("Avaliando recall com 50% de ruído...")
    mean_overlap, std_overlap = system.evaluate(noise_level=0.5)
    print(f"Overlap médio: {mean_overlap:.3f} ± {std_overlap:.3f}")

    # Comparar recall com diferentes níveis de ruído
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    overlaps_mean = []
    for nl in noise_levels:
        m, s = system.evaluate(noise_level=nl)
        overlaps_mean.append(m)
        print(f"  Ruído {nl*100:.0f}%: overlap = {m:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot([nl*100 for nl in noise_levels], overlaps_mean, 'b-o', linewidth=2)
    plt.xlabel('Nível de Ruído no Cue (%)')
    plt.ylabel('Overlap Médio com Original')
    plt.title('Performance do Sistema de Indexação Hipocampal')
    plt.grid(True, alpha=0.3)
    plt.savefig('semana12_indexing.png')
    print("\nGráfico salvo: semana12_indexing.png")
```

### ✅ Como saber se absorveu

- [ ] O sistema armazena e recupera 20 episódios com overlap médio >0.5 com 50% de ruído no cue
- [ ] Você escreveu um comentário longo no código explicando como isso difere de um dicionário Python (`dict`) — (resposta: um dict não generaliza, não completa padrões, não degraça graciosamente com ruído)
- [ ] Você consegue explicar: "O que acontece durante o sono? Por que o hipocampo precisa transferir as memórias para o córtex?"
- [ ] **Teste oral:** explique o sistema de indexação para uma planta (rubber duck) em 3 minutos — sem olhar o código

---

## ✅ CHECKLIST DO TRIMESTRE 1 (Meses 1–3)

Ao final destes 3 meses, você deve conseguir:

- [ ] Implementar e simular neurônio Integrate-and-Fire com parâmetros biofísicos reais
- [ ] Criar sinapses com dinâmica E/I e condutância sináptica
- [ ] Implementar aprendizado Hebbiano com normalização multiplicativa
- [ ] Implementar STDP completo e reproduzir a curva de Bi & Poo
- [ ] Construir redes com balanço E/I e interpretar raster plots
- [ ] Implementar esparsidade com k-WTA
- [ ] Construir e testar rede de Hopfield como CA3
- [ ] Implementar o circuito trisináptico EC→DG→CA3→CA1 com esparsidades biológicas
- [ ] Criar células de lugar com decodificação por população
- [ ] Construir o protótipo do sistema de indexação hipocampal

**Se você chegou até aqui, parabéns.** Você tem um hipocampo artificial funcionando.

---

*Continua no Volume II — Meses 4 a 6: Neocórtex, Integração e Pesquisa Aplicada*
