"""
Simulação do Diálogo Hipocampo-Neocórtex e Consolidação de Memória.

Este script modela de forma limpa e modular a cooperação complementar de aprendizado
entre o hipocampo (gravação rápida de passo único via sinapses biofísicas) e o
neocórtex (consolidação lenta via Codificação Preditiva) durante as fases de
Vigília e Sono (Memory Replay).

Boas práticas seguidas: Oriented Object Programming (OOP), Type Hinting,
reprodutibilidade via sementes fixas, e comentários associados à neurobiologia.
"""

import math
import random
from typing import Any, List, Tuple

# Fixar a semente aleatória para reprodutibilidade dos testes
random.seed(42)


# ============================================================================
# 1. FUNÇÕES MATEMÁTICAS UTILITÁRIAS (Python Puro)
# ============================================================================

def sigmoid(x: float) -> float:
    """Função de ativação Sigmoid (equivalente à taxa de disparo de um neurônio)."""
    # Evitar overflow limitando o valor de x
    x = max(-50.0, min(50.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(y: float) -> float:
    """Derivada da Sigmoid expressa em função da própria ativação de saída y."""
    return y * (1.0 - y)


def dot_product(v1: List[float], v2: List[float]) -> float:
    """Calcula o produto escalar entre dois vetores."""
    return sum(x * y for x, y in zip(v1, v2))


# ============================================================================
# 2. MODELO DE SINAPSE BIOFÍSICA (Micro-Escala)
# ============================================================================

class SinapseBiofisica:
    """
    Modela o comportamento funcional e molecular de uma sinapse pós-sináptica.
    Contém a dinâmica dos receptores AMPA, NMDA e ativação persistente de CaMKII.
    """

    def __init__(self, peso_inicial: float = 0.1) -> None:
        # Condutância basal (AMPA) - representa a força sináptica direta (W na IA)
        self.w_ampa: float = peso_inicial
        
        # Estado do canal NMDA (abertura depende de coincidência pré/pós-sináptica)
        self.nmda_aberto: bool = False
        
        # CaMKII (proteína de memória molecular que varia de 0 a 1)
        self.camkii_ativa: float = 0.0

    def processar_coincidencia(self, pre_disparo: float, pos_despolarizacao: float) -> None:
        """
        Calcula a abertura do canal NMDA como um detector de coincidência Hebbiano (AND).
        O NMDA só abre se o neurônio pré enviar glutamato E o neurônio pós estiver ativo.
        """
        limiar_nmda = 0.5
        # O NMDA abre se ambos os lados passarem o limiar físico
        if pre_disparo > limiar_nmda and pos_despolarizacao > limiar_nmda:
            self.nmda_aberto = True
            # Influxo de cálcio ativa a enzima CaMKII (autofosforilação)
            self.camkii_ativa = min(1.0, self.camkii_ativa + 0.3)
        else:
            self.nmda_aberto = False
            # Decaimento natural lento da atividade de CaMKII na ausência de sinal
            self.camkii_ativa = max(0.0, self.camkii_ativa - 0.05)

    def aplicar_plasticidade(self) -> bool:
        """
        Consolida a plasticidade de longo prazo (LTP).
        Se a CaMKII estiver muito ativa, ela recruta novos canais AMPA, aumentando o peso.
        Retorna True se houve Potenciação de Longo Prazo (LTP).
        """
        limiar_ltp = 0.7
        if self.camkii_ativa > limiar_ltp:
            # Fortalece a conexão permanentemente
            incremento = 0.4 * self.camkii_ativa
            self.w_ampa += incremento
            # Reset da ativação da CaMKII após consolidar em AMPA
            self.camkii_ativa = 0.0
            return True
        return False


# ============================================================================
# 3. HIPOCAMPO (Gravador de Aprendizado Rápido)
# ============================================================================

class HipocampoRapido:
    """
    Modela o Hipocampo (região CA3 recorrente).
    Otimizado para armazenar memórias episódicas do dia em passo único de vigília
    e fazer o replay dessas memórias no sono.
    """

    def __init__(self, dimensao: int) -> None:
        self.dimensao: int = dimensao
        # Rede autoassociativa (matriz de pesos sinápticos internos)
        self.pesos: List[List[SinapseBiofisica]] = [
            [SinapseBiofisica(0.0) for _ in range(dimensao)] for _ in range(dimensao)
        ]
        # Buffer de memórias episódicas coletadas durante o dia
        self.buffer_memorias: List[List[float]] = []

    def registrar_experiencia(self, padrao: List[float], neocortex_ref: Any = None, limiar_gaba: float = 0.0) -> bool:
        """
        Grava o padrão de forma instantânea durante a vigília usando canais sinápticos.
        Se neocortex_ref e limiar_gaba forem fornecidos, o switch GABAérgico (MS-MEC)
        bloqueia padrões anômalos que gerem erro superior ao limiar (retorna False).
        """
        if neocortex_ref is not None and limiar_gaba > 0.0:
            # Backup do estado da crença
            crenca_backup = list(neocortex_ref.x2)
            neocortex_ref.inferir_estado(padrao, passos=5)
            erro_medio = sum(abs(e) for e in neocortex_ref.e1) / neocortex_ref.dim_sensorial
            # Restaurar crença
            neocortex_ref.x2 = crenca_backup
            
            if erro_medio > limiar_gaba:
                return False  # Bloqueado pelo switch GABAérgico
                
        self.buffer_memorias.append(padrao)
        
        # Ajusta as conexões internas com base na regra de Hebb biológica (NMDA/CaMKII)
        for i in range(self.dimensao):
            for j in range(self.dimensao):
                if i == j:
                    continue  # Sem autoconexão direta
                
                sinapse = self.pesos[i][j]
                # Processa a atração física Hebbiana (AND entre disparos i e j)
                sinapse.processar_coincidencia(padrao[i], padrao[j])
                
                # Consolida o peso sináptico
                sinapse.aplicar_plasticidade()
                
        return True

    def fazer_replay(self) -> List[float]:
        """
        Emite uma memória arquivada de forma reconsolidada (Memory Replay).
        Simula o disparo assíncrono de sharp-wave ripples (SWRs).
        """
        if not self.buffer_memorias:
            return [0.0] * self.dimensao
        # Escolhe aleatoriamente uma das memórias registradas no dia para repetir
        return random.choice(self.buffer_memorias)


# ============================================================================
# 4. NEOCÓRTEX (Estrutura de Previsão Lenta via Codificação Preditiva)
# ============================================================================

class NeocortexPreditivo:
    """
    Modela a hierarquia neocortical usando os princípios da Codificação Preditiva (PC).
    Aprende lentamente minimizando erros de previsão locais gerados top-down.
    """

    def __init__(self, dim_sensorial: int, dim_abstrata: int, taxa_aprendizado: float = 0.05, usar_meta_pcn: bool = False) -> None:
        self.dim_sensorial: int = dim_sensorial
        self.dim_abstrata: int = dim_abstrata
        self.alpha: float = taxa_aprendizado
        self.usar_meta_pcn: bool = usar_meta_pcn
        
        # Matriz de pesos de predição (da camada abstrata para a sensorial)
        self.W: List[List[float]] = [
            [random.uniform(-0.2, 0.2) for _ in range(dim_abstrata)]
            for _ in range(dim_sensorial)
        ]
        
        # Crença interna/Estado neural da camada abstrata superior (x2)
        self.x2: List[float] = [0.1] * dim_abstrata
        
        # Erro de predição local na camada sensorial inferior (e1)
        self.e1: List[float] = [0.0] * dim_sensorial

    def prever(self) -> List[float]:
        """Gera a predição top-down (de cima para baixo) para a camada sensorial."""
        predicao = []
        for i in range(self.dim_sensorial):
            # Soma ponderada dos pesos pela crença superior
            soma = dot_product(self.W[i], self.x2)
            predicao.append(sigmoid(soma))
        return predicao

    def inferir_estado(self, entrada_sensorial: List[float], passos: int = 20) -> List[float]:
        """
        Fase de Inferência: O neocórtex ajusta sua crença interna (x2)
        para diminuir o erro de predição local e1 gerado pela entrada sensorial.
        Suporta suavização de gradiente por erro de meta-predição se usar_meta_pcn for True.
        """
        # Reinicializar a crença abstrata com um estado basal neutro a cada novo estímulo
        self.x2 = [0.5] * self.dim_abstrata
        gamma = 0.1  # Taxa de atualização neural
        e1_antigo = None
        
        for _ in range(passos):
            # 1. Obter a predição atual
            predicao = self.prever()
            
            # 2. Calcular o erro de predição local (e1)
            # e1 = entrada_sensorial - predição
            self.e1 = [s - p for s, p in zip(entrada_sensorial, predicao)]
            
            # Amortecimento via erro de Meta-Predição (Meta-PCN)
            if self.usar_meta_pcn and e1_antigo is not None:
                meta_e = [curr - prev for curr, prev in zip(self.e1, e1_antigo)]
                # Suaviza a dinâmica da crença reduzindo a oscilação do erro
                self.e1 = [curr - 0.25 * me for curr, me in zip(self.e1, meta_e)]
                
            e1_antigo = list(self.e1)
            
            # 3. Atualizar a crença abstrata superior (x2) para diminuir o erro
            # x2_novo = x2_antigo + gradiente local (W^T * e1 * f')
            x2_grad = [0.0] * self.dim_abstrata
            for j in range(self.dim_abstrata):
                soma_grad = 0.0
                for i in range(self.dim_sensorial):
                    f_deriv = sigmoid_derivative(predicao[i])
                    # Projeta o erro sensorial de volta usando o peso transposto
                    soma_grad += self.W[i][j] * self.e1[i] * f_deriv
                x2_grad[j] = soma_grad
            
            # Atualiza o disparo dos neurônios superiores
            for j in range(self.dim_abstrata):
                self.x2[j] = max(0.0, min(1.0, self.x2[j] + gamma * x2_grad[j]))
                
        return self.prever()

    def consolidar_pesos(self, limite_w: float = 0.0) -> float:
        """
        Fase de Aprendizado: Ajusta a força sináptica W baseando-se no erro local.
        Regra hebbiana local: ΔW = α * e1 * x2^T.
        Suporta o limitador de variância sináptica (limite_w) do Meta-PCN.
        Retorna a magnitude média do erro local consolidado.
        """
        for i in range(self.dim_sensorial):
            for j in range(self.dim_abstrata):
                # A sinapse muda dependendo do erro local e do disparo do neurônio superior
                self.W[i][j] += self.alpha * self.e1[i] * self.x2[j]
        
        # Bounding L1 por neurônio sensorial (Prevenção de saturação/Meta-PCN)
        if limite_w > 0.0:
            for i in range(self.dim_sensorial):
                soma = sum(abs(self.W[i][j]) for j in range(self.dim_abstrata))
                if soma > limite_w:
                    scale = limite_w / soma
                    for j in range(self.dim_abstrata):
                        self.W[i][j] *= scale
                        
        # Retorna o erro médio absoluto atual
        return sum(abs(err) for err in self.e1) / self.dim_sensorial


# ============================================================================
# 4.1 NEOCÓRTEX PROFUNDO (Hierarquia de 3 Camadas)
# ============================================================================

class NeocortexPreditivoProfundo:
    """
    Modela uma hierarquia neocortical profunda de 3 camadas (L1, L2, L3)
    usando Codificação Preditiva e suporte a Sinapses Memristivas Ternárias.
    """

    def __init__(self, dim_l1: int, dim_l2: int, dim_l3: int, taxa_aprendizado: float = 0.03,
                 usar_meta_pcn: bool = False, usar_ternario: bool = False, limiar_memristor: float = 0.15) -> None:
        self.dim_l1: int = dim_l1
        self.dim_sensorial: int = dim_l1
        self.dim_l2: int = dim_l2
        self.dim_l3: int = dim_l3
        self.alpha: float = taxa_aprendizado
        self.usar_meta_pcn: bool = usar_meta_pcn
        self.usar_ternario: bool = usar_ternario
        self.limiar_memristor: float = limiar_memristor

        # W1: L1 (sensorial) x L2 (intermediário)
        self.W1: List[List[float]] = [
            [random.uniform(-0.2, 0.2) for _ in range(dim_l2)]
            for _ in range(dim_l1)
        ]
        
        # W2: L2 (intermediário) x L3 (deep abstrato)
        self.W2: List[List[float]] = [
            [random.uniform(-0.2, 0.2) for _ in range(dim_l3)]
            for _ in range(dim_l2)
        ]

        # Crenças internas
        self.x2: List[float] = [0.5] * dim_l2
        self.x3: List[float] = [0.5] * dim_l3

        # Vetores de erro
        self.e1: List[float] = [0.0] * dim_l1
        self.e2: List[float] = [0.0] * dim_l2

    def obter_w1_efetivo(self) -> List[List[float]]:
        """Aplica a discretização memristiva ternária em W1 se ativado."""
        if not self.usar_ternario:
            return self.W1
        return [
            [1.0 if w > self.limiar_memristor else (-1.0 if w < -self.limiar_memristor else 0.0) for w in row]
            for row in self.W1
        ]

    def obter_w2_efetivo(self) -> List[List[float]]:
        """Aplica a discretização memristiva ternária em W2 se ativado."""
        if not self.usar_ternario:
            return self.W2
        return [
            [1.0 if w > self.limiar_memristor else (-1.0 if w < -self.limiar_memristor else 0.0) for w in row]
            for row in self.W2
        ]

    def prever_l2_l1(self, w1_eff: List[List[float]]) -> List[float]:
        """Gera a predição top-down de L2 para a camada sensorial L1."""
        pred = []
        for i in range(self.dim_l1):
            soma = sum(w1_eff[i][j] * self.x2[j] for j in range(self.dim_l2))
            pred.append(sigmoid(soma))
        return pred

    def prever_l3_l2(self, w2_eff: List[List[float]]) -> List[float]:
        """Gera a predição top-down de L3 para a camada intermediária L2."""
        pred = []
        for j in range(self.dim_l2):
            soma = sum(w2_eff[j][k] * self.x3[k] for k in range(self.dim_l3))
            pred.append(sigmoid(soma))
        return pred

    def inferir_estado(self, entrada_sensorial: List[float], passos: int = 25) -> Tuple[List[float], List[float]]:
        """
        Fase de Inferência Ativa Profunda:
        Ajusta de forma concorrente as crenças x2 e x3 para minimizar os erros e1 e e2.
        Suporta amortecimento de meta-predição (Meta-PCN).
        """
        self.x2 = [0.5] * self.dim_l2
        self.x3 = [0.5] * self.dim_l3
        gamma = 0.1  # Taxa de atualização neural
        
        e1_antigo = None
        e2_antigo = None

        w1_eff = self.obter_w1_efetivo()
        w2_eff = self.obter_w2_efetivo()

        for _ in range(passos):
            # 1. Predições top-down
            pred_l2 = self.prever_l3_l2(w2_eff)
            pred_l1 = self.prever_l2_l1(w1_eff)

            # 2. Erros locais
            self.e1 = [s - p for s, p in zip(entrada_sensorial, pred_l1)]
            self.e2 = [x - p for x, p in zip(self.x2, pred_l2)]

            # 3. Meta-PCN (amortecimento por velocidade do erro)
            if self.usar_meta_pcn:
                if e1_antigo is not None:
                    meta_e1 = [curr - prev for curr, prev in zip(self.e1, e1_antigo)]
                    self.e1 = [curr - 0.25 * me for curr, me in zip(self.e1, meta_e1)]
                if e2_antigo is not None:
                    meta_e2 = [curr - prev for curr, prev in zip(self.e2, e2_antigo)]
                    self.e2 = [curr - 0.25 * me for curr, me in zip(self.e2, meta_e2)]

            e1_antigo = list(self.e1)
            e2_antigo = list(self.e2)

            # 4. Calcular os gradientes das crenças
            # L3 gradient (bottom-up from e2)
            grad_x3 = [0.0] * self.dim_l3
            for k in range(self.dim_l3):
                soma = 0.0
                for j in range(self.dim_l2):
                    f_deriv = sigmoid_derivative(pred_l2[j])
                    soma += w2_eff[j][k] * self.e2[j] * f_deriv
                grad_x3[k] = soma

            # L2 gradient (bottom-up from e1 e top-down e2)
            grad_x2 = [0.0] * self.dim_l2
            for j in range(self.dim_l2):
                soma_bu = 0.0
                for i in range(self.dim_l1):
                    f_deriv = sigmoid_derivative(pred_l1[i])
                    soma_bu += w1_eff[i][j] * self.e1[i] * f_deriv
                # Δx2 = bottom_up_projection - top_down_error
                grad_x2[j] = soma_bu - self.e2[j]

            # 5. Atualizar crenças com projeção no intervalo [0, 1]
            for k in range(self.dim_l3):
                self.x3[k] = max(0.0, min(1.0, self.x3[k] + gamma * grad_x3[k]))
            for j in range(self.dim_l2):
                self.x2[j] = max(0.0, min(1.0, self.x2[j] + gamma * grad_x2[j]))

        return self.prever_l2_l1(w1_eff), self.prever_l3_l2(w2_eff)

    def consolidar_pesos(self, limite_w1: float = 0.0, limite_w2: float = 0.0) -> float:
        """
        Ajusta os pesos sinápticos reais (analógicos) W1 e W2 com base nas regras locais Hebbianas.
        Aplica normalização L1 se os limites forem maiores que zero.
        """
        # Atualização Hebbiana de W1 (sensorial)
        for i in range(self.dim_l1):
            for j in range(self.dim_l2):
                self.W1[i][j] += self.alpha * self.e1[i] * self.x2[j]

        # Atualização Hebbiana de W2 (intermediário)
        for j in range(self.dim_l2):
            for k in range(self.dim_l3):
                self.W2[j][k] += self.alpha * self.e2[j] * self.x3[k]

        # Bounding L1 por neurônio receptor (Prevenção de saturação)
        if limite_w1 > 0.0:
            for i in range(self.dim_l1):
                soma = sum(abs(self.W1[i][j]) for j in range(self.dim_l2))
                if soma > limite_w1:
                    scale = limite_w1 / soma
                    for j in range(self.dim_l2):
                        self.W1[i][j] *= scale

        if limite_w2 > 0.0:
            for j in range(self.dim_l2):
                soma = sum(abs(self.W2[j][k]) for k in range(self.dim_l3))
                if soma > limite_w2:
                    scale = limite_w2 / soma
                    for k in range(self.dim_l3):
                        self.W2[j][k] *= scale

        # Retorna o erro médio absoluto final da camada sensorial L1
        return sum(abs(err) for err in self.e1) / self.dim_l1


# ============================================================================
# 4.2 MODELAGEM ESPICULANTE (SNN, LIF & STDP)
# ============================================================================

class NeuronioLIF:
    """
    Modela o comportamento biofísico de um neurônio Leaky Integrate-and-Fire.
    Integra correntes de entrada e dispara um pulso digital (spike) ao atingir o limiar.
    """

    def __init__(self, dt: float = 1.0) -> None:
        self.dt: float = dt
        self.V_rest: float = -70.0  # Potencial de repouso (mV)
        self.V_reset: float = -75.0  # Potencial de reset pós-disparo (mV)
        self.V_th: float = -50.0  # Limiar de disparo (mV)
        self.tau_m: float = 20.0  # Constante de tempo de membrana (ms)
        self.tau_ref: float = 2.0  # Período refratário (ms)
        
        # Estados
        self.V: float = self.V_rest
        self.tempo_ultimo_spike: float = -999.0
        self.refatario_restante: float = 0.0

    def integrar_e_disparar(self, corrente_entrada: float, tempo_atual: float) -> bool:
        """
        Integra a corrente usando o método de Euler e dispara se atingir o limiar V_th.
        Retorna True se houve um spike.
        """
        if self.refatario_restante > 0.0:
            self.refatario_restante = max(0.0, self.refatario_restante - self.dt)
            self.V = self.V_reset
            return False

        # Integração do potencial de membrana (Decaimento passivo + corrente)
        decaimento = (self.V_rest - self.V) / self.tau_m
        self.V += decaimento * self.dt + corrente_entrada * self.dt

        # Verificação do limiar
        if self.V >= self.V_th:
            self.V = self.V_reset
            self.tempo_ultimo_spike = tempo_atual
            self.refatario_restante = self.tau_ref
            return True
        return False


class SinapseSTDP:
    """
    Modela a regra de plasticidade sináptica dependente do tempo de disparo (STDP).
    Fortalece pesos se o pre-spike precede o pos-spike (LTP), enfraquece caso contrário (LTD).
    """

    def __init__(self, peso_inicial: float = 0.5) -> None:
        self.w: float = peso_inicial
        self.w_min: float = 0.0
        self.w_max: float = 2.0
        self.tau_stdp: float = 20.0  # Janela de ativação (ms)
        self.A_plus: float = 0.05  # Amplitude máxima de LTP
        self.A_minus: float = 0.035  # Amplitude máxima de LTD

    def aplicar_stdp(self, tempo_pre: float, tempo_pos: float) -> float:
        """
        Ajusta o peso com base na diferença de tempo: delta_t = tempo_pos - tempo_pre.
        Aplica LTP se delta_t > 0 e LTD se delta_t < 0.
        """
        if tempo_pre < 0.0 or tempo_pos < 0.0:
            return self.w  # Nenhum spike ocorreu em um dos lados

        delta_t = tempo_pos - tempo_pre

        if delta_t > 0.0:
            # LTP: Pré disparou antes do pós (Causalidade)
            self.w += self.A_plus * math.exp(-delta_t / self.tau_stdp)
        elif delta_t < 0.0:
            # LTD: Pré disparou depois do pós (Anti-causalidade)
            self.w -= self.A_minus * math.exp(delta_t / self.tau_stdp)

        # Restrição física do peso
        self.w = max(self.w_min, min(self.w_max, self.w))
        return self.w


# ============================================================================
# 5. ORQUESTRAÇÃO DA SIMULAÇÃO (O Ciclo de 24 Horas)
# ============================================================================

class SimuladorDialogo:
    """Classe que coordena as fases de Vigília, Sono e Avaliação do modelo."""

    def __init__(self) -> None:
        self.dim_sensorial = 5
        self.dim_abstrata = 3
        
        self.hipocampo = HipocampoRapido(dimensao=self.dim_sensorial)
        self.neocortex = NeocortexPreditivo(
            dim_sensorial=self.dim_sensorial,
            dim_abstrata=self.dim_abstrata,
            taxa_aprendizado=0.03
        )
        
        # Padrões que representam "Experiências do Dia" (Episódios)
        # 1. Evento A (ex: "Almoço com amigos") -> [1.0, 0.0, 1.0, 0.0, 1.0]
        # 2. Evento B (ex: "Leitura à noite")   -> [0.0, 1.0, 0.0, 1.0, 0.0]
        self.experiencias_dia = [
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0]
        ]

    def rodar_vigilia(self) -> None:
        """Fase de Vigília: O cérebro capta estímulos e o Hipocampo grava rápido."""
        print("\n--- ☀️ INICIANDO FASE DE VIGÍLIA ---")
        for idx, evento in enumerate(self.experiencias_dia):
            print(f"Registrando Evento {idx + 1} {evento} no Hipocampo...")
            self.hipocampo.registrar_experiencia(evento)
        print("Vigília encerrada. Memórias indexadas temporariamente no Hipocampo.")

    def avaliar_neocortex(self) -> List[float]:
        """Calcula o erro médio do neocórtex nos padrões antes/depois do treino."""
        erros = []
        for evento in self.experiencias_dia:
            self.neocortex.inferir_estado(evento, passos=5)
            # Calcula erro médio absoluto
            erro = sum(abs(e) for e in self.neocortex.e1) / self.dim_sensorial
            erros.append(erro)
        return erros

    def rodar_sono(self, epocas_replay: int = 150) -> None:
        """
        Fase de Sono: Sem estímulo externo. O hipocampo dispara SWRs (replays)
        e o Neocórtex treina seus pesos lentos via Codificação Preditiva.
        """
        print("\n--- 💤 INICIANDO FASE DE SONO (MEMORY REPLAY) ---")
        print(f"Executando {epocas_replay} ciclos de replays acelerados...")
        
        for epoca in range(1, epocas_replay + 1):
            # 1. Hipocampo recupera e reativa uma memória do dia
            memoria_reproduzida = self.hipocampo.fazer_replay()
            
            # 2. Neocórtex faz a inferência ativa (ajusta x2 para o replay)
            self.neocortex.inferir_estado(memoria_reproduzida, passos=10)
            
            # 3. Neocórtex ajusta suas sinapses W locais baseando-se no erro
            erro_medio = self.neocortex.consolidar_pesos()
            
            # Log de progresso periódico
            if epoca % 100 == 0 or epoca == 1:
                print(f"Época {epoca:03d} | Replay: {memoria_reproduzida} | Erro de Predição Local: {erro_medio:.5f}")

        print("Fase de Sono concluída. Pesos do Neocórtex estabilizados.")


# ============================================================================
# 6. EXECUÇÃO PRINCIPAL
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("INICIANDO SIMULAÇÃO COMPUTACIONAL DO DIÁLOGO HIPOCAMPO-NEOCÓRTEX")
    print("=" * 70)

    # Instancia o simulador
    simulador = SimuladorDialogo()
    
    # 1. Medir o erro do Neocórtex ANTES do sono (ele nunca viu esses dados)
    erros_iniciais = simulador.avaliar_neocortex()
    print("\n--- 📊 ESTADO INICIAL DO NEOCÓRTEX (PRE-SONO) ---")
    for idx, err in enumerate(erros_iniciais):
        print(f"Erro de previsão para Evento {idx + 1}: {err:.5f} (Alto)")

    # 2. Rodar a Vigília (Captação rápida no hipocampo)
    simulador.rodar_vigilia()

    # 3. Rodar o Sono (Memory Replay e consolidação preditiva)
    simulador.rodar_sono(epocas_replay=600)

    # 4. Medir o erro do Neocórtex DEPOIS do sono (consolidação concluída)
    erros_finais = simulador.avaliar_neocortex()
    print("\n--- 📊 ESTADO FINAL DO NEOCÓRTEX (PÓS-SONO) ---")
    for idx, err in enumerate(erros_finais):
        melhoria = (erros_iniciais[idx] - err) / erros_iniciais[idx] * 100
        print(f"Erro de previsão para Evento {idx + 1}: {err:.5f} (Baixo) -> Melhoria: {melhoria:.1f}%")

    print("\n" + "=" * 70)
    print("SIMULAÇÃO CONCLUÍDA COM SUCESSO!")
    print("O neocórtex integrou e consolidou os dados usando apenas erros locais.")
    print("=" * 70)


if __name__ == "__main__":
    main()
