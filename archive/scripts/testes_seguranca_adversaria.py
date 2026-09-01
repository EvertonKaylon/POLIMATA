"""
Script de Simulação de Red Teaming e Ataques Adversários (Cibersegurança).

Este script simula vetores de ataque realistas inspirados em pesquisas coreanas (KCI/BoanNews)
contra a nossa arquitetura hipocampo-neocortical:
1. Replay Poisoning (Envenenamento de Replay Gerativo na fase de Vigília/Sono).
2. Prediction Error Hijacking (Sequestro da crença interna neocortical x2 na fase de Inferência).

Compara a resiliência dos modelos sem defesa (Vulnerável) vs. com defesa (Resiliente: Switch MS-MEC + Meta-PCN).
"""

import random
from typing import Dict, List, Tuple
from simulacao_dialogo_hipocampo import HipocampoRapido, NeocortexPreditivo


def gerar_experiencias_basicas(dim_sensorial: int) -> Tuple[List[float], List[float]]:
    """Gera dois padrões normais claramente distintos."""
    random.seed(42)
    # Evento A (normal)
    padrao_a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    # Evento B (normal)
    padrao_b = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    return padrao_a, padrao_b


def gerar_padrao_envenenado(padrao_alvo: List[float]) -> List[float]:
    """
    Gera um padrão envenenado (backdoor) invertendo alguns bits
    de forma a criar uma perturbação anômala.
    """
    # Copia o padrão original e perturba bits específicos
    envenenado = list(padrao_alvo)
    # Altera os bits do meio para simular o payload de envenenamento (Data Poisoning)
    envenenado[3] = 1.0 - envenenado[3]
    envenenado[4] = 1.0 - envenenado[4]
    envenenado[5] = 1.0 - envenenado[5]
    return envenenado


def simular_ataque_envenenamento(usar_defesa: bool) -> Dict[str, float]:
    """
    Simula o cenário de ataque de Replay Poisoning.
    Pre-treina o Neocórtex com padrões limpos para criar priors,
    em seguida injeta um padrão envenenado durante a vigília e avalia o impacto pós-sono.
    """
    dim_sens = 10
    dim_abst = 6
    alpha = 0.03
    epocas = 400
    passos = 12
    limiar_gaba = 0.30  # Limiar do switch inibidor MS-MEC ajustado para priors estabelecidos
    
    padrao_a, padrao_b = gerar_experiencias_basicas(dim_sens)
    padrao_poison = gerar_padrao_envenenado(padrao_a)
    
    # 1. Inicializar redes
    hipocampo = HipocampoRapido(dimensao=dim_sens)
    neocortex = NeocortexPreditivo(dim_sensorial=dim_sens, dim_abstrata=dim_abst, taxa_aprendizado=alpha)
    
    # 2. Pré-treinamento do Neocórtex com padrões limpos (Estabelecendo Priors do Agente)
    for _ in range(100):
        neocortex.inferir_estado(padrao_a, passos=passos)
        neocortex.consolidar_pesos()
        neocortex.inferir_estado(padrao_b, passos=passos)
        neocortex.consolidar_pesos()
    
    # 3. Vigília - Injeção das experiências (A, B e o Poison) sob ataque
    registros_sucesso = []
    
    # Registra A (normal) - deve ser aceito
    reg_a = hipocampo.registrar_experiencia(padrao_a, neocortex_ref=neocortex, limiar_gaba=limiar_gaba if usar_defesa else 0.0)
    registros_sucesso.append(reg_a)
    
    # Registra B (normal) - deve ser aceito
    reg_b = hipocampo.registrar_experiencia(padrao_b, neocortex_ref=neocortex, limiar_gaba=limiar_gaba if usar_defesa else 0.0)
    registros_sucesso.append(reg_b)
    
    # Registra Poison (envenenamento) - se defendido, deve ser bloqueado
    reg_poison = hipocampo.registrar_experiencia(padrao_poison, neocortex_ref=neocortex, limiar_gaba=limiar_gaba if usar_defesa else 0.0)
    registros_sucesso.append(reg_poison)
    
    # 4. Sono - Replay e consolidação adicional no neocórtex
    # Se o envenenamento foi registrado, ele participará dos replays do Hipocampo.
    for _ in range(epocas):
        memoria = hipocampo.fazer_replay()
        neocortex.inferir_estado(memoria, passos=passos)
        neocortex.consolidar_pesos()
        
    # 5. Avaliação pós-sono
    # Medir erro final nos padrões normais (esperamos erro baixo)
    neocortex.inferir_estado(padrao_a, passos=passos)
    erro_a = sum(abs(e) for e in neocortex.e1) / dim_sens
    
    neocortex.inferir_estado(padrao_b, passos=passos)
    erro_b = sum(abs(e) for e in neocortex.e1) / dim_sens
    
    erro_normal_medio = (erro_a + erro_b) / 2.0
    
    # Medir erro no padrão envenenado (Poison)
    neocortex.inferir_estado(padrao_poison, passos=passos)
    erro_poison = sum(abs(e) for e in neocortex.e1) / dim_sens
    
    # Taxa de Sobrevivência da Memória Limpa (MSR) - escala de 0 a 100
    msr = (1.0 - erro_normal_medio) * 100
    
    # Taxa de Sucesso do Ataque (ASR) - Se o erro do poison for baixo, o ataque teve sucesso
    # Se defendido, o ASR deve ser baixo porque o poison não foi registrado no Hipocampo.
    # Se vulnerável, o ASR é alto porque o poison foi reativado e treinado no sono.
    asr = (1.0 - erro_poison) * 100
    
    return {
        "reg_poison": 1.0 if reg_poison else 0.0,
        "erro_normal": erro_normal_medio,
        "erro_poison": erro_poison,
        "msr": msr,
        "asr": asr
    }


def simular_ataque_evasao(usar_defesa: bool) -> Dict[str, float]:
    """
    Simula o cenário de Prediction Error Hijacking (sequestro cognitivo).
    Injeta ruído adversário na inferência para tentar forçar o neocórtex
    a disparar uma crença interna (x2) incorreta.
    """
    dim_sens = 10
    dim_abst = 6
    alpha = 0.03
    passos = 15
    
    padrao_a, padrao_b = gerar_experiencias_basicas(dim_sens)
    
    # Inicializar redes e treinar o neocórtex no padrão A de forma limpa
    neocortex = NeocortexPreditivo(
        dim_sensorial=dim_sens, 
        dim_abstrata=dim_abst, 
        taxa_aprendizado=alpha,
        usar_meta_pcn=usar_defesa
    )
    
    # Treinamento rápido de consolidação limpa de A
    for _ in range(200):
        neocortex.inferir_estado(padrao_a, passos=passos)
        neocortex.consolidar_pesos(limite_w=1.5 if usar_defesa else 0.0)
        
    # Salvar a crença real ideal x2 obtida com o padrão limpo A
    neocortex.inferir_estado(padrao_a, passos=passos)
    x2_ideal = list(neocortex.x2)
    
    # Atacante gera um padrão A perturbado com ruído adversarial
    # O ruído é sutil, mas projetado para desviar a inferência
    ruido_adversarial = [0.0] * dim_sens
    # Altera alguns bits na entrada sensorial para manipular o erro e1
    ruido_adversarial[1] = 0.35
    ruido_adversarial[7] = -0.35
    
    padrao_perturbado = [max(0.0, min(1.0, v + r)) for v, r in zip(padrao_a, ruido_adversarial)]
    
    # Executa a inferência no padrão perturbado
    neocortex.inferir_estado(padrao_perturbado, passos=passos)
    x2_adversario = list(neocortex.x2)
    
    # Medir a correlação (ou desvio) entre a crença adversária e a ideal
    # Se o desvio for baixo, a crença foi mantida estável (Defendido)
    # Se o desvio for alto, a crença foi sequestrada (Vulnerável)
    desvio_crenca = sum(abs(i - a) for i, a in zip(x2_ideal, x2_adversario)) / dim_abst
    
    # Robustez Cognitiva (CR) - escala de 0 a 100
    cr = (1.0 - desvio_crenca) * 100
    
    return {
        "desvio_crenca": desvio_crenca,
        "cr": cr
    }


def main() -> None:
    largura = 115
    print("=" * largura)
    print(" " * 32 + "SIMULADOR DE RED TEAMING E ADVERSARIAL TESTING (IA NEUROMÓRFICA)")
    print("=" * largura)
    
    # -------------------------------------------------------------------------
    # TESTE 1: REPLAY POISONING ATTACK
    # -------------------------------------------------------------------------
    print("\n[TESTE 1] Iniciando Simulação de Replay Poisoning Attack...")
    res_vuln_p = simular_ataque_envenenamento(usar_defesa=False)
    res_def_p = simular_ataque_envenenamento(usar_defesa=True)
    
    print("-" * largura)
    print(f"{'Cenário de Envenenamento':<30} | {'MS-MEC Switch':<15} | {'Registro Poison':<16} | {'Erro Mems Limpas':<17} | {'ASR (Sucesso Ataque)':<20}")
    print("-" * largura)
    
    status_reg_vuln = "REGISTRADO (FALHA)" if res_vuln_p["reg_poison"] > 0 else "BLOQUEADO"
    status_reg_def = "REGISTRADO" if res_def_p["reg_poison"] > 0 else "BLOQUEADO (SUCESSO)"
    
    print(f"{'Modelo Vulnerável':<30} | {'Desativado':<15} | {status_reg_vuln:<16} | {res_vuln_p['erro_normal']:<17.5f} | {res_vuln_p['asr']:>18.1f}%")
    print(f"{'Modelo Resiliente (Gated)':<30} | {'Ativado':<15} | {status_reg_def:<16} | {res_def_p['erro_normal']:<17.5f} | {res_def_p['asr']:>18.1f}%")
    print("=" * largura)
    print("Análise de Envenenamento:")
    print(" - No Modelo Vulnerável, o exploit é registrado no Hipocampo e consolidado no Neocórtex, atingindo ASR de {:.1f}%.".format(res_vuln_p["asr"]))
    print(" - No Modelo Resiliente, o switch GABAérgico detecta anomalia de erro na Vigília e bloqueia o registro, mantendo ASR em {:.1f}%.".format(res_def_p["asr"]))
    
    # -------------------------------------------------------------------------
    # TESTE 2: PREDICTION ERROR HIJACKING
    # -------------------------------------------------------------------------
    print("\n\n[TESTE 2] Iniciando Simulação de Prediction Error Hijacking (Evasão)...")
    res_vuln_e = simular_ataque_evasao(usar_defesa=False)
    res_def_e = simular_ataque_evasao(usar_defesa=True)
    
    print("-" * largura)
    print(f"{'Cenário de Evasão':<30} | {'Mecanismo Meta-PCN':<20} | {'Desvio de Crença x2':<22} | {'Robustez Cognitiva (CR)':<25}")
    print("-" * largura)
    print(f"{'Modelo Vulnerável':<30} | {'Desativado':<20} | {res_vuln_e['desvio_crenca']:<22.5f} | {res_vuln_e['cr']:>23.1f}%")
    print(f"{'Modelo Resiliente':<30} | {'Ativado':<20} | {res_def_e['desvio_crenca']:<22.5f} | {res_def_e['cr']:>23.1f}%")
    print("=" * largura)
    print("Análise de Evasão:")
    print(" - No Modelo Vulnerável, o ruído adversarial desvia a crença x2 do neocórtex (desvio de {:.5f}).".format(res_vuln_e["desvio_crenca"]))
    print(" - No Modelo Resiliente, a suavização de Meta-PCN estabiliza a inferência, reduzindo o desvio e elevando a Robustez Cognitiva para {:.1f}%.".format(res_def_e["cr"]))
    print("=" * largura)


if __name__ == "__main__":
    main()
