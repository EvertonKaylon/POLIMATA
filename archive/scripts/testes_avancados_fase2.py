"""
Script de Experimentos e Testes Avançados - Fase 2 R&D.

Executa simulações estatísticas detalhadas comparando:
1. Hierarquia de 3 camadas vs. 2 camadas sob diferentes níveis de ruído sensorial.
2. Sinapses memristivas ternárias discretizadas vs. analógicas sob ruído.
3. Experimento de Aprendizado Contínuo (Task Switching) com e sem mecanismos
   homeostáticos (poda sináptica, normalização L1 e switch GABAérgico MS-MEC).

Boas práticas seguidas: Python puro, type hints, semente fixa e formatação limpa.
"""

import random
from typing import Dict, List, Tuple, Any
from simulacao_dialogo_hipocampo import HipocampoRapido, NeocortexPreditivo, NeocortexPreditivoProfundo


def aplicar_ruido(padrao: List[float], taxa_ruido: float) -> List[float]:
    """Inverte valores binários com base em uma probabilidade para simular ruído."""
    if taxa_ruido <= 0.0:
        return padrao
    return [
        1.0 - val if random.random() < taxa_ruido else val
        for val in padrao
    ]


def gerar_experiencias_fase2(dim: int) -> Tuple[List[float], List[float], List[float]]:
    """Gera três padrões normais distintos para os testes."""
    random.seed(42)
    p1 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    p2 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    p3 = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    return p1, p2, p3


# ============================================================================
# EXPERIMENTO 1: PROFUNDIDADE E ROBUSTEZ A RUÍDO
# ============================================================================

def rodar_experimento_profundidade() -> None:
    print("\n" + "=" * 105)
    print(" EXPERIMENTO 1: HIERARQUIA DE 3 CAMADAS VS. 2 CAMADAS SOB RUÍDO")
    print("=" * 105)
    print(f"{'Modelo':<25} | {'Nível de Ruído':<15} | {'Erro Inicial':<15} | {'Erro Final':<15} | {'Melhoria %':<15}")
    print("-" * 105)

    dim_sens = 10
    dim_abst_2l = 4
    dim_int_3l = 6
    dim_abst_3l = 4
    epocas = 400
    passos = 15
    niveis_ruido = [0.0, 0.1, 0.2, 0.3, 0.4]

    for ruido in niveis_ruido:
        # --- Modelo 2 Camadas (Original) ---
        random.seed(42)
        p1, p2, p3 = gerar_experiencias_fase2(dim_sens)
        experiencias = [p1, p2, p3]
        
        hipo_2l = HipocampoRapido(dimensao=dim_sens)
        for p in experiencias:
            hipo_2l.registrar_experiencia(p)
            
        neo_2l = NeocortexPreditivo(dim_sensorial=dim_sens, dim_abstrata=dim_abst_2l, taxa_aprendizado=0.03)

        # Erro inicial
        erros_ini_2l = []
        for p in experiencias:
            neo_2l.inferir_estado(p, passos=passos)
            erros_ini_2l.append(sum(abs(e) for e in neo_2l.e1) / dim_sens)
        err_ini_2l_med = sum(erros_ini_2l) / len(erros_ini_2l)

        # Sono (treinamento)
        for _ in range(epocas):
            mem = hipo_2l.fazer_replay()
            mem_r = aplicar_ruido(mem, ruido)
            neo_2l.inferir_estado(mem_r, passos=passos)
            neo_2l.consolidar_pesos()

        # Erro final
        erros_fim_2l = []
        for p in experiencias:
            neo_2l.inferir_estado(p, passos=passos)
            erros_fim_2l.append(sum(abs(e) for e in neo_2l.e1) / dim_sens)
        err_fim_2l_med = sum(erros_fim_2l) / len(erros_fim_2l)
        melhoria_2l = ((err_ini_2l_med - err_fim_2l_med) / err_ini_2l_med) * 100

        # --- Modelo 3 Camadas ---
        random.seed(42)
        hipo_3l = HipocampoRapido(dimensao=dim_sens)
        for p in experiencias:
            hipo_3l.registrar_experiencia(p)
            
        neo_3l = NeocortexPreditivoProfundo(
            dim_l1=dim_sens, dim_l2=dim_int_3l, dim_l3=dim_abst_3l, taxa_aprendizado=0.03, usar_meta_pcn=True
        )

        # Erro inicial
        erros_ini_3l = []
        for p in experiencias:
            neo_3l.inferir_estado(p, passos=passos)
            erros_ini_3l.append(sum(abs(e) for e in neo_3l.e1) / dim_sens)
        err_ini_3l_med = sum(erros_ini_3l) / len(erros_ini_3l)

        # Sono
        for _ in range(epocas):
            mem = hipo_3l.fazer_replay()
            mem_r = aplicar_ruido(mem, ruido)
            neo_3l.inferir_estado(mem_r, passos=passos)
            neo_3l.consolidar_pesos()

        # Erro final
        erros_fim_3l = []
        for p in experiencias:
            neo_3l.inferir_estado(p, passos=passos)
            erros_fim_3l.append(sum(abs(e) for e in neo_3l.e1) / dim_sens)
        err_fim_3l_med = sum(erros_fim_3l) / len(erros_fim_3l)
        melhoria_3l = ((err_ini_3l_med - err_fim_3l_med) / err_ini_3l_med) * 100

        ruido_pct = f"{int(ruido * 100)}%"
        print(f"{'2 Camadas (PCN)':<25} | {ruido_pct:<15} | {err_ini_2l_med:<15.5f} | {err_fim_2l_med:<15.5f} | {melhoria_2l:>13.1f}%")
        print(f"{'3 Camadas (Meta-PCN)':<25} | {ruido_pct:<15} | {err_ini_3l_med:<15.5f} | {err_fim_3l_med:<15.5f} | {melhoria_3l:>13.1f}%")
        print("-" * 105)


# ============================================================================
# EXPERIMENTO 2: ESTABILIZAÇÃO POR SINAPSE MEMRISTIVA TERNÁRIA
# ============================================================================

def calcular_esparsidade(W: List[List[float]]) -> float:
    """Calcula a porcentagem de pesos sinápticos zerados (inativos)."""
    total = len(W) * len(W[0])
    zeros = sum(1 for row in W for w in row if w == 0.0)
    return (zeros / total) * 100


def rodar_experimento_memristores() -> None:
    print("\n" + "=" * 115)
    print(" EXPERIMENTO 2: CONVERSÃO DE PESOS ANALÓGICOS PARA SINAPSE TERNÁRIA")
    print("=" * 115)
    print(f"{'Modo Sináptico':<25} | {'Nível Ruído':<12} | {'Limiar θ':<10} | {'Erro Inicial':<14} | {'Erro Final':<14} | {'Esparsidade':<12} | {'Melhoria':<10}")
    print("-" * 115)

    dim_sens = 10
    dim_int = 6
    dim_abst = 4
    epocas = 500
    passos = 15
    ruido = 0.15  # Ruído moderado fixo

    configuracoes = [
        {"usar_ternario": False, "limiar": 0.0,  "desc": "Analógico (Contínuo)"},
        {"usar_ternario": True,  "limiar": 0.08, "desc": "Ternário (Limiar Baixo)"},
        {"usar_ternario": True,  "limiar": 0.15, "desc": "Ternário (Limiar Médio)"},
        {"usar_ternario": True,  "limiar": 0.22, "desc": "Ternário (Limiar Alto)"},
    ]

    for config in configuracoes:
        random.seed(42)
        p1, p2, p3 = gerar_experiencias_fase2(dim_sens)
        experiencias = [p1, p2, p3]

        hipo = HipocampoRapido(dimensao=dim_sens)
        for p in experiencias:
            hipo.registrar_experiencia(p)

        neo = NeocortexPreditivoProfundo(
            dim_l1=dim_sens,
            dim_l2=dim_int,
            dim_l3=dim_abst,
            taxa_aprendizado=0.03,
            usar_meta_pcn=True,
            usar_ternario=config["usar_ternario"],
            limiar_memristor=config["limiar"]
        )

        # Erro inicial
        erros_ini = []
        for p in experiencias:
            neo.inferir_estado(p, passos=passos)
            erros_ini.append(sum(abs(e) for e in neo.e1) / dim_sens)
        err_ini_med = sum(erros_ini) / len(erros_ini)

        # Sono
        for _ in range(epocas):
            mem = hipo.fazer_replay()
            mem_r = aplicar_ruido(mem, ruido)
            neo.inferir_estado(mem_r, passos=passos)
            neo.consolidar_pesos()

        # Erro final
        erros_fim = []
        for p in experiencias:
            neo.inferir_estado(p, passos=passos)
            erros_fim.append(sum(abs(e) for e in neo.e1) / dim_sens)
        err_fim_med = sum(erros_fim) / len(erros_fim)
        melhoria = ((err_ini_med - err_fim_med) / err_ini_med) * 100

        # Calcular esparsidade dos pesos efetivos
        w1_eff = neo.obter_w1_efetivo()
        esparsidade = calcular_esparsidade(w1_eff)

        theta_str = f"{config['limiar']:.2f}" if config["limiar"] > 0.0 else "N/A"
        print(f"{config['desc']:<25} | {int(ruido*100):>10}% | {theta_str:<10} | {err_ini_med:<14.5f} | {err_fim_med:<14.5f} | {esparsidade:<10.1f}% | {melhoria:>8.1f}%")
    print("-" * 115)


# ============================================================================
# EXPERIMENTO 3: APRENDIZADO CONTÍNUO E DYNAMIC TASK SWITCHING
# ============================================================================

def rodar_experimento_task_switching() -> None:
    print("\n" + "=" * 125)
    print(" EXPERIMENTO 3: DYNAMIC TASK SWITCHING (APRENDIZADO CONTÍNUO DE LONGO PRAZO)")
    print("=" * 125)
    print(f"{'Configuração do Modelo':<35} | {'Erro Final Task A':<20} | {'Erro Final Task B':<20} | {'Retenção Task A %':<18} | {'Sucesso Task B %':<18}")
    print("-" * 125)

    dim_sens = 10
    dim_int = 6
    dim_abst = 4
    epocas = 400
    passos = 15
    limiar_gaba = 0.32

    # Task A: Padrões focados nos bits pares
    task_a = [
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ]
    # Task B: Padrões focados nos bits ímpares (desvio de distribuição)
    task_b = [
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ]

    modelos = [
        {
            "nome": "Vulnerável (Sem Regulação)",
            "limite_w1": 0.0, "limite_w2": 0.0, "poda": 0.0,
            "usar_gaba": False, "desc": "Sem normalização ou poda"
        },
        {
            "nome": "Regulado (Poda + Normalização L1)",
            "limite_w1": 1.5, "limite_w2": 1.5, "poda": 0.005,
            "usar_gaba": False, "desc": "L1 e poda ativos"
        },
        {
            "nome": "Resiliente Completo (Switch GABA ativo)",
            "limite_w1": 1.5, "limite_w2": 1.5, "poda": 0.005,
            "usar_gaba": True, "desc": "Switch GABAérgico e regulação L1"
        }
    ]

    for mod in modelos:
        random.seed(42)
        hipo = HipocampoRapido(dimensao=dim_sens)
        neo = NeocortexPreditivoProfundo(dim_l1=dim_sens, dim_l2=dim_int, dim_l3=dim_abst, taxa_aprendizado=0.03, usar_meta_pcn=True)

        # --- DIA 1: Apresentação e Consolidação de Task A ---
        hipo.buffer_memorias = []
        for p in task_a:
            hipo.registrar_experiencia(p)
            
        for _ in range(epocas):
            mem = hipo.fazer_replay()
            neo.inferir_estado(mem, passos=passos)
            neo.consolidar_pesos(limite_w1=mod["limite_w1"], limite_w2=mod["limite_w2"])
            
            # Aplicar poda manual
            if mod["poda"] > 0.0:
                for i in range(dim_sens):
                    for j in range(dim_int):
                        if abs(neo.W1[i][j]) < mod["poda"]:
                            neo.W1[i][j] = 0.0
                for j in range(dim_int):
                    for k in range(dim_abst):
                        if abs(neo.W2[j][k]) < mod["poda"]:
                            neo.W2[j][k] = 0.0

        # Avaliar erro inicial pós-A
        err_a_inicial = []
        for p in task_a:
            neo.inferir_estado(p, passos=passos)
            err_a_inicial.append(sum(abs(e) for e in neo.e1) / dim_sens)
        err_a_ini_med = sum(err_a_inicial) / len(err_a_inicial)

        # --- DIA 2: Apresentação e Consolidação de Task B (Mudança de Distribuição) ---
        hipo.buffer_memorias = []
        for p in task_b:
            # Se usar GABA, filtra/gerencia anomalias muito fora do prior
            gaba_ref = neo if mod["usar_gaba"] else None
            hipo.registrar_experiencia(p, neocortex_ref=gaba_ref, limiar_gaba=limiar_gaba)

        for _ in range(epocas):
            mem = hipo.fazer_replay()
            # Se o replay estiver vazio (porque o GABA bloqueou as experiências da Task B)
            if sum(abs(v) for v in mem) == 0.0:
                continue
                
            neo.inferir_estado(mem, passos=passos)
            neo.consolidar_pesos(limite_w1=mod["limite_w1"], limite_w2=mod["limite_w2"])
            
            if mod["poda"] > 0.0:
                for i in range(dim_sens):
                    for j in range(dim_int):
                        if abs(neo.W1[i][j]) < mod["poda"]:
                            neo.W1[i][j] = 0.0
                for j in range(dim_int):
                    for k in range(dim_abst):
                        if abs(neo.W2[j][k]) < mod["poda"]:
                            neo.W2[j][k] = 0.0

        # --- DIA 3: Avaliação de Retenção Final ---
        # Avaliar erros finais em ambas as Tasks
        erros_a_final = []
        for p in task_a:
            neo.inferir_estado(p, passos=passos)
            erros_a_final.append(sum(abs(e) for e in neo.e1) / dim_sens)
        err_a_fim_med = sum(erros_a_final) / len(erros_a_final)

        erros_b_final = []
        for p in task_b:
            neo.inferir_estado(p, passos=passos)
            erros_b_final.append(sum(abs(e) for e in neo.e1) / dim_sens)
        err_b_fim_med = sum(erros_b_final) / len(erros_b_final)

        # Métricas em % (quanto menor o erro, melhor a retenção/sucesso)
        retencao_a = (1.0 - err_a_fim_med) * 100
        sucesso_b = (1.0 - err_b_fim_med) * 100

        print(f"{mod['nome']:<35} | {err_a_fim_med:<20.5f} | {err_b_fim_med:<20.5f} | {retencao_a:>16.1f}% | {sucesso_b:>16.1f}%")
    print("=" * 125)


# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("=" * 125)
    print(" " * 34 + "EXECUÇÃO DE EXPERIMENTOS DE REDE NEUROMÓRFICA - FASE 2 R&D")
    print("=" * 125)
    
    rodar_experimento_profundidade()
    rodar_experimento_memristores()
    rodar_experimento_task_switching()
    
    print("\n" + "=" * 125)
    print("Experimentos concluídos com sucesso.")
    print("=" * 125)
