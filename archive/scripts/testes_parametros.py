"""
Script de testes e varredura de parâmetros (Parameter Sweep) Avançado e Denso.

Este script realiza múltiplos experimentos automatizados na simulação do diálogo
hipocampo-neocórtex, testando:
1. Hiperparâmetros clássicos de curto prazo (1 dia).
2. Mecanismos homeostáticos: Decaimento (Downscaling), Poda (Pruning) e Normalização L1.
3. Experimento de Longo Prazo / Aprendizado Contínuo (Multi-dia, 3 dias):
   - Avalia a retenção de memórias antigas vs. consolidação de novas memórias.
   - Revela o paradoxo da homeostase e o esquecimento catastrófico.

Imports modulares de `simulacao_dialogo_hipocampo.py`.
"""

import random
from typing import Any, Dict, List, Tuple

# Importar as classes originais do nosso arquivo principal
from simulacao_dialogo_hipocampo import HipocampoRapido, NeocortexPreditivo


def aplicar_ruido(padrao: List[float], taxa_ruido: float) -> List[float]:
    """
    Simula falhas sinápticas no replay injetando ruído de inversão (bit flip)
    com base em uma taxa de probabilidade.
    """
    if taxa_ruido <= 0.0:
        return padrao
    
    # Inverte os valores binários (1.0 vira 0.0, e vice-versa) sob a taxa_ruido
    return [
        1.0 - val if random.random() < taxa_ruido else val
        for val in padrao
    ]


def gerar_experiencias_dia(dim_sensorial: int, qtd_mems: int) -> List[List[float]]:
    """Gera experiências binárias aleatórias estáveis baseadas na semente fixa."""
    experiencias = []
    tentativas = 0
    
    while len(experiencias) < qtd_mems and tentativas < 200:
        tentativas += 1
        # Gera vetor binário aleatório
        vetor = [float(random.choice([0, 1])) for _ in range(dim_sensorial)]
        
        # Evita duplicatas ou vetores vazios
        if vetor not in experiencias and sum(vetor) > 0:
            experiencias.append(vetor)
            
    # Fallback se a geração falhar
    if len(experiencias) < qtd_mems:
        experiencias = [[1.0 if (idx + i) % 2 == 0 else 0.0 for idx in range(dim_sensorial)] for i in range(qtd_mems)]
        
    return experiencias


def rodar_experimento(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inicializa a simulação e roda um cenário específico com os parâmetros fornecidos.
    Suporta múltiplos dias, ruído, decaimento homeostático, normalização L1 e poda sináptica.
    Retorna um dicionário com os erros e melhorias de forma estruturada.
    """
    # Fixar a semente antes de cada rodada para garantir comparação estatística justa
    random.seed(42)
    
    dim_sens = config["dim_sens"]
    dim_abst = config["dim_abst"]
    alpha = config["alpha"]
    epocas = config["epocas"]
    passos = config["passos"]
    qtd_mems_dia = config["qtd_mems"]
    ruido = config["ruido"]
    
    # Hiperparâmetros homeostáticos
    decaimento = config.get("decaimento", 0.0)
    poda = config.get("poda", 0.0)
    capacidade = config.get("capacidade", 0.0)  # Capacidade L1 para normalização sináptica
    
    dias = config.get("dias", 1)
    
    # 1. Gerar os dados experimentais para todos os dias
    total_mems = qtd_mems_dia * dias
    experiencias = gerar_experiencias_dia(dim_sens, total_mems)
    
    # 2. Inicializar as redes
    hipocampo = HipocampoRapido(dimensao=dim_sens)
    neocortex = NeocortexPreditivo(dim_sensorial=dim_sens, dim_abstrata=dim_abst, taxa_aprendizado=alpha)
    
    # 3. Medir erro pré-sono (Neocórtex sem treinamento) para todos os padrões
    erros_pre = []
    for exp in experiencias:
        neocortex.inferir_estado(exp, passos=passos)
        erro = sum(abs(e) for e in neocortex.e1) / dim_sens
        erros_pre.append(erro)
    erro_inicial_medio = sum(erros_pre) / len(erros_pre)
    
    # Calcular erros pré-sono médios por "dia"
    erros_pre_dia = []
    for d in range(dias):
        start_idx = d * qtd_mems_dia
        end_idx = (d + 1) * qtd_mems_dia
        erros_pre_dia.append(sum(erros_pre[start_idx:end_idx]) / qtd_mems_dia)
        
    # 4. Loop Multi-dia (Consolidação sequencial de memórias)
    for d in range(dias):
        start_idx = d * qtd_mems_dia
        end_idx = (d + 1) * qtd_mems_dia
        exp_dia = experiencias[start_idx:end_idx]
        
        # Limpa o buffer de memórias episódicas do Hipocampo no início do dia
        # (simulando a transição do dia e expiração das memórias de curto prazo)
        hipocampo.buffer_memorias = []
        
        # Fase de Vigília do dia 'd' (Gravação rápida no Hipocampo)
        for exp in exp_dia:
            hipocampo.registrar_experiencia(exp)
            
        # Fase de Sono do dia 'd' (Replay + Predictive Coding no Neocórtex)
        for _ in range(epocas):
            memoria = hipocampo.fazer_replay()
            memoria_ruidosa = aplicar_ruido(memoria, ruido)
            
            # Neocórtex infere e aprende localmente
            neocortex.inferir_estado(memoria_ruidosa, passos=passos)
            neocortex.consolidar_pesos()
            
            # --- Dinâmicas Homeostáticas Corticais ---
            # A. Decaimento Homeostático (Downscaling global lento)
            if decaimento > 0.0:
                for i in range(neocortex.dim_sensorial):
                    for j in range(neocortex.dim_abstrata):
                        neocortex.W[i][j] *= (1.0 - decaimento)
            
            # B. Normalização Sináptica L1 (Homeostase Cooperativa)
            # Mantém a soma dos pesos absolutos de cada neurônio sensorial limitada
            if capacidade > 0.0:
                for i in range(neocortex.dim_sensorial):
                    soma_pesos = sum(abs(neocortex.W[i][j]) for j in range(neocortex.dim_abstrata))
                    if soma_pesos > capacidade:
                        fator_escala = capacidade / soma_pesos
                        for j in range(neocortex.dim_abstrata):
                            neocortex.W[i][j] *= fator_escala
            
            # C. Poda Sináptica (Pruning de conexões fracas)
            if poda > 0.0:
                for i in range(neocortex.dim_sensorial):
                    for j in range(neocortex.dim_abstrata):
                        if abs(neocortex.W[i][j]) < poda:
                            neocortex.W[i][j] = 0.0
                            
    # 5. Medir erro pós-sono (Consolidação concluída) para todos os padrões
    erros_pos = []
    for exp in experiencias:
        neocortex.inferir_estado(exp, passos=passos)
        erro = sum(abs(e) for e in neocortex.e1) / dim_sens
        erros_pos.append(erro)
    erro_final_medio = sum(erros_pos) / len(erros_pos)
    
    # Calcular erros finais e melhoria por "dia"
    erros_pos_dia = []
    melhoria_dia = []
    for d in range(dias):
        start_idx = d * qtd_mems_dia
        end_idx = (d + 1) * qtd_mems_dia
        pre = erros_pre_dia[d]
        pos = sum(erros_pos[start_idx:end_idx]) / qtd_mems_dia
        erros_pos_dia.append(pos)
        mel_d = ((pre - pos) / pre) * 100 if pre > 0 else 0.0
        melhoria_dia.append(mel_d)
        
    melhoria_geral = ((erro_inicial_medio - erro_final_medio) / erro_inicial_medio) * 100
    
    return {
        "erro_ini": erro_inicial_medio,
        "erro_fim": erro_final_medio,
        "melhoria": melhoria_geral,
        "melhoria_dia": melhoria_dia,
        "erros_pre_dia": erros_pre_dia,
        "erros_pos_dia": erros_pos_dia
    }


def main() -> None:
    # -------------------------------------------------------------------------
    # TABELA 1: Varredura Clássica de Curto Prazo (1 Dia)
    # -------------------------------------------------------------------------
    cenarios_1dia = [
        # --- Cenários Clássicos ---
        {"nome": "Baseline (Original)",   "dim_sens": 5, "dim_abst": 3, "alpha": 0.030, "epocas": 600, "passos": 10, "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.00, "poda": 0.00},
        {"nome": "Alpha muito baixo",     "dim_sens": 5, "dim_abst": 3, "alpha": 0.005, "epocas": 600, "passos": 10, "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.00, "poda": 0.00},
        {"nome": "Alpha alto",            "dim_sens": 5, "dim_abst": 3, "alpha": 0.100, "epocas": 600, "passos": 10, "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.00, "poda": 0.00},
        {"nome": "Poucas épocas (100)",   "dim_sens": 5, "dim_abst": 3, "alpha": 0.030, "epocas": 100, "passos": 10, "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.00, "poda": 0.00},
        {"nome": "Muitas épocas (1200)",  "dim_sens": 5, "dim_abst": 3, "alpha": 0.030, "epocas": 1200, "passos": 10, "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.00, "poda": 0.00},
        {"nome": "Poucos passos (2)",     "dim_sens": 5, "dim_abst": 3, "alpha": 0.030, "epocas": 600, "passos": 2,  "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.00, "poda": 0.00},
        {"nome": "Muitos passos (30)",    "dim_sens": 5, "dim_abst": 3, "alpha": 0.030, "epocas": 600, "passos": 30, "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.00, "poda": 0.00},
        {"nome": "Alta compressão (10->2)","dim_sens": 10, "dim_abst": 2, "alpha": 0.030, "epocas": 800, "passos": 15, "qtd_mems": 3, "ruido": 0.00, "decaimento": 0.00, "poda": 0.00},
        {"nome": "Compressão mod. (10->6)","dim_sens": 10, "dim_abst": 6, "alpha": 0.030, "epocas": 800, "passos": 15, "qtd_mems": 3, "ruido": 0.00, "decaimento": 0.00, "poda": 0.00},
        
        # --- Sobrecarga de Memória (Interferência) ---
        {"nome": "Sobrecarga (6 mems - Sem Reg)", "dim_sens": 10, "dim_abst": 6, "alpha": 0.030, "epocas": 800, "passos": 15, "qtd_mems": 6, "ruido": 0.00, "decaimento": 0.00, "poda": 0.00},
        {"nome": "Sobrecarga + Homeostase Leve",  "dim_sens": 10, "dim_abst": 6, "alpha": 0.030, "epocas": 800, "passos": 15, "qtd_mems": 6, "ruido": 0.00, "decaimento": 0.002, "poda": 0.00},
        {"nome": "Sobrecarga + Homeostase + Poda","dim_sens": 10, "dim_abst": 6, "alpha": 0.030, "epocas": 800, "passos": 15, "qtd_mems": 6, "ruido": 0.00, "decaimento": 0.002, "poda": 0.005},
        {"nome": "Sobrecarga + Homeostase Agr.  ","dim_sens": 10, "dim_abst": 6, "alpha": 0.030, "epocas": 800, "passos": 15, "qtd_mems": 6, "ruido": 0.00, "decaimento": 0.010, "poda": 0.020},
        
        # --- Replay com Ruído + Filtro de Homeostase ---
        {"nome": "Replay 20% Ruído (Sem Reg)",    "dim_sens": 8, "dim_abst": 4, "alpha": 0.030, "epocas": 800, "passos": 12, "qtd_mems": 3, "ruido": 0.20, "decaimento": 0.00, "poda": 0.00},
        {"nome": "Replay 20% Ruído + Regulador",  "dim_sens": 8, "dim_abst": 4, "alpha": 0.030, "epocas": 800, "passos": 12, "qtd_mems": 3, "ruido": 0.20, "decaimento": 0.003, "poda": 0.010},
    ]

    largura_tabela = 135
    print("=" * largura_tabela)
    print(" " * 44 + "TABELA 1: VARREDURA CLÁSSICA DE CURTO PRAZO (1 DIA)")
    print("=" * largura_tabela)
    print(f"{'Cenário':<32} | {'Dim.S/A':<7} | {'Alpha':<6} | {'Épocas':<6} | {'Passos':<6} | {'Mems':<4} | {'Ruído':<5} | {'Decai.':<6} | {'Poda':<5} | {'Erro Ini.':<9} | {'Erro Fim':<9} | {'Melhoria':<8}")
    print("-" * largura_tabela)
    
    for c in cenarios_1dia:
        res = rodar_experimento(c)
        dim_str = f"{c['dim_sens']}/{c['dim_abst']}"
        ruido_pct = f"{int(c['ruido'] * 100)}%"
        decai_str = f"{c['decaimento']:.3f}" if c['decaimento'] > 0 else "0.000"
        poda_str = f"{c['poda']:.3f}" if c['poda'] > 0 else "0.000"
        
        print(f"{c['nome']:<32} | {dim_str:<7} | {c['alpha']:<6.3f} | {c['epocas']:<6} | {c['passos']:<6} | {c['qtd_mems']:<4} | {ruido_pct:<5} | {decai_str:<6} | {poda_str:<5} | {res['erro_ini']:<9.5f} | {res['erro_fim']:<9.5f} | {res['melhoria']:>6.1f}%")
    print("=" * largura_tabela + "\n\n")

    # -------------------------------------------------------------------------
    # TABELA 2: Experimento de Aprendizado Contínuo de Longo Prazo (Multi-dia)
    # -------------------------------------------------------------------------
    # Simulamos 3 dias consecutivos. A cada dia o hipocampo recebe 2 novas experiências.
    # Avaliamos a melhoria geral, bem como a melhoria específica do Dia 1 (memórias antigas)
    # e do Dia 3 (memórias novas) para ver o esquecimento catastrófico.
    cenarios_multidia = [
        {"nome": "Multi-dia Baseline (Sem Reg)",  "dim_sens": 10, "dim_abst": 6, "alpha": 0.030, "epocas": 500, "passos": 12, "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.00,  "poda": 0.00,  "capacidade": 0.0, "dias": 3},
        {"nome": "Multi-dia + Decaimento Leve",   "dim_sens": 10, "dim_abst": 6, "alpha": 0.030, "epocas": 500, "passos": 12, "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.002, "poda": 0.00,  "capacidade": 0.0, "dias": 3},
        {"nome": "Multi-dia + Poda Leve",         "dim_sens": 10, "dim_abst": 6, "alpha": 0.030, "epocas": 500, "passos": 12, "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.00,  "poda": 0.010, "capacidade": 0.0, "dias": 3},
        {"nome": "Multi-dia + Normalização L1",   "dim_sens": 10, "dim_abst": 6, "alpha": 0.030, "epocas": 500, "passos": 12, "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.00,  "poda": 0.00,  "capacidade": 1.5, "dias": 3},
        {"nome": "Multi-dia + Normalização + Poda","dim_sens": 10, "dim_abst": 6, "alpha": 0.030, "epocas": 500, "passos": 12, "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.00,  "poda": 0.010, "capacidade": 1.5, "dias": 3},
        {"nome": "Multi-dia + Tudo (Reg. Densa)", "dim_sens": 10, "dim_abst": 6, "alpha": 0.030, "epocas": 500, "passos": 12, "qtd_mems": 2, "ruido": 0.00, "decaimento": 0.001, "poda": 0.005, "capacidade": 1.5, "dias": 3},
        {"nome": "Multi-dia + Tudo + Ruído 15%",  "dim_sens": 10, "dim_abst": 6, "alpha": 0.030, "epocas": 500, "passos": 12, "qtd_mems": 2, "ruido": 0.15, "decaimento": 0.001, "poda": 0.005, "capacidade": 1.5, "dias": 3},
    ]

    print("=" * largura_tabela)
    print(" " * 34 + "TABELA 2: APRENDIZADO CONTÍNUO E EVOLUÇÃO MULTI-DIA (3 DIAS - 6 MEMÓRIAS)")
    print("=" * largura_tabela)
    print(f"{'Cenário':<30} | {'Decai.':<6} | {'Poda':<5} | {'L1 Cap':<6} | {'Ruído':<5} | {'Erro Ini.':<9} | {'Erro Fim':<9} | {'Melhoria G.':<11} | {'Mel. Dia 1':<10} | {'Mel. Dia 2':<10} | {'Mel. Dia 3':<10}")
    print("-" * largura_tabela)

    for c in cenarios_multidia:
        res = rodar_experimento(c)
        decai_str = f"{c['decaimento']:.3f}" if c['decaimento'] > 0 else "0.000"
        poda_str = f"{c['poda']:.3f}" if c['poda'] > 0 else "0.000"
        cap_str = f"{c['capacidade']:.1f}" if c['capacidade'] > 0 else "0.0"
        ruido_pct = f"{int(c['ruido'] * 100)}%"
        
        m_g = f"{res['melhoria']:>10.1f}%"
        m_d1 = f"{res['melhoria_dia'][0]:>9.1f}%"
        m_d2 = f"{res['melhoria_dia'][1]:>9.1f}%"
        m_d3 = f"{res['melhoria_dia'][2]:>9.1f}%"
        
        print(f"{c['nome']:<30} | {decai_str:<6} | {poda_str:<5} | {cap_str:<6} | {ruido_pct:<5} | {res['erro_ini']:<9.5f} | {res['erro_fim']:<9.5f} | {m_g} | {m_d1} | {m_d2} | {m_d3}")
    print("=" * largura_tabela)
    print("Simulações estatísticas de aprendizado contínuo multi-dia executadas com sucesso.")


if __name__ == "__main__":
    main()
