# =====================================================================
# 🧠 REDE NEURAL MULTICAMADAS (MLP) DO ZERO: RESOLVENDO O XOR
# =====================================================================
# Este script implementa uma rede neural com 1 camada oculta usando
# Backpropagation em Python puro (sem numpy ou outras bibliotecas).
#
# Arquitetura da Rede (2-2-1):
#   Entradas (x1, x2) -> Camada Oculta (h1, h2) -> Saida (y)
# =====================================================================

import math
import random

# Definir semente para que os testes sejam reproduzíveis
random.seed(15)

# 1. Função de ativação Sigmoide e sua derivada
def sigmoide(z):
    # Evita overflow numérico limitando z entre -500 e 500
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

def sigmoide_derivada(a):
    # A derivada da sigmoide expressa em função da própria saída 'a' da sigmoide
    # Se a = sigmoide(z), então a derivada é a * (1 - a)
    return a * (1.0 - a)

class RedeNeuralXOR:
    def __init__(self, taxa_aprendizado=0.5):
        self.lr = taxa_aprendizado
        
        # Inicialização dos pesos da Camada Oculta (2 entradas -> 2 neurônios ocultos)
        # w_hi_j representa o peso da entrada i para o neurônio oculto j
        self.w_h1_1 = random.uniform(-1.0, 1.0)
        self.w_h1_2 = random.uniform(-1.0, 1.0) # Pesos do neurônio oculto 1
        self.b_h1 = 0.0                          # Bias do neurônio oculto 1
        
        self.w_h2_1 = random.uniform(-1.0, 1.0)
        self.w_h2_2 = random.uniform(-1.0, 1.0) # Pesos do neurônio oculto 2
        self.b_h2 = 0.0                          # Bias do neurônio oculto 2
        
        # Inicialização dos pesos da Camada de Saída (2 neurônios ocultos -> 1 saída)
        self.w_o_1 = random.uniform(-1.0, 1.0)
        self.w_o_2 = random.uniform(-1.0, 1.0)  # Pesos do neurônio de saída
        self.b_o = 0.0                           # Bias do neurônio de saída

    def forward(self, x1, x2):
        """Passo de propagação para frente (Forward Pass)"""
        # --- CAMADA OCULTA ---
        # Neurônio Oculto 1: soma ponderada + ativação sigmoide
        self.z_h1 = (x1 * self.w_h1_1) + (x2 * self.w_h1_2) + self.b_h1
        self.a_h1 = sigmoide(self.z_h1)
        
        # Neurônio Oculto 2: soma ponderada + ativação sigmoide
        self.z_h2 = (x1 * self.w_h2_1) + (x2 * self.w_h2_2) + self.b_h2
        self.a_h2 = sigmoide(self.z_h2)
        
        # --- CAMADA DE SAÍDA ---
        # Neurônio de Saída: soma ponderada usando as ativações da camada oculta
        self.z_o = (self.a_h1 * self.w_o_1) + (self.a_h2 * self.w_o_2) + self.b_o
        self.a_o = sigmoide(self.z_o)
        
        return self.a_o

    def backward(self, x1, x2, esperado):
        """Passo de retropropagação do erro (Backpropagation)"""
        # 1. Calcular o erro na saída
        obtido = self.a_o
        erro_saida = esperado - obtido
        
        # 2. Gradiente do neurônio de saída (erro ponderado pela derivada da sigmoide)
        # Delta indica a direção e magnitude do ajuste
        delta_o = erro_saida * sigmoide_derivada(obtido)
        
        # 3. Gradientes da camada oculta (usando a Regra da Cadeia)
        # O erro da saída é "distribuído" de volta baseado no peso da conexão
        delta_h1 = delta_o * self.w_o_1 * sigmoide_derivada(self.a_h1)
        delta_h2 = delta_o * self.w_o_2 * sigmoide_derivada(self.a_h2)
        
        # 4. Ajustar pesos da Camada de Saída
        self.w_o_1 += self.lr * delta_o * self.a_h1
        self.w_o_2 += self.lr * delta_o * self.a_h2
        self.b_o += self.lr * delta_o
        
        # 5. Ajustar pesos da Camada Oculta
        self.w_h1_1 += self.lr * delta_h1 * x1
        self.w_h1_2 += self.lr * delta_h1 * x2
        self.b_h1 += self.lr * delta_h1
        
        self.w_h2_1 += self.lr * delta_h2 * x1
        self.w_h2_2 += self.lr * delta_h2 * x2
        self.b_h2 += self.lr * delta_h2
        
        # Retorna o erro quadrático desse exemplo para monitoramento
        return 0.5 * (erro_saida ** 2)

    def treinar(self, X, y, epocas=20000):
        print("Iniciando o treinamento da Rede Neural MLP (XOR)...")
        print("-" * 60)
        
        for epoca in range(1, epocas + 1):
            erro_acumulado = 0.0
            
            # Treinar com cada exemplo do dataset
            for (x1, x2), esperado in zip(X, y):
                self.forward(x1, x2)
                erro = self.backward(x1, x2, esperado)
                erro_acumulado += erro
            
            # Print do progresso a cada 2000 épocas
            if epoca % 2000 == 0 or epoca == 1:
                erro_medio = erro_acumulado / len(X)
                print(f"Epoca {epoca:5d} | Erro Medio: {erro_medio:.6f}")
                
            # Criterio de parada precoce se o erro médio for muito baixo
            if erro_acumulado < 0.005:
                print(f"\nConvergiu com sucesso na Epoca {epoca}!")
                break
                
        print("-" * 60)


if __name__ == "__main__":
    # Dados da tabela verdade do XOR real
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    y = [0, 1, 1, 0]
    
    # Instanciar a rede neural
    rede = RedeNeuralXOR(taxa_aprendizado=0.5)
    
    # Treinar por até 20.000 épocas
    rede.treinar(X, y, epocas=20000)
    
    # Testar o resultado final após o treinamento
    print("\nResultados finais apos o treinamento do XOR:")
    for x1, x2 in X:
        predicao = rede.forward(x1, x2)
        resultado_binario = 1 if predicao >= 0.5 else 0
        esperado = 1 if (x1 != x2) else 0
        status = "Correto" if resultado_binario == esperado else "ERRADO"
        
        print(f"Entrada: [{x1}, {x2}] -> Probabilidade: {predicao:.4f} -> Saida: {resultado_binario} [{status}]")
