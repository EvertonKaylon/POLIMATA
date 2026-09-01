# =====================================================================
# ⚡ TESTE PRÁTICO: IMPLEMENTANDO UM PERCEPTRON SIMPLES DO ZERO
# =====================================================================
# Este script implementa o Perceptron de Rosenblatt (1958) em Python
# puro (sem dependências como numpy ou pytorch).
# 
# Objetivo: Ensinar passo a passo como o neurônio aprende a lógica AND.
# =====================================================================

class PerceptronSimples:
    def __init__(self, numero_entradas, taxa_aprendizado=0.1):
        # 1. Inicializamos os pesos com 0 (um peso para cada entrada)
        # Na biologia, isso seria a "força" inicial de cada sinapse (neutra)
        self.pesos = [0.0] * numero_entradas
        
        # 2. Inicializamos o bias (viés) com 0.0
        # Na biologia, isso regula o limiar de disparo (threshold) no cone axonal
        self.bias = 0.0
        
        # 3. Taxa de aprendizado (learning rate, eta η)
        # Regula o tamanho do ajuste que faremos nos pesos a cada erro
        self.taxa_ap = taxa_aprendizado

    def funcao_ativacao(self, z):
        """Função Degrau (Step Function) - O 'Tudo ou Nada' de Cajal"""
        # Se a soma ponderada z for maior ou igual a zero, o neurônio dispara (1)
        # Caso contrário, permanece em repouso (0)
        return 1 if z >= 0 else 0

    def predizer(self, entradas):
        """Forward Pass: Integração no Soma + Disparo no Axônio"""
        # z = (x1 * w1) + (x2 * w2) + ... + bias
        # Fazemos a soma ponderada de forma manual e simples:
        soma_ponderada = 0.0
        for x, w in zip(entradas, self.pesos):
            soma_ponderada += x * w
        
        soma_ponderada += self.bias
        
        # Passa o sinal integrado pela função de ativação
        return self.funcao_ativacao(soma_ponderada)

    def treinar(self, dados_treino, rotulos_esperados, epocas=10):
        """Loop de Treinamento: Hebbian Learning / Ajuste por Erro"""
        print("Iniciando o treinamento do Perceptron...")
        print("-" * 50)
        
        for epoca in range(1, epocas + 1):
            total_erros = 0
            
            # Passamos por cada exemplo do nosso dataset
            for entradas, esperado in zip(dados_treino, rotulos_esperados):
                # 1. Faz o neurônio tentar adivinhar a saída (Forward Pass)
                obtido = self.predizer(entradas)
                
                # 2. Calcula o erro
                # Se esperado = 1 e obtido = 0 -> erro = 1 (Falso Negativo)
                # Se esperado = 0 e obtido = 1 -> erro = -1 (Falso Positivo)
                # Se ambos forem iguais -> erro = 0 (Acerto!)
                erro = esperado - obtido
                
                if erro != 0:
                    total_erros += abs(erro)
                    
                    # 3. Ajustamos os pesos: peso_novo = peso_antigo + (η * erro * entrada)
                    for i in range(len(self.pesos)):
                        ajuste = self.taxa_ap * erro * entradas[i]
                        self.pesos[i] += ajuste
                    
                    # 4. Ajustamos o bias: bias_novo = bias_antigo + (η * erro)
                    self.bias += self.taxa_ap * erro
                    
            print(f"Epoca {epoca} | Pesos: {[round(w, 2) for w in self.pesos]} | Bias: {round(self.bias, 2)} | Erros no ciclo: {total_erros}")
            
            # Critério de parada: se não houver mais erros, o neurônio convergiu!
            if total_erros == 0:
                print(f"\nO Perceptron convergiu com sucesso na Epoca {epoca}!")
                break
        
        print("-" * 50)


# =====================================================================
# 🧪 EXECUÇÃO DO TESTE: PORTA LÓGICA AND
# =====================================================================
# A porta AND só resulta em 1 se ambas as entradas forem 1.
# 
# Tabela Verdade do AND:
# x1  x2  |  Esperado (y)
# 0   0   |  0
# 0   1   |  0
# 1   0   |  0
# 1   1   |  1
# =====================================================================

if __name__ == "__main__":
    # Dados de entrada (X) para a porta XOR
    X = [
        [1, 1],
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
    ]
    
    # Rótulos esperados (y) para a porta XOR
    # Note que a saída é 1 apenas se as entradas forem diferentes!
    y = [1, 0, 0, 0]

    # Criamos o perceptron com 2 entradas (x1 e x2)
    modelo = PerceptronSimples(numero_entradas=2, taxa_aprendizado=0.1)
    
    # Treinamos por até 20 épocas para ver a oscilação infinita
    modelo.treinar(X, y, epocas=20)
    
    # Testamos o resultado final para mostrar a falha
    print("\nResultados apos o treinamento do XOR (Observe as falhas!):")
    for entradas, esperado in zip(X, y):
        resultado = modelo.predizer(entradas)
        status = "Correto" if resultado == esperado else "ERRADO"
        print(f"Entrada: {entradas} -> Esperado: {esperado} | Saida: {resultado} [{status}]")
