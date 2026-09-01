numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def media(numeros):
    soma = 0
    for numero in numeros:
        soma+= numero
    resultado = soma / len(numeros)
    return resultado

print("A média entre ", numeros, " é ", media(numeros))

    