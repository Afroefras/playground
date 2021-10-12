def modyMass():
    # Pide el peso de la usuaria o usuario
    peso = input('Dame tu peso en kilogramos: ')
    # Valida si el precio recibido es un número
    error_numero = True
    while error_numero:
        try:
            # Intenta convertirlo a entero. Ej: 1, 2, 3, etc
            peso = float(peso)
            # Si lo logra, rompe el ciclo "while"
            error_numero = False
        except: 
            # De no ser un número, seguirá preguntando por el precio
            peso = input('\nNo es un número válido. Intenta de nuevo!\nDame tu peso en kilogramos: ')
        
    # Pide la altura de la usuaria o usuario
    altura = input('Dame tu estatura en metros: ')
    # Valida si el precio recibido es un número
    error_numero = True
    while error_numero:
        try:
            # Intenta convertirlo a entero. Ej: 1, 2, 3, etc
            altura = float(altura)
            # Si lo logra, rompe el ciclo "while"
            error_numero = False
        except: 
            # De no ser un número, seguirá preguntando por el precio
            altura = input('\nNo es un número válido. Intenta de nuevo!\nDame tu estatura en metros: ')

    # Calcula el índice de masa corporal
    bmi = peso/altura**2
    # Imprime el resultado
    print(f'\nTu BMI es de {round(bmi,2)}')

if __name__ == '__main__':
    modyMass()
