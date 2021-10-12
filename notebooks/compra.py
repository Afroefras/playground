def total_compra():
    # Pide el número de artículos
    n_articulos = input('Cuántos artículos vas a ingresar? ')

    # Valida si el precio recibido es un número
    error_numero = True
    while error_numero:
        try:
            # Intenta convertirlo a entero. Ej: 1, 2, 3, etc
            n_articulos = int(n_articulos)
            # Si lo logra, rompe el ciclo "while"
            error_numero = False
        except: 
            # De no ser un número, seguirá preguntando por el número de artículos
            n_articulos = input('\nNo es un número válido. Intenta de nuevo!\nCuántos artículos vas a ingresar? ')


    # Listas vacías para acumular artículos y precios
    total_articulos = []
    total_precio = []

    # Ciclo para ingresar cada artículo
    for i in range(n_articulos):
        # Recibe el nombre del artículo
        pedir_nombre = input(f'\nCuál es el nombre del artículo #{i+1}? ')
        pedir_precio = input(f'Cuál es el precio del artículo #{i+1}? ')

        # Valida si el precio recibido es un número
        error_numero = True
        while error_numero:
            try:
                # Intenta convertirlo a flotante. Ej: 1.0, 2.0, 3.0, etc
                pedir_precio = float(pedir_precio)
                # Si lo logra, rompe el ciclo "while"
                error_numero = False
            except: 
                # De no ser un número, seguirá preguntando por el precio
                pedir_precio = input(f'\nNo es un número válido. Intenta de nuevo!\nCuál es el precio del artículo #{i+1}? ')

        # Acumula los nombres y precios de los artículos
        total_articulos.append(pedir_nombre)
        total_precio.append(pedir_precio)

    # Imprime el resultado, cuántos artículos fueron y el precio acumulado
    print(f'\nFueron {len(total_articulos)} artículos con un total de ${sum(total_precio)}')

if __name__ == '__main__':
    total_compra()
