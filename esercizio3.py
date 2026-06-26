import numpy as np
from tabulate import tabulate

APICE_MENO_UNO = "\u207B\u00B9"
LAMBDA = "\u03BB"

# Definizione della matrice A
A = np.array([
    [4.0, 1.0, 0.0],
    [1.0, 3.0, 1.0],
    [0.0, 1.0, 2.0]
])
print("\n")
print("="*40)
print("1. Calcolo autovalori e autovettori")
print("="*40)
print("\n")

# eig restituisce gli autovalori nell'array "autovalori" e gli autovettori (per colonna) nella matrice V
autovalori, V = np.linalg.eig(A)

print(f"Autovalori ({LAMBDA}):")
print(tabulate([autovalori],tablefmt="fancy_grid"))
print("\nMatrice degli autovettori (V):")
print(tabulate(V,tablefmt="fancy_grid"))

print("\n")
print("="*40)
print(f"2. Verifica numerica A*v_i = {LAMBDA}_i*v_i")
print("="*40)
print("\n")
for i in range(len(autovalori)):
    # Estraggo l'i-esimo autovalore e la corrispondente colonna di V
    lambda_i = autovalori[i]
    v_i = V[:, i]
    
    # Calcolo i due lati dell'equazione
    lato_sinistro = A @ v_i           # Prodotto matrice-vettore
    lato_destro = lambda_i * v_i      # Prodotto scalare-vettore
    
    # Verifico se sono uguali (usando allclose per tollerare errori di arrotondamento macchina)
    verifica = np.allclose(lato_sinistro, lato_destro)
    print(f"Autovettore {i+1}: Verifica A*v_{i+1} = {LAMBDA}_{i+1}*v_{i+1} -> {verifica}")

print("\n")
print("="*40)
print("3. Verifica dell'indipendenza lineare")
print("="*40) 
print("\n")


# Il rango della matrice degli autovettori deve essere massimo (uguale a 3)
rango_V = np.linalg.matrix_rank(V)
indipendenti = (rango_V == A.shape[0])
print(f"Rango della matrice V= {rango_V}")
print(f"Gli autovettori sono linearmente indipendenti? {'sì' if indipendenti else 'no'}")

print("\n")
print("="*50)
print(f"4. Costruzione di V, D e verifica A = V @ D @ V{APICE_MENO_UNO}")
print("="*50)
print("\n")


# Creo la matrice diagonale D con gli autovalori sulla diagonale principale
D = np.diag(autovalori)

# Calcolo l'inversa di V
V_inv = np.linalg.inv(V)

# Ricostruisco A e verifico
A_ricostruita = V @ D @ V_inv

print(f"Matrice V * D * V{APICE_MENO_UNO} calcolata:")

print(tabulate(A_ricostruita,tablefmt="fancy_grid",floatfmt=".6e"))
verifica_A = np.allclose(A, A_ricostruita)
print(f"La relazione A = V @ D @ V{APICE_MENO_UNO} è verificata numericamente? {'sì' if verifica_A else 'no'}")


print("\n")
print("="*40)
print("5. Condizionamento della matrice V")
print("="*40)
# Calcolo del numero di condizionamento (in norma 2 di default)
cond_V = np.linalg.cond(V)
print(f"Numero di condizionamento di V: {cond_V:.6f}")

print("\n")
print("-- Programma terminato --")