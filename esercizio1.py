import numpy as np
import scipy.linalg as las
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons # libreria per implementare i bottoni
from tabulate import tabulate # utilizzato per la formattazione della tabella finale

# ====================================================================
# Punto 1: Definizione della funzione e dei punti equispaziati
# ====================================================================

def f(x):
    # La funzione f(x) = e^(-x) * cos(2x) da interpolare
    return np.exp(-x) * np.cos(2*x)

# vado a creare l'array di 9 punti equispaziati nell'intervallo [-2, 2]
n_punti = 9
x_eq = np.linspace(-2, 2, n_punti)  # mi restituisce l'array di punti 
print("Nodi equidistanti: "+ str(x_eq)) # stampo i punti sul terminale 
y_eq = f(x_eq) 

# ====================================================================
# Punto 2 & 3: Base monomiale, Matrice di Vandermonde e Fattorizzazione LU
# ====================================================================

# Costruisco la Matrice di Vandermonde per i punti equispaziati.
# increasing=True genera le potenze in ordine crescente: 1, x, x^2, ..., x^8
V_eq = np.vander(x_eq, increasing=True)

# Esegue la fattorizzazione LU con pivot totale di una matrice quadrata A.
# Ritorna P, L, U, Q tali che P @ A @ Q = L @ U 
# P = matrici di permutazione (riga)
# Q = matrici di permutazione (colonna)
def fatt_lu_pivot_totale(A):

    n = A.shape[0]
    # Inizializzo le matrici di permutazione come matrici identità
    P = np.eye(n)
    Q = np.eye(n)
    
    # Lavoro su una copia per non modificare la matrice originale
    A_lu = A.copy().astype(float)
    
    for k in range(n - 1):
        # 1. RICERCA DEL PIVOT TOTALE
        # Cerco il massimo valore assoluto nella sottomatrice A_lu[k:n, k:n]
        sub_matrix = np.abs(A_lu[k:n, k:n])
        # r_rel e c_rel sono gli indici del massimo relativi alla sottomatrice
        r_rel, c_rel = np.unravel_index(np.argmax(sub_matrix), sub_matrix.shape)
        
        # Riporto gli indici assoluti nella matrice intera
        r = r_rel + k
        c = c_rel + k
        
        # 2. SCAMBIO RIGHE (nella matrice A, e tracciamento in P)
        if r != k:
            A_lu[[k, r], :] = A_lu[[r, k], :]
            P[[k, r], :] = P[[r, k], :]
            
        # 3. SCAMBIO COLONNE (nella matrice A, e tracciamento in Q)
        if c != k:
            A_lu[:, [k, c]] = A_lu[:, [c, k]]
            Q[:, [k, c]] = Q[:, [c, k]]
            
        # Controllo tolleranza per evitare divisioni per zero
        if np.abs(A_lu[k, k]) < 1e-15:
            raise ValueError("La matrice ha un elemento pivotale quasi nullo. Sistema singolare o malcondizionato.")
            
        # 4. CALCOLO DEI MOLTIPLICATORI E AGGIORNAMENTO matrice (Algoritmo di Gauss)
        # Memorizzo moltiplicatori nella parte inferiore di A_lu
        A_lu[k+1:n, k] = A_lu[k+1:n, k] / A_lu[k, k] 
        # Aggiorniamo la sottomatrice restante utilizzando il prodotto esterno
        A_lu[k+1:n, k+1:n] -= np.outer(A_lu[k+1:n, k], A_lu[k, k+1:n])
        
    # Estraggo L e U dalla matrice combinata A_lu
    # L ha i moltiplicatori sotto la diagonale e 1 sulla diagonale
    L = np.tril(A_lu, -1) + np.eye(n)
    # U ha gli elementi sulla diagonale e sopra
    U = np.triu(A_lu)
    
    return P, L, U, Q

# 1. Calcolo la fattorizzazione con il pivot totale sulla matrice di Vandermonde
P_eq, L_eq, U_eq, Q_eq = fatt_lu_pivot_totale(V_eq)

# 2. Risoluzione del sistema strutturato a passi:
# Passo a) Moltiplichiamo il termine noto per P: y_perm = P * y
# (P è la matrice di permutazione di RIGA, costruita scambiando le righe
# dell'identità in parallelo a quelle di V; quindi P @ y applica al termine
# noto le stesse permutazioni di riga subite dalla matrice del sistema.)
y_perm_eq = P_eq @ y_eq

# Passo b) Sostituzione in avanti per risolvere L * z = y_perm
z_eq = las.solve_triangular(L_eq, y_perm_eq, lower=True)

# Passo c) Sostituzione all'indietro per risolvere U * a_tilde = z
a_tilde = las.solve_triangular(U_eq, z_eq)

# Passo d) Poiché L * U * (Q.T * a) = P * y, allora a_tilde = Q.T * a.
a_eq = Q_eq @ a_tilde




# ====================================================================
# Punto 4: Nodi di Chebyshev e relativo polinomio
# ====================================================================

a, b = -2, 2
k = np.arange(n_punti)
# Formula per i Nodi di Chebyshev nell'intervallo [a, b] 
x_cheb = (a + b + (b - a) * np.cos((2 * k + 1) * np.pi / (2 * n_punti))) / 2
y_cheb = f(x_cheb)

# Matrice di Vandermonde per i nodi di Chebyshev e risoluzione sistema (stesso iter di prima)
V_cheb = np.vander(x_cheb, increasing=True)
P_cheb, L_cheb, U_cheb,Q_cheb = fatt_lu_pivot_totale(V_cheb)

y_perm_cheb = P_cheb @ y_cheb
z_cheb = las.solve_triangular(L_cheb, y_perm_cheb, lower=True)
a_tilde_cheb = las.solve_triangular(U_cheb, z_cheb)
a_cheb = Q_cheb @ a_tilde_cheb

# ====================================================================
# Punto 5: Valutazione dei polinomi e Grafico (550 punti)
# ====================================================================

# Generiamo i 550 punti equidistanti
x_plot = np.linspace(-2, 2, 550)
y_esatto = f(x_plot)

y_plot_eq = np.polyval(a_eq[::-1], x_plot)
y_plot_cheb = np.polyval(a_cheb[::-1], x_plot)

# Uso subplots per avere un controllo migliore sulla figura
fig, ax = plt.subplots(figsize=(10, 6))

# Sposto il grafico principale un po' verso destra per fare spazio ai bottoni
plt.subplots_adjust(left=0.25)

# 1. Disegno le linee e i nodi, SALVANDOLI in delle variabili.
l0, = ax.plot(x_plot, y_esatto, 'k-', label='f(x) Esatta', linewidth=2)
l1, = ax.plot(x_plot, y_plot_eq, 'r--', label='Equispaziati')
l2, = ax.plot(x_plot, y_plot_cheb, 'b-.', label='Chebyshev')

# Salvo anche i nodi così li nascondo assieme alle curve
p1, = ax.plot(x_eq, y_eq, 'ro', markersize=6)
p2, = ax.plot(x_cheb, y_cheb, 'bs', markersize=6)

ax.set_title('Interpolazione: Equispaziati vs Chebyshev')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim([-8, 8])
ax.grid(True)

# --- 2. Selezione bottoni  ---
ax_check = plt.axes([0.02, 0.45, 0.18, 0.2]) 

# Rimuovo il riquadro nero attorno all'area dei bottoni
for spine in ax_check.spines.values():
    spine.set_visible(False)
ax_check.set_facecolor('#f4f4f4') # Sfondo grigino elegante

labels = ['f(x) Esatta', 'Equispaziati', 'Chebyshev']
visibility = [True, True, True] 
colori = ['black', 'red', 'blue']

# Moltiplico per 3 le liste a singolo elemento per evitare l'errore del cycler
check = CheckButtons(
    ax=ax_check, 
    labels=labels, 
    actives=visibility,
    # Stile del testo
    label_props={'color': colori, 'fontweight': ['bold'] * 3, 'fontsize': [10] * 3},
    # Stile dei quadratini
    frame_props={'edgecolor': colori, 'facecolor': ['white'] * 3, 'linewidth': [1.5] * 3},
    # Stile delle spunte "X"
    check_props={'color': colori, 'linewidth': [2.5] * 3}
)

# --- 3. FUNZIONE DI AGGIORNAMENTO ---
def func(label):
    if label == 'f(x) Esatta':
        l0.set_visible(not l0.get_visible())
    elif label == 'Equispaziati':
        l1.set_visible(not l1.get_visible())
        p1.set_visible(not p1.get_visible()) 
    elif label == 'Chebyshev':
        l2.set_visible(not l2.get_visible())
        p2.set_visible(not p2.get_visible()) 
    
    plt.draw()

check.on_clicked(func)



# ====================================================================
# Punto 6: Calcolo Condizionamento, Residuo ed Errore Massimo
# ====================================================================

# 1. Numero di condizionamento (cond) di Vandermonde, L e U
cond_V_eq, cond_V_cheb = np.linalg.cond(V_eq), np.linalg.cond(V_cheb)
cond_L_eq, cond_L_cheb = np.linalg.cond(L_eq), np.linalg.cond(L_cheb)
cond_U_eq, cond_U_cheb = np.linalg.cond(U_eq), np.linalg.cond(U_cheb)

# 2. Residuo del sistema lineare: norma di (V*a - y)
res_eq = np.linalg.norm(V_eq @ a_eq - y_eq)
res_cheb = np.linalg.norm(V_cheb @ a_cheb - y_cheb)

# 3. Errore massimo di interpolazione su 550 punti: max|P(x) - f(x)|
err_max_eq = np.max(np.abs(y_plot_eq - y_esatto))
err_max_cheb = np.max(np.abs(y_plot_cheb - y_esatto))



# 1. Definisco le righe di dati
righe = [
    ["Equispaziati", cond_V_eq, cond_L_eq, cond_U_eq, res_eq, err_max_eq],
    ["Chebyshev", cond_V_cheb, cond_L_cheb, cond_U_cheb, res_cheb, err_max_cheb]
]

# 2. Definisco i nomi delle colonne
intestazioni = ["Metodo", "Cond(V)", "Cond(L)", "Cond(U)", "Residuo Sistema", "Errore Max (550 pt)"]

# 3. Stampo la tabella
print("\n--- TABELLA DI CONFRONTO ---")


# floatfmt=".5e" applica automaticamente la notazione scientifica a tutti i numeri.
print(tabulate(righe, headers=intestazioni, tablefmt="fancy_grid", floatfmt=".5e"))
print("\n-- Chiudere la finestra del grafico per terminare il programma --\n")
plt.show()