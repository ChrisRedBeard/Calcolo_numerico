
import numpy as np                    # array e funzioni matematiche vettoriali
import matplotlib.pylab as plt        # grafici (stessa convenzione delle lezioni)
from scipy.optimize import newton as scipy_newton   # routine di riferimento (punto 6)
from tabulate import tabulate         # stampa ordinata delle tabelle


# intestazione comune alle tabelle di iterazione dei due metodi
_HEADER_IT = ['num_iterazioni', 'x_n', 'f(x_n)', '|x_n - x_(n-1)|']
# formato per colonna: nit intero, x_n con 12 decimali, residuo ed errore in
# notazione scientifica
_FMT_IT = ('d', '.12f', '.6e', '.6e')


# -------------------------------------------------------------------------
#  Funzione di cui cercare lo zero e sua derivata prima
# -------------------------------------------------------------------------
def f(x):
#  funzione: f(x) = cos(x) - x
    return np.cos(x) - x


def df(x):
#   Derivata prima:  f'(x) = -sin(x) - 1
    return -np.sin(x) - 1.0


# -------------------------------------------------------------------------
#  Metodo di Newton (o delle tangenti)
# -------------------------------------------------------------------------
def newton(f, df, x0, atol=1e-8, rtol=1e-8, nmax=100, displayit=1):
    # --- inizializzazione ---
    x  = x0                # iterata corrente: si parte da x0
    fx = f(x)              # f(x0): la si calcola una volta sola
    nfun = 1               # contatore valutazioni di funzione (f(x0) -> 1)
    nit  = 0               # contatore iterazioni
    exitflag = 0           # 0 finche' non si raggiunge la convergenza
    stima_err = np.inf     # stima errore iniziale "grande" per entrare nel ciclo
    storia = []            # raccoglie le righe della tabella di iterazione

    # --- ciclo iterativo ---
    while nit < nmax:                    # salvaguardia sul numero massimo di passi
        dfx = df(x)                      # valuto la derivata nel punto corrente
        nfun += 1                        #   -> una valutazione di funzione in piu'

        if dfx == 0:                     # derivata nulla: la tangente e' orizzontale,
            print('Derivata nulla: impossibile proseguire.')
            break                        #   il passo non e' definito -> esco

        dx = fx / dfx                    # passo di Newton:  f(x_n) / f'(x_n)
        x  = x - dx                      # nuova iterata: x_(n+1) = x_n - f/f'
        fx = f(x)                        # f nella nuova iterata (serve al passo
        nfun += 1                        #   successivo e per riportare f(x_n) finale)
        nit += 1                         # iterazione completata
        stima_err = abs(dx)              # stima errore = |x_(n+1) - x_n| = |dx|

        storia.append([nit, x, fx, stima_err])   # memorizzo la riga per la tabella

        # criterio d'arresto sull'errore misto:
        #     |x_(n+1) - x_n| < atol + rtol * |x_(n+1)|
        if stima_err < atol + rtol * abs(x):
            exitflag = 1                 # convergenza raggiunta -> esco dal ciclo
            break

    if displayit:                        # stampo la tabella delle iterazioni
        print(tabulate(storia, headers=_HEADER_IT,
                       floatfmt=_FMT_IT, tablefmt='fancy_grid'))

    return (x, fx, exitflag, nfun, nit, stima_err)


# -------------------------------------------------------------------------
#  Metodo delle secanti
# -------------------------------------------------------------------------
def secanti(f, x0, x1, atol=1e-8, rtol=1e-8, nmax=100, displayit=1):
    # --- inizializzazione: servono due punti e i rispettivi valori di f ---
    xprec = x0             # x_(n-1): penultima iterata
    fprec = f(xprec)       # f(x_(n-1))
    x     = x1             # x_n:     ultima iterata
    fx    = f(x)           # f(x_n)
    nfun  = 2              # gia' eseguite 2 valutazioni: f(x0) e f(x1)
    nit   = 0
    exitflag = 0
    stima_err = np.inf
    storia = []            # raccoglie le righe della tabella di iterazione

    # --- ciclo iterativo ---
    while nit < nmax:
        den = fx - fprec               # denominatore del rapporto incrementale
        if den == 0:                   # i due valori di f coincidono: secante
            print('Denominatore nullo: impossibile proseguire.')  # orizzontale
            break

        dx = fx * (x - xprec) / den    # passo delle secanti

        # aggiorno la "memoria" PRIMA di sovrascrivere x:
        xprec = x                      # la x corrente diventa la penultima
        fprec = fx                     #   e con essa il suo valore di f
        x  = x - dx                    # nuova iterata x_(n+1)
        fx = f(x)                      # f nella nuova iterata: UNICA valutazione
        nfun += 1                      #   del passo -> una sola valutazione di f
        nit += 1
        stima_err = abs(dx)            # stima errore = |x_(n+1) - x_n|

        storia.append([nit, x, fx, stima_err])   # memorizzo la riga per la tabella

        # stesso criterio d'arresto sull'errore misto usato per Newton
        if stima_err < atol + rtol * abs(x):
            exitflag = 1
            break

    if displayit:                      # stampo la tabella delle iterazioni
        print(tabulate(storia, headers=_HEADER_IT,
                       floatfmt=_FMT_IT, tablefmt='fancy_grid'))

    return (x, fx, exitflag, nfun, nit, stima_err)


# -------------------------------------------------------------------------
#  Punto 1 - localizzazione grafica della radice positiva
# -------------------------------------------------------------------------
def localizza(a=0.0, b=1.5, x0=0.5):
    """
    Disegna f(x)=cos(x)-x sull'intervallo [a,b] per localizzare la radice
    positiva, evidenzia il punto iniziale x0 e SEGNA lo zero della funzione
    (individuato dal cambio di segno tra i punti campionati). Stampa inoltre
    il controllo del cambio di segno agli estremi di [0,1].
    """
    xp = np.linspace(a, b, 200)        # griglia di punti per il grafico
    yp = f(xp)                         # valori di f sulla griglia (servono per lo zero)

    plt.figure(figsize=(7, 4))
    plt.plot(xp, yp, label=r'$f(x)=\cos(x)-x$')      # grafico di f
    plt.plot(xp, 0 * xp, 'k--', linewidth=0.8)       # asse x (retta y = 0)
    plt.axvline(x0, color='red', linestyle=':',      # punto iniziale
                label=r'$x_0=%.2f$' % x0)

    # --- individuo e segno lo zero della funzione ---
    # np.sign(yp) da' il segno (+1/-1) in ogni punto; np.diff e' diverso da zero
    # solo dove il segno cambia: tra quei due punti la funzione attraversa lo zero.
    idx = np.where(np.diff(np.sign(yp)) != 0)[0]
    for i in idx:
        # stima dell'ascissa dello zero per interpolazione lineare fra i due punti
        xz = xp[i] - yp[i] * (xp[i + 1] - xp[i]) / (yp[i + 1] - yp[i])
        plt.plot(xz, 0, 'go', markersize=10, label=r'radice $\approx %.4f$' % xz)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Localizzazione grafica della radice positiva')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Controllo numerico del cambio di segno agli estremi di [0, 1]:
    # cos(x) in [-1,1] interseca y=x solo per x in [0,1]; f(0)=1>0, f(1)<0,
    # quindi per il teorema degli zeri la radice cade in (0,1).
    print('f(0)      = %.4f' % f(0.0))
    print('f(1)      = %.4f' % f(1.0))
    print('f(0)*f(1) = %.4f  -> segni opposti: esiste una radice in (0,1)'
          % (f(0.0) * f(1.0)))


# -------------------------------------------------------------------------
#  Visualizzazione grafica dei passi dei due metodi
# -------------------------------------------------------------------------
# colori distinti per i passi successivi (ben staccati dal nero della curva)
_COLORI_PASSI = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#1f77b4']


def _griglia_plot(xs, extra=0.06):
    """Intervallo dell'asse x su cui disegnare f, ricavato dalle iterate xs."""
    lo, hi = min(xs), max(xs)
    m = 0.18 * (hi - lo) + extra        # un po' di margine ai lati
    return np.linspace(lo - m, hi + m, 400)

def _etichetta_iterate(xs):
    """Scrive x_0, x_1, ... sotto l'asse x, saltando le iterate troppo vicine
    fra loro (che, per la rapida convergenza, si sovrapporrebbero)."""
    soglia = 0.03 * (max(xs) - min(xs))
    fatte = []
    for k, xk in enumerate(xs):
        if all(abs(xk - lx) > soglia for lx in fatte):
            plt.annotate(r'$x_{%d}$' % k, (xk, 0), textcoords='offset points',
                         xytext=(0, -17), ha='center', fontsize=12)
            fatte.append(xk)

def plot_newton(f, df, x0, atol=1e-8, rtol=1e-8, nmax=100, n_passi=None):
    """
    Visualizza i passi del metodo di Newton. Per ogni iterazione disegna la
    retta TANGENTE nel punto corrente (x_n, f(x_n)) e la sua intersezione con
    l'asse x, che fornisce l'iterata successiva x_(n+1). Con n_passi si puo'
    limitare il numero di passi disegnati.
    """
    # rieseguo l'iterazione raccogliendo tutte le iterate x0, x1, x2, ...
    xs, x = [x0], x0
    for _ in range(nmax):
        d = df(x)
        if d == 0:
            break
        xnew = x - f(x) / d
        xs.append(xnew)
        if abs(xnew - x) < atol + rtol * abs(xnew):
            break
        x = xnew
    if n_passi is not None:
        xs = xs[:n_passi + 1]

    xp = _griglia_plot(xs)
    plt.figure(figsize=(8, 5))
    plt.plot(xp, f(xp), 'k-', linewidth=2, label=r'$f(x)=\cos(x)-x$')  # curva (nera)
    plt.axhline(0, color='gray', linewidth=0.8)                        # asse x

    for k in range(len(xs) - 1):
        c = _COLORI_PASSI[k % len(_COLORI_PASSI)]
        xn, xn1 = xs[k], xs[k + 1]
        plt.plot([xn, xn1], [f(xn), 0], '-', color=c, lw=1.8)    # tangente -> asse x
        plt.plot([xn1, xn1], [0, f(xn1)], ':', color=c, lw=1.2)  # verticale al nuovo punto
        plt.plot(xn, f(xn), 'o', color=c, markersize=8)          # punto sulla curva

    _etichetta_iterate(xs)
    plt.plot(xs[-1], 0, '*', color='limegreen', markersize=18,    # radice
             markeredgecolor='k', markeredgewidth=0.5, label='radice')
    plt.xlabel('x'); plt.ylabel('f(x)')
    plt.title('Passi del metodo di Newton (rette tangenti)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.show()

def plot_secanti(f, x0, x1, atol=1e-8, rtol=1e-8, nmax=100, n_passi=None):
    """
    Visualizza i passi del metodo delle secanti. Per ogni iterazione disegna la
    retta SECANTE passante per gli ultimi due punti e la sua intersezione con
    l'asse x, che fornisce l'iterata successiva.
    """
    # rieseguo l'iterazione raccogliendo le iterate x0, x1, x2, ...
    xs = [x0, x1]
    xprec, fprec, x, fx = x0, f(x0), x1, f(x1)
    for _ in range(nmax):
        den = fx - fprec
        if den == 0:
            break
        xnew = x - fx * (x - xprec) / den
        xs.append(xnew)
        if abs(xnew - x) < atol + rtol * abs(xnew):
            break
        xprec, fprec, x, fx = x, fx, xnew, f(xnew)
    if n_passi is not None:
        xs = xs[:n_passi + 2]

    xp = _griglia_plot(xs)
    plt.figure(figsize=(8, 5))
    plt.plot(xp, f(xp), 'k-', linewidth=2, label=r'$f(x)=\cos(x)-x$')  # curva (nera)
    plt.axhline(0, color='gray', linewidth=0.8)                        # asse x

    for k in range(1, len(xs) - 1):
        c = _COLORI_PASSI[(k - 1) % len(_COLORI_PASSI)]
        xa, xb, xc = xs[k - 1], xs[k], xs[k + 1]       # due punti + intercetta
        xx = np.array([min(xa, xb, xc), max(xa, xb, xc)])
        m = (f(xb) - f(xa)) / (xb - xa)                # pendenza della secante
        plt.plot(xx, f(xa) + m * (xx - xa), '-', color=c, lw=1.8)   # secante
        plt.plot([xc, xc], [0, f(xc)], ':', color=c, lw=1.2)        # verticale
        plt.plot([xa, xb], [f(xa), f(xb)], 'o', color=c, markersize=8)  # i due punti

    _etichetta_iterate(xs)
    plt.plot(xs[-1], 0, '*', color='limegreen', markersize=18,    # radice
             markeredgecolor='k', markeredgewidth=0.5, label='radice')
    plt.xlabel('x'); plt.ylabel('f(x)')
    plt.title('Passi del metodo delle secanti')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.show()

# -------------------------------------------------------------------------
#  Esecuzione: punti 3-4-5-6
# -------------------------------------------------------------------------
if __name__ == '__main__':

    # ---- Punti 3 e 4: dati del problema ----
    x0   = 0.5      # punto iniziale per Newton e PRIMO punto per le secanti
    x1   = 1.0      # secondo punto iniziale, richiesto dal metodo delle secanti
    atol = 1e-8     # tolleranza assoluta
    rtol = 1e-8     # tolleranza relativa

    # ---- Punto 1: localizzazione grafica ----
    print('=== PUNTO 1: localizzazione grafica ===')
    localizza(0.0, 1.5, x0)
    print()

    # ---- Esecuzione del metodo di Newton ----
    print('=== METODO DI NEWTON ===')
    xN, fxN, flagN, nfunN, nitN, errN = newton(f, df, x0, atol, rtol, displayit=1)
    print('Radice approssimata : %.15f   (convergenza flag = %d)' % (xN, flagN))
    print()

    # ---- Esecuzione del metodo delle secanti ----
    print('=== METODO DELLE SECANTI ===')
    xS, fxS, flagS, nfunS, nitS, errS = secanti(f, x0, x1, atol, rtol, displayit=1)
    print('Radice approssimata : %.15f   (convergenza flag = %d)' % (xS, flagS))
    print()

    # ---- Visualizzazione grafica dei passi dei due metodi ----
    print('=== VISUALIZZAZIONE DEI PASSI (chiudere ogni finestra per proseguire) ===')
    plot_newton(f, df, x0, atol, rtol)
    plot_secanti(f, x0, x1, atol, rtol)
    print()

    # ---- Punto 5: confronto fra i due metodi ----
    # Ogni cella viene pre-formattata come stringa, cosi' ogni indicatore ha il
    # suo formato (interi, notazione scientifica, decimali). disable_numparse
    # impedisce a tabulate di reinterpretare e riformattare queste stringhe.
    print('=== PUNTO 5: confronto Newton / Secanti ===')
    tab5 = [
        ['numero di iterazioni',           '%d' % nitN,    '%d' % nitS],
        ['valutazioni di funzione (nfun)', '%d' % nfunN,   '%d' % nfunS],
        ['f(x_n) ultima iterata',          '%.3e' % fxN,   '%.3e' % fxS],
        ['errore stimato finale',          '%.3e' % errN,  '%.3e' % errS],
        ['radice approssimata',            '%.12f' % xN,   '%.12f' % xS],
    ]
    print(tabulate(tab5, headers=['', 'Newton', 'Secanti'], tablefmt='fancy_grid',
                   colalign=('left', 'right', 'right'), disable_numparse=True))
    print()

    # ---- Punto 6: confronto con scipy.optimize.newton ----
    # Variante "tangenti": con fprime=df scipy esegue il metodo di Newton.
    radN, infoN = scipy_newton(f, x0, fprime=df,
                               tol=atol, rtol=rtol, full_output=True)
    # Variante "secanti": senza fprime ma con x1 scipy esegue il metodo delle secanti.
    radS, infoS = scipy_newton(f, x0, x1=x1,
                               tol=atol, rtol=rtol, full_output=True)

    print('=== PUNTO 6: confronto con scipy.optimize.newton ===')
    tab6 = [
        ['iterazioni',
         '%d' % nitN,  '%d' % infoN.iterations,
         '%d' % nitS,  '%d' % infoS.iterations],
        ['valutazioni di funzione',
         '%d' % nfunN, '%d' % infoN.function_calls,
         '%d' % nfunS, '%d' % infoS.function_calls],
        ['radice',
         '%.10f' % xN, '%.10f' % radN,
         '%.10f' % xS, '%.10f' % radS],
    ]
    print(tabulate(tab6,
                   headers=['', 'Newton', 'sp.Newton', 'Secanti', 'sp.Secanti'],
                   tablefmt='fancy_grid', disable_numparse=True,
                   colalign=('left', 'right', 'right', 'right', 'right')))