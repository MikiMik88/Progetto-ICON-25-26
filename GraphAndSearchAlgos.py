import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import heapq
import time
import random
import numpy as np
import statistics

# =============================================================================
# FUNZIONI PER LA CREAZIONE E VISUALIZZAZIONE DEL GRAFO
# =============================================================================

# Creazione del grafo
def create_graph(df, is_weighted=False):
    # Creazione del grafo
    G = nx.Graph()

    # Calcolo della frequenza dei sintomi se il grafo è pesato
    if is_weighted:
        frequenza_sintomi = df.drop("prognosis", axis=1).sum(axis=0) / len(df)

    # Aggiunta di nodi suddivisi per sintomi e malattie
    for index, row in df.iterrows():
        malattia = row["prognosis"]
        sintomi_lista = row.drop("prognosis").index[row.drop("prognosis") == 1].tolist()

        G.add_node(malattia, type="disease")        # Nodo per la malattia
        for sintomo in sintomi_lista:
            G.add_node(sintomo, type="sinthome")      # Nodo per il sintomo
            if is_weighted:
                # Peso inversamente proporzionale alla frequenza
                peso = 1 / (frequenza_sintomi[sintomo] + 0.01)
                G.add_edge(sintomo, malattia, weight=peso)
            else:
                G.add_edge(sintomo, malattia)           # Collegamento sintomo-malattia
    return G

# Visualizzazione del grafo
def visualize_graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)  # Layout per la disposizione dei nodi
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", font_size=8)
    plt.title("Grafo Sintomi-Malattie")
    plt.show()

# Analisi del grafo: numero di nodi, archi, distribuzione dei gradi
def graphAnalisys(G):
    num_nodi = G.number_of_nodes()
    num_archi = G.number_of_edges()
    print(f"Il grafo ha {num_nodi} nodi e {num_archi} archi.")
    malattie = [n for n, attr in G.nodes(data=True) if attr["type"] == "disease"]
    malattia_max_sintomi = max(malattie, key=lambda m: G.degree(m))
    malattia_min_sintomi = min(malattie, key=lambda m: G.degree(m))
    print(f"La malattia con più sintomi associati è: {malattia_max_sintomi} con {G.degree(malattia_max_sintomi)} sintomi.")
    print(f"La malattia con meno sintomi associati è: {malattia_min_sintomi} con {G.degree(malattia_min_sintomi)} sintomi.")
    gradi = [G.degree(n) for n in G.nodes()]
    plt.figure(figsize=(8, 6))
    sns.histplot(gradi, bins=20, kde=True)
    plt.xlabel("Grado del nodo")
    plt.ylabel("Frequenza")
    plt.title("Distribuzione dei gradi nel grafo")
    plt.show()

# Controllo dei pesi degli archi
def check_edge_weights(G):
    pesi = [G[u][v]['weight'] for u, v in G.edges]
    print("Min peso:", min(pesi))
    print("Max peso:", max(pesi))
    print("Media pesi:", sum(pesi) / len(pesi))

# Normalizzazione dei pesi degli archi (scala 0-1)
def normalize_weights(G):
    pesi = [G[u][v]['weight'] for u, v in G.edges]
    min_p, max_p = min(pesi), max(pesi)
    for u, v in G.edges:
        G[u][v]['weight'] = (G[u][v]['weight'] - min_p) / (max_p - min_p)

# =============================================================================
# ALGORITMI DI RICERCA
# =============================================================================

# -----------------------------------------------------------------------------
# Algoritmo di ricerca BFS
# Modifica: Restituisce una lista di triple (malattia, percorso, costo)
# -----------------------------------------------------------------------------
def bfs_diagnosi(G, sintomi_inseriti):
    # Inizializzazione della coda con i sintomi inseriti; ogni elemento è (nodo, [percorso])
    queue = [(s, [s]) for s in sintomi_inseriti]
    visitati = set()
    risultati = []
    while queue:
        nodo, percorso = queue.pop(0)
        if nodo in visitati:
            continue
        visitati.add(nodo)
        # Se il nodo è una malattia, aggiungiamo la tripla (malattia, percorso, costo)
        if G.nodes[nodo]["type"] == "disease":
            risultati.append((nodo, percorso, len(percorso)))
        for vicino in G.neighbors(nodo):
            if vicino not in visitati:
                queue.append((vicino, percorso + [vicino]))
    return risultati

# -----------------------------------------------------------------------------
# Algoritmo di ricerca DFS
# Restituisce una lista di triple (malattia, percorso, costo)
# -----------------------------------------------------------------------------
def dfs_best_disease(G, sintomi):
    # Inizializzazione dello stack con i sintomi inseriti (nodo, [percorso])
    stack = [(s, [s]) for s in sintomi]
    risultati = []
    visitati = set()
    while stack:
        nodo, percorso = stack.pop()
        if nodo in visitati:
            continue
        visitati.add(nodo)
        if G.nodes[nodo]["type"] == "disease":
            risultati.append((nodo, percorso, len(percorso)))
        for vicino in G.neighbors(nodo):
            if vicino not in visitati:
                stack.append((vicino, percorso + [vicino]))
    return risultati

# -----------------------------------------------------------------------------
# Funzione euristica per A*
# Spiegazione: Questa funzione calcola il numero di sintomi “mancanti” per raggiungere la
# malattia target. In pratica, prende l'insieme dei sintomi associati alla malattia target (cioè,
# i nodi collegati alla malattia) e lo confronta con l'insieme dei sintomi collegati al nodo corrente.
# Il risultato, cioè il numero di elementi mancanti, viene usato come stima del costo residuo.
# L'euristica è ammissibile perché non sovrastima il costo reale, ma non considera i pesi degli archi;
# un possibile miglioramento sarebbe moltiplicare il numero di sintomi mancanti per il peso medio dei sintomi.
# -----------------------------------------------------------------------------
def euristica(nodo, malattia_target, G):
    if G.nodes[nodo]["type"] == "disease":
        return 0
    sintomi_malattia = set(G.neighbors(malattia_target))
    sintomi_correnti = set(G.neighbors(nodo))
    return len(sintomi_malattia - sintomi_correnti)

# -----------------------------------------------------------------------------
# Algoritmo di ricerca A*
# Restituisce una lista di triple (malattia, percorso, costo)
# -----------------------------------------------------------------------------
def a_star_best_disease(G, sintomi):
    coda = []
    # Inizializza la coda con tutti i sintomi di partenza; ogni elemento è (f(n), g(n), [percorso])
    for sintomo in sintomi:
        heapq.heappush(coda, (0, 0, [sintomo]))
    visitati = set()
    migliori_malattie = []
    min_costo = float("inf")
    while coda:
        f_n, costo, percorso = heapq.heappop(coda)
        nodo = percorso[-1]
        if nodo in visitati:
            continue
        visitati.add(nodo)
        # Se il nodo è una malattia, aggiorna i risultati se il costo è minore o uguale a min_costo
        if G.nodes[nodo]["type"] == "disease":
            if costo < min_costo:
                migliori_malattie = [(nodo, percorso, costo)]
                min_costo = costo
            elif costo == min_costo:
                migliori_malattie.append((nodo, percorso, costo))
            continue
        for vicino in G.neighbors(nodo):
            if vicino not in visitati:
                nuovo_costo = costo + G[nodo][vicino]['weight']
                # Calcola l'euristica: per ogni malattia nel grafo, prendi il costo residuo minimo
                h_n = min(euristica(vicino, m, G) for m in G.nodes if G.nodes[m]["type"] == "disease")
                f_n = nuovo_costo + h_n
                heapq.heappush(coda, (f_n, nuovo_costo, percorso + [vicino]))
    return migliori_malattie

# -----------------------------------------------------------------------------
# Algoritmo di ricerca Branch & Bound
# Restituisce una lista di triple (malattia, percorso, costo)
# -----------------------------------------------------------------------------
def branch_and_bound_best_disease(G, sintomi):
    coda = []
    # Inizializza la coda con ogni sintomo come partenza (costo 0, percorso [sintomo])
    for sintomo in sintomi:
        heapq.heappush(coda, (0, [sintomo]))
    visitati = set()
    migliori_malattie = []
    min_costo = float("inf")
    while coda:
        costo, percorso = heapq.heappop(coda)
        nodo = percorso[-1]
        if nodo in visitati:
            continue
        visitati.add(nodo)
        # Se il nodo è una malattia, aggiorna i risultati se il costo è minore o uguale a min_costo
        if G.nodes[nodo]["type"] == "disease":
            if costo < min_costo:
                migliori_malattie = [(nodo, percorso, costo)]
                min_costo = costo
            elif costo == min_costo:
                migliori_malattie.append((nodo, percorso, costo))
            continue
        for vicino in G.neighbors(nodo):
            if vicino not in visitati:
                nuovo_costo = costo + G[nodo][vicino]['weight']
                if nuovo_costo < min_costo:  # Pruning: espandi solo se il nuovo costo è inferiore al minimo attuale
                    heapq.heappush(coda, (nuovo_costo, percorso + [vicino]))
    return migliori_malattie

# =============================================================================
# FUNZIONI PER IL CALCOLO DELLE METRICHE DI CONFRONTO
# =============================================================================
# Le metriche di confronto sono le seguenti:
# - Tempo di esecuzione: il tempo impiegato dall'algoritmo per trovare la soluzione.
# - Nodi esplorati: il numero di percorsi (soluzioni) trovati.
# - Profondità media: la lunghezza media dei percorsi (numero di nodi nel percorso).
# - Costo totale: la somma dei costi dei percorsi; per BFS/DFS, costo = lunghezza del percorso;
#   per A* e Branch & Bound, il costo viene calcolato in base ai pesi.
# - Costo medio per nodo: il costo totale diviso per il numero di soluzioni trovate.
# - Efficienza di esplorazione: il rapporto tra il numero di nodi esplorati e il numero totale di nodi nel grafo.
def calcola_profondita_media(risultati):
    if not risultati:
        return 0
    lunghezze = [len(percorso) for _, percorso, _ in risultati]
    return sum(lunghezze) / len(lunghezze)

def costo_medio_per_nodo(risultati, nodi_esplorati):
    if nodi_esplorati == 0:
        return float("inf")
    costo_totale = sum(costo for _, _, costo in risultati)
    return costo_totale / nodi_esplorati

def efficienza_ricerca(nodi_esplorati, G):
    return nodi_esplorati / len(G.nodes)

# =============================================================================
# FUNZIONE PER CONFRONTARE LE PRESTAZIONI DEGLI ALGORITMI
# =============================================================================
# Le metriche comuni che confrontiamo sono:
# - Tempo di esecuzione
# - Nodi esplorati
# - Profondità media
# - Costo totale
# - Costo medio per nodo
# - Efficienza di esplorazione
# - Malattia trovata: la malattia con il percorso a costo minimo
# Spiegazione delle metriche:
# • Tempo: indica quanto velocemente l'algoritmo trova la soluzione.
# • Nodi esplorati: numero di percorsi trovati; un numero minore indica una ricerca più mirata.
# • Profondità media: la lunghezza media dei percorsi; soluzioni più brevi sono preferibili.
# • Costo totale: la somma dei costi dei percorsi; un costo inferiore indica soluzioni migliori.
# • Costo medio per nodo: il costo medio per ogni soluzione trovata.
# • Efficienza: il rapporto tra i nodi esplorati e i nodi totali del grafo.
# • Malattia trovata: il valore (nome) della malattia corrispondente al percorso a costo minimo.
def confronta_algoritmi_estesi(G, num_test=5, min_sintomi=1, max_sintomi=3):
    # Inizializza un dizionario per accumulare le metriche per ogni algoritmo
    metriche_globali = {
        algo: {metrica: [] for metrica in ["Tempo", "Nodi esplorati", "Profondità media", "Costo totale", "Costo medio per nodo", "Efficienza", "Malattia trovata"]}
        for algo in ["BFS", "DFS", "A*", "B&B"]
    }

    sintomi_disponibili = [n for n, attr in G.nodes(data=True) if attr["type"] == "sinthome"]

    # Esegui i test su insiemi di sintomi casuali
    for i in range(num_test):
        sintomi_test = random.sample(sintomi_disponibili, random.randint(min_sintomi, max_sintomi))
        print(f"\nTest {i+1} - Sintomi: {sintomi_test}")

        risultati = {}

        # BFS
        start = time.time()
        bfs_risultati = bfs_diagnosi(G, sintomi_test)
        tempo_bfs = time.time() - start
        nodi_bfs = len(bfs_risultati)
        costo_tot_bfs = sum(c for _, _, c in bfs_risultati) if bfs_risultati else 0
        malattia_bfs = min(bfs_risultati, key=lambda x: x[2])[0] if bfs_risultati else None
        risultati["BFS"] = {
            "Tempo": tempo_bfs,
            "Nodi esplorati": nodi_bfs,
            "Profondità media": calcola_profondita_media(bfs_risultati),
            "Costo totale": costo_tot_bfs,
            "Costo medio per nodo": costo_tot_bfs / nodi_bfs if nodi_bfs > 0 else float("inf"),
            "Efficienza": efficienza_ricerca(nodi_bfs, G),
            "Malattia trovata": malattia_bfs
        }

        # DFS
        start = time.time()
        dfs_risultati = dfs_best_disease(G, sintomi_test)
        tempo_dfs = time.time() - start
        nodi_dfs = len(dfs_risultati)
        costo_tot_dfs = sum(c for _, _, c in dfs_risultati) if dfs_risultati else 0
        malattia_dfs = min(dfs_risultati, key=lambda x: x[2])[0] if dfs_risultati else None
        risultati["DFS"] = {
            "Tempo": tempo_dfs,
            "Nodi esplorati": nodi_dfs,
            "Profondità media": calcola_profondita_media(dfs_risultati),
            "Costo totale": costo_tot_dfs,
            "Costo medio per nodo": costo_tot_dfs / nodi_dfs if nodi_dfs > 0 else float("inf"),
            "Efficienza": efficienza_ricerca(nodi_dfs, G),
            "Malattia trovata": malattia_dfs
        }

        # A*
        start = time.time()
        a_star_risultati = a_star_best_disease(G, sintomi_test)
        tempo_a_star = time.time() - start
        nodi_astar = len(a_star_risultati)
        costo_tot_astar = sum(c for _, _, c in a_star_risultati) if a_star_risultati else 0
        malattia_astar = min(a_star_risultati, key=lambda x: x[2])[0] if a_star_risultati else None
        risultati["A*"] = {
            "Tempo": tempo_a_star,
            "Nodi esplorati": nodi_astar,
            "Profondità media": calcola_profondita_media(a_star_risultati),
            "Costo totale": costo_tot_astar,
            "Costo medio per nodo": costo_tot_astar / nodi_astar if nodi_astar > 0 else float("inf"),
            "Efficienza": efficienza_ricerca(nodi_astar, G),
            "Malattia trovata": malattia_astar
        }

        # Branch & Bound
        start = time.time()
        bb_risultati = branch_and_bound_best_disease(G, sintomi_test)
        tempo_bb = time.time() - start
        nodi_bb = len(bb_risultati)
        costo_tot_bb = sum(c for _, _, c in bb_risultati) if bb_risultati else 0
        malattia_bb = min(bb_risultati, key=lambda x: x[2])[0] if bb_risultati else None
        risultati["B&B"] = {
            "Tempo": tempo_bb,
            "Nodi esplorati": nodi_bb,
            "Profondità media": calcola_profondita_media(bb_risultati),
            "Costo totale": costo_tot_bb,
            "Costo medio per nodo": costo_tot_bb / nodi_bb if nodi_bb > 0 else float("inf"),
            "Efficienza": efficienza_ricerca(nodi_bb, G),
            "Malattia trovata": malattia_bb
        }

        # Stampa dei risultati per il test corrente
        print(f"\nRisultati per il test {i+1}:")
        for algo, dati in risultati.items():
            print(f"\n🔹 {algo}")
            for metrica, valore in dati.items():
                if metrica == "Malattia trovata":
                    print(f"{metrica}: {valore}")
                else:
                    print(f"{metrica}: {valore:.4f}")

        # Accumulo dei valori per il calcolo della media globale
        for algo in risultati:
            for metrica, valore in risultati[algo].items():
                metriche_globali[algo][metrica].append(valore)

    # Calcolo delle medie delle metriche per ogni algoritmo
    metriche_medie = {}
    for algo, dati in metriche_globali.items():
        metriche_medie[algo] = {}
        for metrica, valori in dati.items():
            if metrica == "Malattia trovata":
                # Calcola la moda se possibile, altrimenti prende il primo valore
                try:
                    metriche_medie[algo][metrica] = statistics.mode(valori)
                except statistics.StatisticsError:
                    metriche_medie[algo][metrica] = valori[0] if valori else None
            else:
                metriche_medie[algo][metrica] = np.mean(valori)

    print("\n📊 Metriche medie complessive:")
    for algo, dati in metriche_medie.items():
        print(f"\n🔹 {algo}")
        for metrica, valore in dati.items():
            if metrica == "Malattia trovata":
                continue
            else:
                print(f"{metrica}: {valore:.4f}")

    # Generazione di un grafico di confronto per le metriche comuni (eccetto la malattia trovata)
    plt.figure(figsize=(10,6))
    metriche_comuni = ["Tempo", "Nodi esplorati", "Profondità media", "Costo totale", "Costo medio per nodo", "Efficienza"]
    for metrica in metriche_comuni:
        valori = [metriche_medie[algo][metrica] for algo in metriche_medie]
        plt.plot(list(metriche_medie.keys()), valori, marker='o', label=metrica)
    plt.xlabel("Algoritmi")
    plt.ylabel("Valore medio della metrica")
    plt.title("Confronto globale delle metriche medie tra algoritmi di ricerca")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Confronto finale delle malattie trovate
    malattie_finali = {algo: metriche_medie[algo]["Malattia trovata"] for algo in metriche_medie}
    if len(set(malattie_finali.values())) == 1:
        print("Tutti gli algoritmi hanno trovato la stessa malattia.")
    else:
        print("Gli algoritmi hanno trovato malattie differenti.")

# =============================================================================
# CORPO PRINCIPALE: FLUSSO DI ESECUZIONE
# =============================================================================

# Caricamento del dataset
df = pd.read_csv("symbipredict_2022.csv")

# Creazione del grafo (con pesi)
G = create_graph(df, is_weighted=True)

# Normalizzazione dei pesi degli archi
normalize_weights(G)

# Visualizzazione del grafo
visualize_graph(G)

# Analisi del grafo: numero di nodi, archi, distribuzione dei gradi
graphAnalisys(G)

# Confronto degli algoritmi utilizzando il vettore di sintomi generato e test multipli
confronta_algoritmi_estesi(G, num_test=10, min_sintomi=1, max_sintomi=3)
