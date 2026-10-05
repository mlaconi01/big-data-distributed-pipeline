"""
PROGETTO FINALE DI LABORATORIO DI BIG DATA
Autore: Mirco Laconi
Professore: Alessandro Giuliani

Descrizione:
Script per l'elaborazione distribuita di log di navigazione, validazione empirica 
del costo delle operazioni Shuffle-Heavy in Dask e addestramento 
parallelo di modelli di classificazione (Random Forest vs AdaBoost).
"""

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
import time
import joblib

# Importazioni per il Machine Learning (Scikit-Learn)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

def main():
    # ==========================================
    # FASE 1: SETUP E CARICAMENTO DATI (TAGLIA L)
    # ==========================================
    print("\n[FASE 1] Inizializzazione ambiente per Dataset L...")
    
    # Configurazione del Cluster Locale. 
    # Scelta ingegneristica: limitiamo la RAM a 1GB per singolo worker.
    # Questo serve a simulare i vincoli di un vero ambiente distribuito,
    # innescando scenari di "spilling" su disco e misurando il traffico di rete reale.
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='1GB')
    client = Client(cluster)
    print(f"---> Dashboard Dask: {client.dashboard_link} <---")

    # OVER-PARTITIONING ADATTIVO
    # Il file L pesa circa 178.7 MB. Avendo 8 thread logici a disposizione (4 worker * 2),
    # impostiamo il blocksize a 22.33MB per ottenere esattamente 8 partizioni bilanciate.
    # Questo ottimizza il parallelismo e previene lo Straggler Effect.
    MIO_BLOCKSIZE = "22.33MB" 
    
    df_events = dd.read_csv('events_L.csv', dtype=str, blocksize=MIO_BLOCKSIZE)
    print(f"Dataset Events L caricato. Partizioni: {df_events.npartitions}")
    
    # ==========================================
    # FASE 2: DATA CLEANING (Esecuzione Lazy)
    # ==========================================
    print("\n[FASE 2] Pulizia dati in corso...")
    df_cleaned = df_events.copy()
    
    # CAST TIPOLOGICO: Forziamo la conversione a float. 
    # Utilizziamo map_partitions per parallelizzare la funzione pd.to_numeric.
    # errors='coerce' cattura le stringhe "sporche" e le trasforma in NaN senza bloccare il job.
    df_cleaned['dwell_time_sec'] = df_cleaned['dwell_time_sec'].map_partitions(pd.to_numeric, errors='coerce')
    df_cleaned['scroll_depth'] = df_cleaned['scroll_depth'].map_partitions(pd.to_numeric, errors='coerce')
    
    # IMPUTAZIONE MISSING VALUES (Quantitativi)
    # Un tempo o uno scroll mancanti sono semanticamente associabili a un'interazione nulla (0.0).
    df_cleaned['dwell_time_sec'] = df_cleaned['dwell_time_sec'].fillna(0.0)
    df_cleaned['scroll_depth'] = df_cleaned['scroll_depth'].fillna(0.0)

    # IMPUTAZIONE MISSING VALUES (Categorici)
    # Per non perdere righe essenziali ai fini del conteggio, categorizziamo i NaN come "unknown".
    for col in ['device_type', 'referrer', 'content_category']:
        df_cleaned[col] = df_cleaned[col].str.lower().fillna('unknown')

    # ==========================================
    # FASE 3: FEATURE ENGINEERING
    # ==========================================
    print("\n[FASE 3] Esecuzione Benchmark su scala L...")
    
    # Regole logiche per il passaggio da "Granularità ad evento" a "Granularità a sessione"
    agg_rules = {
        'event_id': 'count',       # Volume di interazione
        'dwell_time_sec': 'sum',   # Coinvolgimento temporale cumulato
        'scroll_depth': 'mean',    # Profondità media di esplorazione
        'device_type': 'first'     # Dispositivo di accesso iniziale
    }

    # --- PIPELINE A: OTTIMIZZATA (Tree Reduction) ---
    print("-> Pipeline A (Locale/Tree Reduction)...")
    t0_a = time.time()
    
    # Il groupby base applica una riduzione ad albero: le aggregazioni vengono calcolate
    # localmente in ogni singolo worker e solo i risultati parziali viaggiano sulla rete.
    session_A = df_cleaned.groupby('session_id').agg(agg_rules)
    
    # Calcolo della Variabile Target: se l'utente ha visitato > 1 comune, cross_municipality = 1
    mun_nunique_A = df_cleaned.groupby('session_id')['municipality'].nunique()
    session_A['cross_municipality'] = (mun_nunique_A > 1).astype(int)
    
    res_A = session_A.compute() # Trigger fisico dell'esecuzione lazy
    time_A = time.time() - t0_a

    # --- PIPELINE B: SHUFFLE-HEAVY ---
    print("-> Pipeline B (Heavy Shuffle/set_index)...")
    print("   [!] Attenzione: su dataset L questa operazione potrebbe richiedere molta RAM e tempo...")
    t0_b = time.time()
    
    # Il set_index riallinea l'intero dataset. Costringe i worker a smontare e scambiare
    # i blocchi di RAM attraverso la rete per riposizionare le righe in base al session_id.
    # Operazione estremamente costosa (Shuffle).
    df_shuffled = df_cleaned.set_index('session_id')
    session_B = df_shuffled.groupby('session_id').agg(agg_rules)
    
    mun_nunique_B = df_shuffled.groupby('session_id')['municipality'].nunique()
    session_B['cross_municipality'] = (mun_nunique_B > 1).astype(int)
    
    res_B = session_B.compute()
    time_B = time.time() - t0_b

    # ==========================================
    # ANALISI PRESTAZIONI PIPELINE E INTEGRITA'
    # ==========================================
    print("\n" + "="*50)
    print(f"BENCHMARK DATASET L (PIPELINE):")
    print(f"Tempo Pipeline A: {time_A:.2f}s")
    print(f"Tempo Pipeline B: {time_B:.2f}s")
    print(f"Fattore di rallentamento: {time_B/time_A:.2f}x")
    print("="*50)

    print("\n--- DIAGNOSTICA INTEGRITA' DATI ---")
    col_num = ['event_id', 'dwell_time_sec', 'scroll_depth', 'cross_municipality']
    df_a = res_A[col_num].sort_index().fillna(0)
    df_b = res_B[col_num].sort_index().fillna(0)

    # Validazione tramite tolleranza d'errore (e non controllo binario puro, perchè mi ha causato 
    # errori in precenza).
    # Lo shuffle modifica l'ordine dei record processati dalle CPU. Questo innesca
    # il "Floating Point Non-Determinism": le somme di frazioni possono presentare 
    # scostamenti infinitesimali a causa degli arrotondamenti hardware (standard IEEE 754).
    for col in col_num:
        max_diff = (df_a[col] - df_b[col]).abs().max()
        # Valutiamo il test come superato se l'errore è trascurabile (< 0.001)
        esito = "OK" if max_diff < 0.001 else "FALLITO"
        print(f"Colonna '{col}': Errore Max = {max_diff} [{esito}]")
        
    # ==========================================
    # FASE 4: MACHINE LEARNING DISTRIBUITO
    # ==========================================
    print("\n[FASE 4] Training Modelli su Dataset L...")
    
    df_ml = res_A.dropna()
    
    # Prevenzione Data Leakage: Escludiamo i codici anagrafici dei comuni dalla matrice X, se no il modello bara 
    X = df_ml.drop(columns=['cross_municipality'])
    y = df_ml['cross_municipality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline di trasformazione per le feature
    # - Standardizzazione per limitare le differenze di scala
    # - OneHotEncoder per evitare gerarchie matematiche implicite nelle categorie
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['event_id', 'dwell_time_sec', 'scroll_depth']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['device_type'])
    ])

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Iniezione del task nel cluster distribuito tramite joblib backend (avrei potuto farlo 
    # in un ambiente non distribuito ma non è il focus dell'esame, ho cercato di simularlo in questo modo)
    with joblib.parallel_backend('dask'):
        print("-> Training Random Forest (max_depth=5)...")
        t0_rf = time.time()
        # limitare la max_depth a 5 è cruciale per prevenire esplosioni della RAM (Out-Of-Memory)
        # e per arginare fenomeni di Overfitting su dati squilibrati.
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train_p, y_train)
        time_rf = time.time() - t0_rf
        
        print("-> Training AdaBoost...")
        t0_ada = time.time()
        ada = AdaBoostClassifier(n_estimators=100, random_state=42).fit(X_train_p, y_train)
        time_ada = time.time() - t0_ada

    print("\n=== TEMPI DI ESECUZIONE ML ===")
    print(f"Random Forest : {time_rf:.2f} s") # Embarrassingly parallel (Bagging)
    print(f"AdaBoost      : {time_ada:.2f} s") # Soffre di colli di bottiglia architetturali, dunque dovrebbe metterci di più (Boosting sequenziale)
    
    print("\n=== REPORT FINALE L ===")
    print("RANDOM FOREST:\n", classification_report(y_test, rf.predict(X_test_p)))
    print("ADABOOST:\n", classification_report(y_test, ada.predict(X_test_p)))

    client.close()
    cluster.close()

if __name__ == '__main__':
    main()
