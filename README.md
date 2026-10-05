# Pipeline distribuita in Dask e classificazione delle sessioni web

Progetto finale del corso di **Laboratorio di Big Data** – Laurea Magistrale in Data Science, Business Analytics e Innovazione, Università di Cagliari (A.A. 2025/2026).

**Strumenti:** Python · Dask (cluster distribuito) · scikit-learn · pandas · joblib

## Obiettivo

Partendo da un registro grezzo di eventi di navigazione (circa 180 MB, taglia "L"), costruire una tabella a livello di sessione pronta per il machine learning. Il modello deve prevedere se una sessione coinvolge **un solo comune o più comuni** (`cross_municipality`).

Il focus tecnico del progetto era misurare il **costo delle operazioni con shuffle** in un ambiente distribuito.

## Cosa fa la pipeline

1. **Cluster vincolato**: 4 worker × 2 thread con 1 GB di RAM ciascuno, per simulare i limiti di un cluster reale.
2. **Partizionamento calibrato**: blocchi da circa 22 MB, così da ottenere 8 partizioni bilanciate, una per thread.
3. **Pulizia dei dati**: conversione dei tipi con `map_partitions`, valori numerici mancanti posti a 0 e categorie mancanti codificate come `unknown`.
4. **Feature engineering**: aggregazione da evento a sessione (numero di eventi, tempo totale, scroll medio, dispositivo).
5. **Benchmark**: aggregazione locale con tree reduction contro `set_index` con shuffle completo.
6. **Machine learning distribuito**: Random Forest e AdaBoost addestrati con il backend Dask di joblib, senza data leakage (i codici dei comuni sono esclusi dalle feature).

## Risultati principali

| Misura | Risultato |
|---|---|
| Aggregazione locale (Pipeline A) | 3,7 s |
| Aggregazione con shuffle (Pipeline B) | 16,0 s, cioè **4,3 volte più lenta** |
| Differenza massima tra le due pipeline | 7,3 × 10⁻¹² (solo arrotondamenti in virgola mobile) |
| Accuracy (entrambi i modelli) | **96%** su 172.007 sessioni di test |
| Recall sulla classe minoritaria | 98% Random Forest, 100% AdaBoost |
| Tempo di training | Random Forest 5,2 s, AdaBoost 15,6 s |

La Random Forest è la scelta migliore in un contesto distribuito: a parità di qualità, i suoi alberi si addestrano in parallelo, mentre il boosting è sequenziale.

![Task Stream della dashboard Dask](img/dask_task_stream.png)

![Classification report dei modelli](img/risultati_modelli.png)

## File

- `pipeline_dask.py`: script completo della pipeline
- `report_laboratorio_big_data.pdf`: relazione con metodologia, analisi e limiti
- `requirements.txt`: dipendenze

Il dataset è stato fornito dal docente per il corso e non è incluso nel repository.

## Come eseguirlo

```bash
pip install -r requirements.txt
# inserire events_L.csv nella stessa cartella dello script
python pipeline_dask.py
```

## Autore

Mirco Laconi · [LinkedIn](https://www.linkedin.com/in/mirco-laconi-b046ba267/)
