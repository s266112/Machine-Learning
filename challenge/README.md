# 💶 Sfida 1: Banknote Authentication

Analisi del dataset *banknote authentication* per il corso di Introduzione al Machine Learning. L'obiettivo è distinguere le banconote autentiche da quelle contraffatte usando caratteristiche estratte dalle immagini con la trasformata wavelet.

## 📊 Dataset

Il dataset contiene 1372 banconote, ognuna descritta da 4 feature continue e da una classe (0 o 1):

| Feature | Descrizione |
|---|---|
| varianza | varianza dell'immagine trasformata con wavelet |
| asimmetria | asimmetria (*skewness*) dell'immagine trasformata |
| curtosi | curtosi dell'immagine trasformata |
| entropia | entropia dell'immagine |

🔗 Fonte: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

## 🗂️ Struttura del notebook

**🧹 1. Pre-trattamento dei dati**
- caricamento, controllo dei valori mancanti e rimozione dei duplicati;
- analisi delle scale delle feature e standardizzazione;
- verifica dell'ordinamento dei dati e rimescolamento.

**🔍 2. Apprendimento non supervisionato**
- PCA e proiezione sulle prime due componenti;
- k-means con 2 cluster, sulle prime due componenti e su tutte le feature;
- proiezione con t-SNE;
- clustering con DBSCAN.

**🤖 3. Apprendimento supervisionato**
- divisione stratificata in training set (976) e test set (372);
- regressione logistica, con scelta di C tramite cross-validation;
- albero di decisione ID3 (implementazione propria), con scelta del numero di bin tramite cross-validation;
- Naive Bayes gaussiano;
- k-NN, con scelta di k tramite cross-validation;
- confronto finale sul test set con accuratezza, precisione, recall e F1-score.

## 🏆 Risultati sul test set

| Modello | Accuratezza | Precisione | Recall | F1-score |
|---|---|---|---|---|
| Regressione logistica 🥉 | 0.989 | 0.982 | 0.994 | 0.988 |
| ID3 🥈 | 0.995 | 0.988 | 1.000 | 0.994 |
| Naive Bayes | 0.871 | 0.895 | 0.810 | 0.850 |
| k-NN 🥇 | 1.000 | 1.000 | 1.000 | 1.000 |

In breve:
- ❌ k-means non riesce a ritrovare le classi, perché non formano due gruppi compatti;
- ✅ t-SNE e DBSCAN mostrano che le banconote vicine tra loro appartengono quasi sempre alla stessa classe;
- 🎯 per questo k-NN ottiene i risultati migliori, mentre Naive Bayes è penalizzato dalla correlazione tra le feature.

## ▶️ Come eseguire il notebook

1. Installa le librerie necessarie:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
2. Metti il file `data_banknote_authentication.txt` nella stessa cartella del notebook.
3. Apri `challenge_one_ita.ipynb` ed esegui tutte le celle in ordine (*Restart → Run All*).

## 📁 File

| File | Contenuto |
|---|---|
| `challenge_one_ita.ipynb` | notebook con codice, grafici e commenti |
| `data_banknote_authentication.txt` | dataset |
| `README.md` | questo file |

## 📝 Note

- 🌳 L'implementazione di ID3 è basata su quella vista nel Lab-05 e adattata ad array numpy. Siccome ID3 lavora solo con feature discrete, le feature continue vengono prima divise in intervalli (bin).
- 🎲 Tutti i risultati sono riproducibili grazie al seed fisso (`random_state=42`).