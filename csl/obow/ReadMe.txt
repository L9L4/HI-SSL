v0: versione di obow modificata a Roma con l'aggiunta di un dataloader per dati diversi da ImageNet
v1: versione di obow scaricata dal cluster il 15/03/2021, prima dell'aggiunta delle mie augmentation e quindi della rimozione di quelle esistenti
v2: versione di obow modificata il 16/03/2021, con l'aggiunta delle mie augmentation, la rimozione di quelle esistenti e la correzione del file .yaml
v3: versione di obow modificata il 29/04/2021, da non considerare per future prove, per valutare la qualità del pretraining di OBOW nel caso in cui non sia garantito l'overlap fra target patch e crop
v4: versione modificata a partire dall'11/05/2021, basata sulla versione v3, che è risultata migliore della versione v2, e pensata per i manoscritti binarizzati
v5: versione modificata il 16/05/2021, con patch più piccole, a partire dalla versione v3, con la possibilità di abilitare l'overlap e regolare le patch dall'esterno, e di generare media e deviazione standard a partire dal dataset di training 