import pandas as pd
import numpy as np

"""
prende tutte le cartelle 

    estrai nome modello
    estrai configurazione
    leggi pkl distance e best distance 
    
    {
        'modello0':
            'conf1 opt_sch...': robust,
            ...
        'modello1':
            ...
    }

compute_robust su ogni esperimento 
    chiamare funzione
    leggere valore a 8/255
    2aggiungerlo a dizionario['modello']['configurazione']

}
"""
