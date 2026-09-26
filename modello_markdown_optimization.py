"""
@author: gpolirpo
"""

import pandas as pd
from itertools import combinations_with_replacement 
import numpy as np
import time

# parametri di esecuzione del modello
NUM_SIMULAZIONI_MONTECARLO = 100
NUM_SIMULAZIONI_TEST = 10
NUM_SIMULAZIONI_SENSIBILITA = 30

tempo_inizio = time.perf_counter()


#  1. DATI DI INPUT E ASSUNZIONI DEI CASI STUDIO

# dati dei tre prodotti scelti come casi studio per testare il modello
dati_prodotti = {
    "prodotto": [
        "Sushi Roll",
        "Giovanni Rana Sfogliavelo",
        "FAGE Total 0%"
    ],
    "formato_grammi": [250, 250, 150],
    "numero_periodi": [3, 5, 5],
    "prezzo_lordo": [9.99, 3.69, 1.55],
    "iva": [0.10, 0.04, 0.10],
    "beta": [2.0, 1.5, 1.2],
    "stock_iniziale": [14, 20, 30],
    "incertezza_min": [0.60, 0.70, 0.80],
    "incertezza_moda": [1.00, 1.00, 1.00],
    "incertezza_max": [1.40, 1.30, 1.20],
    "costo_smaltimento": [0.0, 0.0, 0.0]
}
df_prodotti = pd.DataFrame(dati_prodotti)

df_prodotti["prezzo_netto"] = (
    df_prodotti["prezzo_lordo"] / (1 + df_prodotti["iva"])
)

# costo unitario stimato (incidenza 70%) 
incidenza_costo = 0.70            
df_prodotti["costo_unitario"] = (
    df_prodotti["prezzo_netto"] * incidenza_costo
)                                                

df_prodotti["margine_unitario"] = (
    df_prodotti["prezzo_netto"] - df_prodotti["costo_unitario"]
)


# dati relativi ai singoli periodi temporali dei tre prodotti
dati_periodi = {
    "prodotto": (
        ["Sushi Roll"] * 3
        + ["Giovanni Rana Sfogliavelo"] * 5
        + ["FAGE Total 0%"] * 5
    ),

    "periodo": [
        1, 2, 3,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5
    ],

    "descrizione_periodo": [
        "Pranzo",
        "Pomeriggio",
        "Sera",
        "5 giorni residui",
        "4 giorni residui",
        "3 giorni residui",
        "2 giorni residui",
        "1 giorno residuo",
        "9-10 giorni residui",
        "7-8 giorni residui",
        "5-6 giorni residui",
        "3-4 giorni residui",
        "1-2 giorni residui"
    ],

    "domanda_base": [
        6, 2, 4,
        4, 4, 4, 4, 4,
        6, 6, 6, 6, 6
    ],

    "fattore_freschezza": [
        1.00, 1.00, 1.00,
        1.00, 0.95, 0.90, 0.80, 0.70,
        1.00, 1.00, 0.95, 0.85, 0.65
    ]
}

df_periodi = pd.DataFrame(dati_periodi)


# scenari generali della domanda
dati_scenari = {
    "scenario": [
        "Sfavorevole",
        "Ordinario",
        "Favorevole"
    ],
    "fattore_scenario": [
        0.85,
        1.00,
        1.15
    ]
}

df_scenari = pd.DataFrame(dati_scenari)


# controlli di coerenza dei dati
# i prodotti devono essere gli stessi nella tabella prodotti e nella tabella periodi
assert set(df_prodotti["prodotto"]) == set(df_periodi["prodotto"]), (
    "Errore: i nomi dei prodotti non coincidono tra le due tabelle."
)
# il numero di periodi presenti deve coincidere con quello previsto per ogni prodotto
periodi_presenti = df_periodi.groupby("prodotto").size()
periodi_previsti = df_prodotti.set_index("prodotto")["numero_periodi"]
assert periodi_presenti.sort_index().equals(periodi_previsti.sort_index()), (
    "Errore: il numero di periodi non coincide con quello previsto."
)
# prezzi, costi e stock devono essere positivi
assert (df_prodotti["prezzo_netto"] > 0).all(), (
    "Errore: esiste un prezzo netto non positivo."
)
assert (df_prodotti["costo_unitario"] > 0).all(), (
    "Errore: esiste un costo unitario non positivo."
)
assert (df_prodotti["stock_iniziale"] > 0).all(), (
    "Errore: esiste uno stock iniziale non positivo."
)
# il costo unitario deve essere inferiore al prezzo netto
assert (
    df_prodotti["costo_unitario"] < df_prodotti["prezzo_netto"]
).all(), "Errore: un costo unitario è uguale o superiore al prezzo netto."
# il margine deve coincidere con prezzo netto meno costo unitario
differenza_margine = (
    df_prodotti["prezzo_netto"]
    - df_prodotti["costo_unitario"]
    - df_prodotti["margine_unitario"]
).abs()
assert (differenza_margine < 0.000001).all(), (
    "Errore: il margine unitario non è stato calcolato correttamente."
)
# domanda base e fattori di freschezza devono avere valori validi
assert (df_periodi["domanda_base"] >= 0).all(), (
    "Errore: esiste una domanda base negativa."
)
assert df_periodi["fattore_freschezza"].between(0, 1).all(), (
    "Errore: un fattore di freschezza non è compreso tra 0 e 1."
)
# i limiti della distribuzione triangolare devono essere ordinati correttamente
assert (
    df_prodotti["incertezza_min"]
    <= df_prodotti["incertezza_moda"]
).all(), "Errore: il minimo dell'incertezza supera la moda."
assert (
    df_prodotti["incertezza_moda"]
    <= df_prodotti["incertezza_max"]
).all(), "Errore: la moda dell'incertezza supera il massimo."
# i fattori degli scenari devono essere positivi
assert (df_scenari["fattore_scenario"] > 0).all(), (
    "Errore: esiste un fattore di scenario non positivo."
)


# 2. FUNZIONI DEL MOTORE GENERALE DEL MODELLO

# funzione della domanda attesa
def calcola_domanda_attesa(
    domanda_base,
    fattore_freschezza,
    fattore_scenario,
    beta,
    sconto
):
    domanda_attesa = (
        domanda_base
        * fattore_freschezza
        * fattore_scenario
        * (1 + beta * sconto)
    )
    return domanda_attesa

# test della funzione della domanda
domanda_test = calcola_domanda_attesa(
    domanda_base=4,
    fattore_freschezza=1.00,
    fattore_scenario=1.00,
    beta=2.00,
    sconto=0.20
)
assert abs(domanda_test - 5.6) < 0.000001, (
    "Errore: la funzione della domanda non restituisce il risultato atteso."
)

# livelli di sconto analizzati
dati_sconti = {
    "sconto": [
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50
    ]
}
df_sconti = pd.DataFrame(dati_sconti)

# collegamento della sensibilità allo sconto ai singoli periodi
df_periodi = df_periodi.merge(
    df_prodotti[["prodotto", "beta"]],
    on="prodotto",
    how="left"
)

# controllo
assert df_periodi["beta"].notna().all(), (
    "Errore: manca il beta per uno o più periodi."
)

# tabella con combinazione di periodi, scenari e livelli di sconto
df_combinazioni = (
    df_periodi
    .merge(df_scenari, how="cross")
    .merge(df_sconti, how="cross")
)

# controllo
assert len(df_combinazioni) == 234, (
    "Errore: il numero delle combinazioni non è corretto."
)

# calcolo della domanda attesa per ogni combinazione (234) - come nuova colonna di df_combinazioni
df_combinazioni["domanda_attesa"] = df_combinazioni.apply(
    lambda riga: calcola_domanda_attesa(
        domanda_base=riga["domanda_base"],
        fattore_freschezza=riga["fattore_freschezza"],
        fattore_scenario=riga["fattore_scenario"],
        beta=riga["beta"],
        sconto=riga["sconto"]
    ),
    axis=1
)

# controllo
assert df_combinazioni["domanda_attesa"].notna().all(), (
    "Errore: esistono valori mancanti nella domanda attesa."
)
assert (df_combinazioni["domanda_attesa"] >= 0).all(), (
    "Errore: esiste una domanda attesa negativa."
)


# funzione di vendita e aggiornamento dello stock
def aggiorna_stock(stock_disponibile, domanda_attesa):
    vendite = min(stock_disponibile, domanda_attesa)

    domanda_persa = max(
        domanda_attesa - stock_disponibile,
        0
    )

    stock_finale = stock_disponibile - vendite

    return vendite, domanda_persa, stock_finale

# test della funzione di aggiornamento dello stock
vendite_test, domanda_persa_test, stock_finale_test = aggiorna_stock(
    stock_disponibile=3,
    domanda_attesa=5.6
)

assert abs(vendite_test - 3) < 0.000001, (
    "Errore: le vendite non sono state calcolate correttamente."
)

assert abs(domanda_persa_test - 2.6) < 0.000001, (
    "Errore: la domanda persa non è stata calcolata correttamente."
)

assert abs(stock_finale_test - 0) < 0.000001, (
    "Errore: lo stock finale non è stato calcolato correttamente."
)


# funzione di simulazione multiperiodale per un singolo prodotto
def simula_prodotto(nome_prodotto, nome_scenario, percorso_sconti):
    
    dati_prodotto = df_prodotti.loc[
        df_prodotti["prodotto"] == nome_prodotto
    ]
    
    assert len(dati_prodotto) == 1, (
        "Errore: il prodotto indicato non è presente in df_prodotti."
    )
    
    stock_corrente = dati_prodotto["stock_iniziale"].iloc[0]
    
    periodi_prodotto = df_periodi.loc[
        df_periodi["prodotto"] == nome_prodotto
    ].sort_values("periodo")
    
    assert len(percorso_sconti) == len(periodi_prodotto), (
        "Errore: il numero degli sconti non coincide con il numero dei periodi."
    )
    
    dati_scenario = df_scenari.loc[
        df_scenari["scenario"] == nome_scenario
    ]
    
    assert len(dati_scenario) == 1, (
        "Errore: lo scenario indicato non è presente in df_scenari."
    )
    
    fattore_scenario = dati_scenario["fattore_scenario"].iloc[0]
    
    risultati = []
    
    for posizione, riga in enumerate(periodi_prodotto.itertuples()):
        
        sconto = percorso_sconti[posizione]
        
        domanda_attesa = calcola_domanda_attesa(
            domanda_base=riga.domanda_base,
            fattore_freschezza=riga.fattore_freschezza,
            fattore_scenario=fattore_scenario,
            beta=riga.beta,
            sconto=sconto
        )
        
        vendite, domanda_persa, stock_finale = aggiorna_stock(
            stock_disponibile=stock_corrente,
            domanda_attesa=domanda_attesa
        )
        
        risultati.append({
            "prodotto": nome_prodotto,
            "scenario": nome_scenario,
            "periodo": riga.periodo,
            "descrizione_periodo": riga.descrizione_periodo,
            "stock_iniziale_periodo": stock_corrente,
            "sconto": sconto,
            "sconto_percentuale": sconto * 100,
            "domanda_attesa": domanda_attesa,
            "vendite": vendite,
            "domanda_persa": domanda_persa,
            "stock_finale_periodo": stock_finale
        })
        
        stock_corrente = stock_finale
    
    df_risultati = pd.DataFrame(risultati)
    
    df_risultati = df_risultati.round(2)
    
    return df_risultati

# test della simulazione multiperiodale sul Sushi Roll
df_simulazione_test = simula_prodotto(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario",
    percorso_sconti=[0.00, 0.20, 0.40]
)
assert len(df_simulazione_test) == 3, (
    "Errore: la simulazione del sushi dovrebbe avere 3 periodi."
)
assert (df_simulazione_test["stock_finale_periodo"] >= 0).all(), (
    "Errore: lo stock finale non può essere negativo."
)
assert (
    df_simulazione_test["vendite"]
    <= df_simulazione_test["stock_iniziale_periodo"]
).all(), "Errore: le vendite non possono superare lo stock disponibile."

# controlli finali della simulazione di test
stock_iniziale_test = df_simulazione_test["stock_iniziale_periodo"].iloc[0]
vendite_totali_test = df_simulazione_test["vendite"].sum()
spreco_finale_test = df_simulazione_test["stock_finale_periodo"].iloc[-1]
assert spreco_finale_test >= 0, (
    "Errore: lo spreco finale non può essere negativo."
)
assert abs(
    vendite_totali_test + spreco_finale_test - stock_iniziale_test
) < 0.000001, (
    "Errore: vendite totali e spreco finale non tornano con lo stock iniziale."
)
stock_finale_periodo_precedente = (
    df_simulazione_test["stock_finale_periodo"]
    .iloc[:-1]
    .reset_index(drop=True)
)
stock_iniziale_periodo_successivo = (
    df_simulazione_test["stock_iniziale_periodo"]
    .iloc[1:]
    .reset_index(drop=True)
)
assert (
    abs(stock_finale_periodo_precedente - stock_iniziale_periodo_successivo)
    < 0.000001
).all(), (
    "Errore: lo stock finale di un periodo non coincide con lo stock iniziale del periodo successivo."
)

    
# funzione per generare le strategie di markdown ammissibili
def genera_strategie_markdown(nome_prodotto):
    
    dati_prodotto = df_prodotti.loc[
        df_prodotti["prodotto"] == nome_prodotto
    ]
    
    assert len(dati_prodotto) == 1, (
        "Errore: il prodotto indicato non è presente in df_prodotti."
    )
    
    numero_periodi = int(dati_prodotto["numero_periodi"].iloc[0])
    
    livelli_sconto = sorted(df_sconti["sconto"].tolist())
    
    strategie = combinations_with_replacement(
        livelli_sconto,
        numero_periodi
    )
    
    risultati = []
    
    for numero_strategia, percorso in enumerate(strategie, start=1):
        
        riga = {
            "strategia_id": f"STR_{numero_strategia:03d}",
            "percorso_sconti": list(percorso)
        }
        
        for periodo, sconto in enumerate(percorso, start=1):
            riga[f"sconto_p{periodo}"] = sconto
            riga[f"sconto_p{periodo}_percentuale"] = sconto * 100
        
        risultati.append(riga)
    
    df_strategie = pd.DataFrame(risultati)
    
    return df_strategie


# 3. APPLICAZIONE DEL MODELLO AI CASI STUDIO 

# generazione delle strategie di markdown per il Sushi Roll
df_strategie_sushi = genera_strategie_markdown("Sushi Roll")

assert len(df_strategie_sushi) == 56, (
    "Errore: il numero di strategie generate per il sushi non è corretto."
)
# generazione delle strategie di markdown per Giovanni Rana Sfogliavelo
df_strategie_ravioli = genera_strategie_markdown("Giovanni Rana Sfogliavelo")
assert len(df_strategie_ravioli) == 252, (
    "Errore: il numero di strategie generate per i ravioli non è corretto."
)
# generazione delle strategie di markdown per FAGE Total 0%
df_strategie_fage = genera_strategie_markdown("FAGE Total 0%")
assert len(df_strategie_fage) == 252, (
    "Errore: il numero di strategie generate per FAGE non è corretto."
)

# funzione per simulare tutte le strategie di un prodotto
def simula_tutte_le_strategie(nome_prodotto, nome_scenario, df_strategie):
    
    risultati = []
    
    for strategia in df_strategie.itertuples():
        
        df_simulazione = simula_prodotto(
            nome_prodotto=nome_prodotto,
            nome_scenario=nome_scenario,
            percorso_sconti=strategia.percorso_sconti
        )
        
        vendite_totali = df_simulazione["vendite"].sum()
        domanda_persa_totale = df_simulazione["domanda_persa"].sum()
        spreco_finale = df_simulazione["stock_finale_periodo"].iloc[-1]
        
        risultati.append({
            "strategia_id": strategia.strategia_id,
            "percorso_sconti": strategia.percorso_sconti,
            "vendite_totali": vendite_totali,
            "domanda_persa_totale": domanda_persa_totale,
            "spreco_finale": spreco_finale
        })
    
    df_confronto = pd.DataFrame(risultati)
    
    df_confronto = df_confronto.round(2)
    
    return df_confronto

# confronto operativo delle strategie per il Sushi Roll
df_confronto_sushi = simula_tutte_le_strategie(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario",
    df_strategie=df_strategie_sushi
)
assert len(df_confronto_sushi) == len(df_strategie_sushi), (
    "Errore: non tutte le strategie del sushi sono state simulate."
)
assert (df_confronto_sushi["spreco_finale"] >= 0).all(), (
    "Errore: lo spreco finale del sushi non può essere negativo."
)
# confronto operativo delle strategie per Giovanni Rana Sfogliavelo
df_confronto_ravioli = simula_tutte_le_strategie(
    nome_prodotto="Giovanni Rana Sfogliavelo",
    nome_scenario="Ordinario",
    df_strategie=df_strategie_ravioli
)
assert len(df_confronto_ravioli) == len(df_strategie_ravioli), (
    "Errore: non tutte le strategie dei ravioli sono state simulate."
)
assert (df_confronto_ravioli["spreco_finale"] >= 0).all(), (
    "Errore: lo spreco finale dei ravioli non può essere negativo."
)
# confronto operativo delle strategie per FAGE Total 0%
df_confronto_fage = simula_tutte_le_strategie(
    nome_prodotto="FAGE Total 0%",
    nome_scenario="Ordinario",
    df_strategie=df_strategie_fage
)
assert len(df_confronto_fage) == len(df_strategie_fage), (
    "Errore: non tutte le strategie di FAGE sono state simulate."
)
assert (df_confronto_fage["spreco_finale"] >= 0).all(), (
    "Errore: lo spreco finale di FAGE non può essere negativo."
)

# strategie manuali di riferimento
def estrai_strategie_manuali(df_confronto, strategie_manuali):
    
    risultati = []
    
    for nome_strategia, percorso in strategie_manuali.items():
        
        riga_strategia = df_confronto.loc[
            df_confronto["percorso_sconti"].apply(lambda x: x == percorso)
        ].copy()
        
        assert len(riga_strategia) == 1, (
            "Errore: una strategia manuale non è stata trovata nel confronto."
        )
        
        riga_strategia["strategia_manuale"] = nome_strategia
        
        risultati.append(riga_strategia)
    
    df_manuali = pd.concat(risultati, ignore_index=True)
    
    df_manuali = df_manuali[
        [
            "strategia_manuale",
            "strategia_id",
            "percorso_sconti",
            "vendite_totali",
            "domanda_persa_totale",
            "spreco_finale"
        ]
    ]
    
    return df_manuali

# strategie manuali per il Sushi Roll
strategie_manuali_sushi = {
    "Nessuno sconto": [0.00, 0.00, 0.00],
    "Markdown progressivo leggero": [0.00, 0.10, 0.20],
    "Markdown progressivo intermedio": [0.00, 0.20, 0.40],
    "Markdown aggressivo": [0.10, 0.30, 0.50]
}
df_manuali_sushi = estrai_strategie_manuali(
    df_confronto=df_confronto_sushi,
    strategie_manuali=strategie_manuali_sushi
)
# strategie manuali per i prodotti con cinque periodi (Giovanni Rana Sfogliavelo e FAGE Total 0%)
strategie_manuali_5_periodi = {
    "Nessuno sconto": [0.00, 0.00, 0.00, 0.00, 0.00],
    "Markdown tardivo": [0.00, 0.00, 0.10, 0.20, 0.30],
    "Markdown progressivo": [0.00, 0.10, 0.20, 0.30, 0.40],
    "Markdown aggressivo": [0.10, 0.20, 0.30, 0.40, 0.50]
}
df_manuali_ravioli = estrai_strategie_manuali(
    df_confronto=df_confronto_ravioli,
    strategie_manuali=strategie_manuali_5_periodi
)
df_manuali_fage = estrai_strategie_manuali(
    df_confronto=df_confronto_fage,
    strategie_manuali=strategie_manuali_5_periodi
)

# controlli finali sulle strategie di markdown
def percorso_non_decrescente(percorso):
    return all(
        percorso[posizione] >= percorso[posizione - 1]
        for posizione in range(1, len(percorso))
    )
assert df_strategie_sushi["percorso_sconti"].apply(percorso_non_decrescente).all(), (
    "Errore: esiste una strategia del sushi con sconti decrescenti."
)
assert df_strategie_ravioli["percorso_sconti"].apply(percorso_non_decrescente).all(), (
    "Errore: esiste una strategia dei ravioli con sconti decrescenti."
)
assert df_strategie_fage["percorso_sconti"].apply(percorso_non_decrescente).all(), (
    "Errore: esiste una strategia di FAGE con sconti decrescenti."
)
livelli_sconto_ammessi = set(df_sconti["sconto"].tolist())
def usa_livelli_sconto_ammessi(percorso):
    return all(
        sconto in livelli_sconto_ammessi
        for sconto in percorso
    )
assert df_strategie_sushi["percorso_sconti"].apply(usa_livelli_sconto_ammessi).all(), (
    "Errore: una strategia del sushi usa uno sconto non ammesso."
)
assert df_strategie_ravioli["percorso_sconti"].apply(usa_livelli_sconto_ammessi).all(), (
    "Errore: una strategia dei ravioli usa uno sconto non ammesso."
)
assert df_strategie_fage["percorso_sconti"].apply(usa_livelli_sconto_ammessi).all(), (
    "Errore: una strategia di FAGE usa uno sconto non ammesso."
)


# funzione per valutare economicamente una strategia di markdown 
def valuta_strategia_economica(nome_prodotto, nome_scenario, percorso_sconti, strategia_id=None):
    
    dati_prodotto = df_prodotti.loc[
        df_prodotti["prodotto"] == nome_prodotto
    ]
    
    assert len(dati_prodotto) == 1, (
        "Errore: il prodotto indicato non è presente in df_prodotti."
    )
    
    prezzo_netto = dati_prodotto["prezzo_netto"].iloc[0]
    costo_unitario = dati_prodotto["costo_unitario"].iloc[0]
    stock_iniziale = dati_prodotto["stock_iniziale"].iloc[0]
    costo_smaltimento = dati_prodotto["costo_smaltimento"].iloc[0]
    
    df_simulazione = simula_prodotto(
        nome_prodotto=nome_prodotto,
        nome_scenario=nome_scenario,
        percorso_sconti=percorso_sconti
    )
    
    df_simulazione["prezzo_scontato"] = (
        prezzo_netto * (1 - df_simulazione["sconto"])
    )
    
    df_simulazione["ricavi_periodo"] = (
        df_simulazione["vendite"] * df_simulazione["prezzo_scontato"]
    )
    
    df_simulazione["costo_venduto_periodo"] = (
        df_simulazione["vendite"] * costo_unitario
    )
    
    df_simulazione["margine_periodo"] = (
        df_simulazione["ricavi_periodo"]
        - df_simulazione["costo_venduto_periodo"]
    )
    
    vendite_totali = df_simulazione["vendite"].sum()
    domanda_persa_totale = df_simulazione["domanda_persa"].sum()
    ricavi_totali = df_simulazione["ricavi_periodo"].sum()
    margine_vendite = df_simulazione["margine_periodo"].sum()
    
    invenduto_finale = df_simulazione["stock_finale_periodo"].iloc[-1]
    
    valore_invenduto = invenduto_finale * costo_unitario
    
    costo_smaltimento_totale = invenduto_finale * costo_smaltimento
    
    costo_spreco_totale = (
        valore_invenduto
        + costo_smaltimento_totale
    )
    
    risultato_economico = (
        margine_vendite
        - costo_spreco_totale
    )
    
    sell_through = vendite_totali / stock_iniziale
    
    tasso_invenduto = invenduto_finale / stock_iniziale
    
    df_valutazione = pd.DataFrame({
        "strategia_id": [strategia_id],
        "prodotto": [nome_prodotto],
        "scenario": [nome_scenario],
        "percorso_sconti": [percorso_sconti],
        "stock_iniziale": [stock_iniziale],
        "vendite_totali": [vendite_totali],
        "domanda_persa_totale": [domanda_persa_totale],
        "sell_through_percentuale": [sell_through * 100],
        "invenduto_finale": [invenduto_finale],
        "tasso_invenduto_percentuale": [tasso_invenduto * 100],
        "ricavi_totali": [ricavi_totali],
        "margine_vendite": [margine_vendite],
        "valore_invenduto": [valore_invenduto],
        "costo_smaltimento_totale": [costo_smaltimento_totale],
        "costo_spreco_totale": [costo_spreco_totale],
        "risultato_economico": [risultato_economico]
    })
    
    df_valutazione = df_valutazione.round(2)
    
    return df_valutazione

# test della valutazione economica su una strategia del Sushi Roll
df_valutazione_test = valuta_strategia_economica(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario",
    percorso_sconti=[0.00, 0.20, 0.40],
    strategia_id="TEST"
)
assert len(df_valutazione_test) == 1, (
    "Errore: la valutazione economica dovrebbe produrre una sola riga."
)
assert df_valutazione_test["ricavi_totali"].iloc[0] >= 0, (
    "Errore: i ricavi totali non possono essere negativi."
)
assert df_valutazione_test["sell_through_percentuale"].iloc[0] <= 100, (
    "Errore: il sell-through non può superare il 100%."
)

# funzione per valutare economicamente tutte le strategie possibili
def valuta_tutte_le_strategie_economiche(nome_prodotto, nome_scenario, df_strategie):
    
    risultati = []
    
    for strategia in df_strategie.itertuples():
        
        df_valutazione = valuta_strategia_economica(
            nome_prodotto=nome_prodotto,
            nome_scenario=nome_scenario,
            percorso_sconti=strategia.percorso_sconti,
            strategia_id=strategia.strategia_id
        )
        
        risultati.append(df_valutazione)
    
    df_valutazioni = pd.concat(risultati, ignore_index=True)
    
    return df_valutazioni

# valutazione economica di tutte le strategie per il Sushi Roll
df_valutazione_sushi = valuta_tutte_le_strategie_economiche(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario",
    df_strategie=df_strategie_sushi
)
assert len(df_valutazione_sushi) == len(df_strategie_sushi), (
    "Errore: non tutte le strategie del sushi sono state valutate economicamente."
)
# valutazione economica di tutte le strategie per Giovanni Rana Sfogliavelo
df_valutazione_ravioli = valuta_tutte_le_strategie_economiche(
    nome_prodotto="Giovanni Rana Sfogliavelo",
    nome_scenario="Ordinario",
    df_strategie=df_strategie_ravioli
)
assert len(df_valutazione_ravioli) == len(df_strategie_ravioli), (
    "Errore: non tutte le strategie dei ravioli sono state valutate economicamente."
)
# valutazione economica di tutte le strategie per FAGE Total 0%
df_valutazione_fage = valuta_tutte_le_strategie_economiche(
    nome_prodotto="FAGE Total 0%",
    nome_scenario="Ordinario",
    df_strategie=df_strategie_fage
)
assert len(df_valutazione_fage) == len(df_strategie_fage), (
    "Errore: non tutte le strategie di FAGE sono state valutate economicamente."
)


# funzione per selezionare la strategia economicamente migliore
def seleziona_strategia_migliore(df_valutazioni):
    
    df_ordinate = df_valutazioni.copy()
    
    df_ordinate["sconto_medio_percentuale"] = (
        df_ordinate["percorso_sconti"]
        .apply(lambda percorso: sum(percorso) / len(percorso) * 100)
    )
    
    df_ordinate = df_ordinate.sort_values(
        by=[
            "risultato_economico",
            "tasso_invenduto_percentuale",
            "sconto_medio_percentuale"
        ],
        ascending=[
            False,
            True,
            True
        ]
    ).reset_index(drop=True)
    
    df_migliore = df_ordinate.head(1).copy()
    
    df_migliore["raccomandazione_modello"] = (
        "Strategia raccomandata nella situazione considerata"
    )
    
    df_migliore["criterio_scelta"] = (
        "Massimo risultato economico; in caso di parità, minore invenduto "
        "e minore sconto medio"
    )
    
    return df_migliore


# FUNZIONE GENERALE DI RACCOMANDAZIONE MARKDOWN

def raccomanda_markdown(nome_prodotto, nome_scenario):
    
    assert nome_prodotto in df_prodotti["prodotto"].tolist(), (
        "Errore: il prodotto indicato non è presente nei dati."
    )
    
    assert nome_scenario in df_scenari["scenario"].tolist(), (
        "Errore: lo scenario indicato non è presente nei dati."
    )
    
    df_strategie = genera_strategie_markdown(nome_prodotto)
    
    df_valutazioni = valuta_tutte_le_strategie_economiche(
        nome_prodotto=nome_prodotto,
        nome_scenario=nome_scenario,
        df_strategie=df_strategie
    )
    
    df_raccomandazione = seleziona_strategia_migliore(
        df_valutazioni
    )
    
    df_raccomandazione["numero_strategie_valutate"] = len(df_valutazioni)
    
    df_raccomandazione = df_raccomandazione[
        [
            "raccomandazione_modello",
            "criterio_scelta",
            "prodotto",
            "scenario",
            "strategia_id",
            "percorso_sconti",
            "numero_strategie_valutate",
            "sconto_medio_percentuale",
            "stock_iniziale",
            "vendite_totali",
            "sell_through_percentuale",
            "invenduto_finale",
            "tasso_invenduto_percentuale",
            "ricavi_totali",
            "margine_vendite",
            "costo_spreco_totale",
            "risultato_economico"
        ]
    ]
    
    return df_raccomandazione

# test della funzione generale di raccomandazione markdown
df_raccomandazione_test = raccomanda_markdown(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario"
)
assert len(df_raccomandazione_test) == 1, (
    "Errore: la funzione dovrebbe restituire una sola raccomandazione."
)
assert df_raccomandazione_test["prodotto"].iloc[0] == "Sushi Roll", (
    "Errore: il prodotto della raccomandazione non coincide con quello richiesto."
)
assert df_raccomandazione_test["scenario"].iloc[0] == "Ordinario", (
    "Errore: lo scenario della raccomandazione non coincide con quello richiesto."
)
assert df_raccomandazione_test["numero_strategie_valutate"].iloc[0] == 56, (
    "Errore: il numero di strategie valutate per il sushi non è corretto."
)

# applicazione della raccomandazione ai casi studio
def raccomanda_markdown_casi_studio(lista_prodotti, lista_scenari):
    
    risultati = []
    
    for prodotto in lista_prodotti:
        
        for scenario in lista_scenari:
            
            df_raccomandazione = raccomanda_markdown(
                nome_prodotto=prodotto,
                nome_scenario=scenario
            )
            
            risultati.append(df_raccomandazione)
    
    df_raccomandazioni = pd.concat(
        risultati,
        ignore_index=True
    )
    
    return df_raccomandazioni

# prodotti e scenari utilizzati nell'applicazione del modello
prodotti_casi_studio = [
    "Sushi Roll",
    "Giovanni Rana Sfogliavelo",
    "FAGE Total 0%"
]
scenari_casi_studio = df_scenari["scenario"].tolist()

# raccomandazioni del modello per prodotti e scenari dei casi studio
df_raccomandazioni_casi_studio = raccomanda_markdown_casi_studio(
    lista_prodotti=prodotti_casi_studio,
    lista_scenari=scenari_casi_studio
)

# controlli
assert len(df_raccomandazioni_casi_studio) == 9, (
    "Errore: dovrebbero esserci 9 raccomandazioni, "
    "una per ogni combinazione prodotto-scenario."
)
assert (
    df_raccomandazioni_casi_studio
    .groupby(["prodotto", "scenario"])
    .size()
    .eq(1)
    .all()
), (
    "Errore: dovrebbe esserci una sola raccomandazione per ogni prodotto e scenario."
)
assert df_raccomandazioni_casi_studio["risultato_economico"].notna().all(), (
    "Errore: esiste una raccomandazione senza risultato economico."
)
assert df_raccomandazioni_casi_studio["sell_through_percentuale"].between(0, 100).all(), (
    "Errore: il sell-through deve essere compreso tra 0% e 100%."
)
assert df_raccomandazioni_casi_studio["tasso_invenduto_percentuale"].between(0, 100).all(), (
    "Errore: il tasso di invenduto deve essere compreso tra 0% e 100%."
)
assert (df_raccomandazioni_casi_studio["invenduto_finale"] >= 0).all(), (
    "Errore: l'invenduto finale non può essere negativo."
)


# INCERTEZZA DELLA DOMANDA

# funzione per generare la domanda reale a partire dalla domanda attesa
def genera_domanda_reale(
    domanda_attesa,
    incertezza_min,
    incertezza_moda,
    incertezza_max,
    generatore_casuale
):
    
    fattore_incertezza = generatore_casuale.triangular(
        left=incertezza_min,
        mode=incertezza_moda,
        right=incertezza_max
    )
    
    domanda_reale = domanda_attesa * fattore_incertezza
    
    domanda_reale = int(round(domanda_reale))
    
    domanda_reale = max(domanda_reale, 0)
    
    return domanda_reale, fattore_incertezza

# test della funzione di generazione della domanda reale
generatore_test = np.random.default_rng(123)
domanda_reale_test, fattore_incertezza_test = genera_domanda_reale(
    domanda_attesa=10,
    incertezza_min=0.80,
    incertezza_moda=1.00,
    incertezza_max=1.20,
    generatore_casuale=generatore_test
)
assert domanda_reale_test >= 0, (
    "Errore: la domanda reale non può essere negativa."
)
assert 0.80 <= fattore_incertezza_test <= 1.20, (
    "Errore: il fattore di incertezza non è compreso nei limiti previsti."
)


# funzione di simulazione stocastica per una singola strategia
def simula_prodotto_stocastico(
    nome_prodotto,
    nome_scenario,
    percorso_sconti,
    generatore_casuale
):
    
    dati_prodotto = df_prodotti.loc[
        df_prodotti["prodotto"] == nome_prodotto
    ]
    
    assert len(dati_prodotto) == 1, (
        "Errore: il prodotto indicato non è presente in df_prodotti."
    )
    
    stock_corrente = dati_prodotto["stock_iniziale"].iloc[0]
    
    incertezza_min = dati_prodotto["incertezza_min"].iloc[0]
    incertezza_moda = dati_prodotto["incertezza_moda"].iloc[0]
    incertezza_max = dati_prodotto["incertezza_max"].iloc[0]
    
    periodi_prodotto = df_periodi.loc[
        df_periodi["prodotto"] == nome_prodotto
    ].sort_values("periodo")
    
    assert len(percorso_sconti) == len(periodi_prodotto), (
        "Errore: il numero degli sconti non coincide con il numero dei periodi."
    )
    
    dati_scenario = df_scenari.loc[
        df_scenari["scenario"] == nome_scenario
    ]
    
    assert len(dati_scenario) == 1, (
        "Errore: lo scenario indicato non è presente in df_scenari."
    )
    
    fattore_scenario = dati_scenario["fattore_scenario"].iloc[0]
    
    risultati = []
    
    for posizione, riga in enumerate(periodi_prodotto.itertuples()):
        
        sconto = percorso_sconti[posizione]
        
        domanda_attesa = calcola_domanda_attesa(
            domanda_base=riga.domanda_base,
            fattore_freschezza=riga.fattore_freschezza,
            fattore_scenario=fattore_scenario,
            beta=riga.beta,
            sconto=sconto
        )
        
        domanda_reale, fattore_incertezza = genera_domanda_reale(
            domanda_attesa=domanda_attesa,
            incertezza_min=incertezza_min,
            incertezza_moda=incertezza_moda,
            incertezza_max=incertezza_max,
            generatore_casuale=generatore_casuale
        )
        
        vendite, domanda_persa, stock_finale = aggiorna_stock(
            stock_disponibile=stock_corrente,
            domanda_attesa=domanda_reale
        )
        
        risultati.append({
            "prodotto": nome_prodotto,
            "scenario": nome_scenario,
            "periodo": riga.periodo,
            "descrizione_periodo": riga.descrizione_periodo,
            "stock_iniziale_periodo": stock_corrente,
            "sconto": sconto,
            "sconto_percentuale": sconto * 100,
            "domanda_attesa": domanda_attesa,
            "fattore_incertezza": fattore_incertezza,
            "domanda_reale": domanda_reale,
            "vendite": vendite,
            "domanda_persa": domanda_persa,
            "stock_finale_periodo": stock_finale
        })
        
        stock_corrente = stock_finale
    
    df_risultati = pd.DataFrame(risultati)
    
    df_risultati = df_risultati.round(2)
    
    return df_risultati

# test della simulazione stocastica su una strategia del Sushi Roll
generatore_stocastico_test = np.random.default_rng(123)
df_simulazione_stocastica_test = simula_prodotto_stocastico(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario",
    percorso_sconti=[0.00, 0.20, 0.40],
    generatore_casuale=generatore_stocastico_test
)
assert len(df_simulazione_stocastica_test) == 3, (
    "Errore: la simulazione stocastica del sushi dovrebbe avere 3 periodi."
)
assert (df_simulazione_stocastica_test["domanda_reale"] >= 0).all(), (
    "Errore: la domanda reale non può essere negativa."
)
assert (
    df_simulazione_stocastica_test["vendite"]
    <= df_simulazione_stocastica_test["stock_iniziale_periodo"]
).all(), (
    "Errore: le vendite non possono superare lo stock disponibile."
)
assert (df_simulazione_stocastica_test["stock_finale_periodo"] >= 0).all(), (
    "Errore: lo stock finale non può essere negativo."
)


# funzione Monte Carlo per una singola strategia di markdown
# versione ottimizzata: calcola direttamente gli indicatori della simulazione senza creare un DataFrame intermedio per ogni periodo

def montecarlo_singola_strategia(
    nome_prodotto,
    nome_scenario,
    percorso_sconti,
    numero_simulazioni=1000,
    seed=123,
    strategia_id=None
):
    
    dati_prodotto = df_prodotti.loc[
        df_prodotti["prodotto"] == nome_prodotto
    ]
    
    assert len(dati_prodotto) == 1, (
        "Errore: il prodotto indicato non è presente in df_prodotti."
    )
    
    prezzo_netto = dati_prodotto["prezzo_netto"].iloc[0]
    costo_unitario = dati_prodotto["costo_unitario"].iloc[0]
    stock_iniziale = dati_prodotto["stock_iniziale"].iloc[0]
    costo_smaltimento = dati_prodotto["costo_smaltimento"].iloc[0]
    
    incertezza_min = dati_prodotto["incertezza_min"].iloc[0]
    incertezza_moda = dati_prodotto["incertezza_moda"].iloc[0]
    incertezza_max = dati_prodotto["incertezza_max"].iloc[0]
    
    periodi_prodotto = df_periodi.loc[
        df_periodi["prodotto"] == nome_prodotto
    ].sort_values("periodo")
    
    assert len(percorso_sconti) == len(periodi_prodotto), (
        "Errore: il numero degli sconti non coincide con il numero dei periodi."
    )
    
    dati_scenario = df_scenari.loc[
        df_scenari["scenario"] == nome_scenario
    ]
    
    assert len(dati_scenario) == 1, (
        "Errore: lo scenario indicato non è presente in df_scenari."
    )
    
    fattore_scenario = dati_scenario["fattore_scenario"].iloc[0]
    
    generatore_casuale = np.random.default_rng(seed)
    
    risultati = []
    
    periodi_lista = list(periodi_prodotto.itertuples())
    
    for simulazione in range(1, numero_simulazioni + 1):
        
        stock_corrente = stock_iniziale
        vendite_totali = 0
        domanda_persa_totale = 0
        ricavi_totali = 0
        margine_vendite = 0
        
        for posizione, riga in enumerate(periodi_lista):
            
            sconto = percorso_sconti[posizione]
            
            domanda_attesa = calcola_domanda_attesa(
                domanda_base=riga.domanda_base,
                fattore_freschezza=riga.fattore_freschezza,
                fattore_scenario=fattore_scenario,
                beta=riga.beta,
                sconto=sconto
            )
            
            domanda_reale, fattore_incertezza = genera_domanda_reale(
                domanda_attesa=domanda_attesa,
                incertezza_min=incertezza_min,
                incertezza_moda=incertezza_moda,
                incertezza_max=incertezza_max,
                generatore_casuale=generatore_casuale
            )
            
            vendite, domanda_persa, stock_finale = aggiorna_stock(
                stock_disponibile=stock_corrente,
                domanda_attesa=domanda_reale
            )
            
            prezzo_scontato = prezzo_netto * (1 - sconto)
            
            ricavi_periodo = vendite * prezzo_scontato
            
            margine_periodo = (
                ricavi_periodo
                - vendite * costo_unitario
            )
            
            vendite_totali = vendite_totali + vendite
            domanda_persa_totale = domanda_persa_totale + domanda_persa
            ricavi_totali = ricavi_totali + ricavi_periodo
            margine_vendite = margine_vendite + margine_periodo
            
            stock_corrente = stock_finale
        
        invenduto_finale = stock_corrente
        
        valore_invenduto = invenduto_finale * costo_unitario
        
        costo_smaltimento_totale = invenduto_finale * costo_smaltimento
        
        costo_spreco_totale = (
            valore_invenduto
            + costo_smaltimento_totale
        )
        
        risultato_economico = (
            margine_vendite
            - costo_spreco_totale
        )
        
        sell_through = vendite_totali / stock_iniziale
        
        tasso_invenduto = invenduto_finale / stock_iniziale
        
        risultati.append({
            "simulazione": simulazione,
            "strategia_id": strategia_id,
            "prodotto": nome_prodotto,
            "scenario": nome_scenario,
            "percorso_sconti": percorso_sconti,
            "stock_iniziale": stock_iniziale,
            "vendite_totali": vendite_totali,
            "domanda_persa_totale": domanda_persa_totale,
            "sell_through_percentuale": sell_through * 100,
            "invenduto_finale": invenduto_finale,
            "tasso_invenduto_percentuale": tasso_invenduto * 100,
            "ricavi_totali": ricavi_totali,
            "margine_vendite": margine_vendite,
            "valore_invenduto": valore_invenduto,
            "costo_smaltimento_totale": costo_smaltimento_totale,
            "costo_spreco_totale": costo_spreco_totale,
            "risultato_economico": risultato_economico
        })
    
    df_montecarlo = pd.DataFrame(risultati)
    
    df_montecarlo = df_montecarlo.round(2)
    
    return df_montecarlo

# test Monte Carlo su una strategia del Sushi Roll
df_montecarlo_test = montecarlo_singola_strategia(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario",
    percorso_sconti=[0.00, 0.20, 0.40],
    numero_simulazioni=NUM_SIMULAZIONI_TEST,
    seed=123,
    strategia_id="TEST"
)
assert len(df_montecarlo_test) == NUM_SIMULAZIONI_TEST, (
    "Errore: il numero di simulazioni Monte Carlo non è corretto."
)
assert (df_montecarlo_test["invenduto_finale"] >= 0).all(), (
    "Errore: l'invenduto finale non può essere negativo."
)
assert df_montecarlo_test["sell_through_percentuale"].between(0, 100).all(), (
    "Errore: il sell-through deve essere compreso tra 0% e 100%."
)
assert df_montecarlo_test["risultato_economico"].notna().all(), (
    "Errore: esiste una simulazione senza risultato economico."
)


# funzione Monte Carlo per tutte le strategie di un prodotto
def montecarlo_tutte_le_strategie(
    nome_prodotto,
    nome_scenario,
    df_strategie,
    numero_simulazioni=100,
    seed=123
):
    
    risultati = []
    
    for strategia in df_strategie.itertuples():
        
        df_montecarlo_strategia = montecarlo_singola_strategia(
            nome_prodotto=nome_prodotto,
            nome_scenario=nome_scenario,
            percorso_sconti=strategia.percorso_sconti,
            numero_simulazioni=numero_simulazioni,
            seed=seed,
            strategia_id=strategia.strategia_id
        )
        
        risultati.append(df_montecarlo_strategia)
    
    df_montecarlo = pd.concat(
        risultati,
        ignore_index=True
    )
    
    return df_montecarlo

# test Monte Carlo su tutte le strategie del Sushi Roll
df_montecarlo_sushi_test = montecarlo_tutte_le_strategie(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario",
    df_strategie=df_strategie_sushi,
    numero_simulazioni=NUM_SIMULAZIONI_TEST,
    seed=123
)
assert len(df_montecarlo_sushi_test) == len(df_strategie_sushi) * NUM_SIMULAZIONI_TEST, (
    "Errore: il numero di simulazioni Monte Carlo per il sushi non è corretto."
)
assert df_montecarlo_sushi_test["strategia_id"].nunique() == len(df_strategie_sushi), (
    "Errore: non tutte le strategie del sushi sono state simulate con Monte Carlo."
)
assert (df_montecarlo_sushi_test["invenduto_finale"] >= 0).all(), (
    "Errore: l'invenduto finale non può essere negativo."
)
assert df_montecarlo_sushi_test["sell_through_percentuale"].between(0, 100).all(), (
    "Errore: il sell-through deve essere compreso tra 0% e 100%."
)
assert df_montecarlo_sushi_test["risultato_economico"].notna().all(), (
    "Errore: esiste una simulazione Monte Carlo senza risultato economico."
)


# funzione per calcolare gli indicatori di rischio Monte Carlo
def calcola_indicatori_rischio_montecarlo(df_montecarlo):
    
    df_lavoro = df_montecarlo.copy()
    
    df_lavoro["evento_spreco"] = (
        df_lavoro["invenduto_finale"] > 0
    )
    
    df_indicatori = (
        df_lavoro
        .groupby("strategia_id")
        .agg(
            prodotto=("prodotto", "first"),
            scenario=("scenario", "first"),
            percorso_sconti=("percorso_sconti", "first"),
            numero_simulazioni=("simulazione", "count"),
            risultato_economico_medio=("risultato_economico", "mean"),
            risultato_economico_minimo=("risultato_economico", "min"),
            risultato_economico_massimo=("risultato_economico", "max"),
            vendite_medie=("vendite_totali", "mean"),
            sell_through_medio_percentuale=("sell_through_percentuale", "mean"),
            invenduto_medio=("invenduto_finale", "mean"),
            costo_spreco_medio=("costo_spreco_totale", "mean"),
            probabilita_spreco_percentuale=("evento_spreco", "mean")
        )
        .reset_index()
    )
    
    df_indicatori["probabilita_spreco_percentuale"] = (
        df_indicatori["probabilita_spreco_percentuale"] * 100
    )
    
    df_indicatori["sconto_medio_percentuale"] = (
        df_indicatori["percorso_sconti"]
        .apply(lambda percorso: sum(percorso) / len(percorso) * 100)
    )
    
    df_indicatori = df_indicatori.sort_values(
        by=[
            "risultato_economico_medio",
            "probabilita_spreco_percentuale",
            "sconto_medio_percentuale"
        ],
        ascending=[
            False,
            True,
            True
        ]
    ).reset_index(drop=True)
    
    df_indicatori = df_indicatori[
        [
            "strategia_id",
            "prodotto",
            "scenario",
            "percorso_sconti",
            "sconto_medio_percentuale",
            "numero_simulazioni",
            "risultato_economico_medio",
            "risultato_economico_minimo",
            "risultato_economico_massimo",
            "vendite_medie",
            "sell_through_medio_percentuale",
            "invenduto_medio",
            "costo_spreco_medio",
            "probabilita_spreco_percentuale"
        ]
    ]
    
    df_indicatori = df_indicatori.round(2)
    
    return df_indicatori

# indicatori di rischio Monte Carlo per le strategie del Sushi Roll
df_indicatori_rischio_sushi_test = calcola_indicatori_rischio_montecarlo(
    df_montecarlo_sushi_test
)
assert len(df_indicatori_rischio_sushi_test) == len(df_strategie_sushi), (
    "Errore: dovrebbe esserci una riga per ogni strategia del sushi."
)
assert (
    df_indicatori_rischio_sushi_test["numero_simulazioni"] == NUM_SIMULAZIONI_TEST
).all(), (
    "Errore: il numero di simulazioni per strategia non è corretto."
) 
assert df_indicatori_rischio_sushi_test["probabilita_spreco_percentuale"].between(0, 100).all(), (
    "Errore: la probabilità di spreco deve essere compresa tra 0% e 100%."
)
assert (df_indicatori_rischio_sushi_test["invenduto_medio"] >= 0).all(), (
    "Errore: l'invenduto medio non può essere negativo."
)

assert df_indicatori_rischio_sushi_test["risultato_economico_medio"].notna().all(), (
    "Errore: esiste una strategia senza risultato economico medio."
)


# CONFRONTI FINALI E ANALISI DI SENSIBILITÀ

# funzione per selezionare la strategia migliore considerando l'incertezza 
def seleziona_strategia_migliore_montecarlo(df_indicatori):
    
    df_ordinate = df_indicatori.copy()
    
    df_ordinate = df_ordinate.sort_values(
        by=[
            "risultato_economico_medio",
            "probabilita_spreco_percentuale",
            "sconto_medio_percentuale"
        ],
        ascending=[
            False,
            True,
            True
        ]
    ).reset_index(drop=True)
    
    df_migliore = df_ordinate.head(1).copy()
    
    df_migliore["raccomandazione_modello"] = (
        "Strategia raccomandata considerando l'incertezza"
    )
    
    df_migliore["criterio_scelta"] = (
        "Massimo risultato economico medio Monte Carlo; "
        "in caso di parità, minore probabilità di spreco "
        "e minore sconto medio"
    )
    
    return df_migliore

# funzione generale di raccomandazione markdown con Monte Carlo
def raccomanda_markdown_montecarlo(
    nome_prodotto,
    nome_scenario,
    numero_simulazioni=100,
    seed=123
):
    
    assert nome_prodotto in df_prodotti["prodotto"].tolist(), (
        "Errore: il prodotto indicato non è presente nei dati."
    )
    
    assert nome_scenario in df_scenari["scenario"].tolist(), (
        "Errore: lo scenario indicato non è presente nei dati."
    )
    
    df_strategie = genera_strategie_markdown(nome_prodotto)
    
    df_montecarlo = montecarlo_tutte_le_strategie(
        nome_prodotto=nome_prodotto,
        nome_scenario=nome_scenario,
        df_strategie=df_strategie,
        numero_simulazioni=numero_simulazioni,
        seed=seed
    )
    
    df_indicatori = calcola_indicatori_rischio_montecarlo(
        df_montecarlo
    )
    
    df_raccomandazione = seleziona_strategia_migliore_montecarlo(
        df_indicatori
    )
    
    df_raccomandazione["numero_strategie_valutate"] = len(df_strategie)
    
    df_raccomandazione = df_raccomandazione[
        [
            "raccomandazione_modello",
            "criterio_scelta",
            "prodotto",
            "scenario",
            "strategia_id",
            "percorso_sconti",
            "numero_strategie_valutate",
            "numero_simulazioni",
            "sconto_medio_percentuale",
            "risultato_economico_medio",
            "risultato_economico_minimo",
            "risultato_economico_massimo",
            "vendite_medie",
            "sell_through_medio_percentuale",
            "invenduto_medio",
            "costo_spreco_medio",
            "probabilita_spreco_percentuale"
        ]
    ]
    
    return df_raccomandazione

# test della raccomandazione Monte Carlo sul Sushi Roll
df_raccomandazione_montecarlo_test = raccomanda_markdown_montecarlo(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario",
    numero_simulazioni=NUM_SIMULAZIONI_TEST,
    seed=123
)
assert len(df_raccomandazione_montecarlo_test) == 1, (
    "Errore: la funzione Monte Carlo dovrebbe restituire una sola raccomandazione."
)
assert df_raccomandazione_montecarlo_test["prodotto"].iloc[0] == "Sushi Roll", (
    "Errore: il prodotto della raccomandazione Monte Carlo non coincide con quello richiesto."
)
assert df_raccomandazione_montecarlo_test["scenario"].iloc[0] == "Ordinario", (
    "Errore: lo scenario della raccomandazione Monte Carlo non coincide con quello richiesto."
)
assert df_raccomandazione_montecarlo_test["numero_simulazioni"].iloc[0] == NUM_SIMULAZIONI_TEST, (
    "Errore: il numero di simulazioni Monte Carlo non è corretto."
)
assert (
    0
    <= df_raccomandazione_montecarlo_test["probabilita_spreco_percentuale"].iloc[0]
    <= 100
), (
    "Errore: la probabilità di spreco deve essere compresa tra 0% e 100%."
)

    
# funzione per applicare la raccomandazione Monte Carlo a una lista di prodotti e scenari
def raccomanda_markdown_montecarlo_lista(
    lista_prodotti,
    lista_scenari,
    numero_simulazioni=100,
    seed=123
):
    
    risultati = []
    
    for prodotto in lista_prodotti:
        
        for scenario in lista_scenari:
            
            df_raccomandazione = raccomanda_markdown_montecarlo(
                nome_prodotto=prodotto,
                nome_scenario=scenario,
                numero_simulazioni=numero_simulazioni,
                seed=seed
            )
            
            risultati.append(df_raccomandazione)
    
    df_raccomandazioni = pd.concat(
        risultati,
        ignore_index=True
    )
    
    return df_raccomandazioni    
    
# applicazione Monte Carlo ai casi studio
df_raccomandazioni_montecarlo_casi_studio = raccomanda_markdown_montecarlo_lista(
    lista_prodotti=prodotti_casi_studio,
    lista_scenari=scenari_casi_studio,
    numero_simulazioni=NUM_SIMULAZIONI_MONTECARLO,
    seed=123
)

# controlli sulla tabella Monte Carlo dei casi studio
assert len(df_raccomandazioni_montecarlo_casi_studio) == 9, (
    "Errore: dovrebbero esserci 9 raccomandazioni Monte Carlo, "
    "una per ogni combinazione prodotto-scenario."
)
assert (
    df_raccomandazioni_montecarlo_casi_studio
    .groupby(["prodotto", "scenario"])
    .size()
    .eq(1)
    .all()
), (
    "Errore: dovrebbe esserci una sola raccomandazione Monte Carlo "
    "per ogni prodotto e scenario."
)
assert df_raccomandazioni_montecarlo_casi_studio["risultato_economico_medio"].notna().all(), (
    "Errore: esiste una raccomandazione Monte Carlo senza risultato economico medio."
)
assert df_raccomandazioni_montecarlo_casi_studio["probabilita_spreco_percentuale"].between(0, 100).all(), (
    "Errore: la probabilità di spreco deve essere compresa tra 0% e 100%."
)
assert (df_raccomandazioni_montecarlo_casi_studio["invenduto_medio"] >= 0).all(), (
    "Errore: l'invenduto medio non può essere negativo."
)
 
# confronto tra strategia manuale, deterministica e stocastica
# funzione per definire una strategia manuale di riferimento
def definisci_strategia_manuale_riferimento(nome_prodotto):
    
    dati_prodotto = df_prodotti.loc[
        df_prodotti["prodotto"] == nome_prodotto
    ]
    
    assert len(dati_prodotto) == 1, (
        "Errore: il prodotto indicato non è presente nei dati."
    )
    
    numero_periodi = int(dati_prodotto["numero_periodi"].iloc[0])
    
    if numero_periodi == 3:
        nome_strategia = "Manuale progressiva intermedia"
        percorso_sconti = [0.00, 0.20, 0.40]
    
    elif numero_periodi == 5:
        nome_strategia = "Manuale progressiva"
        percorso_sconti = [0.00, 0.10, 0.20, 0.30, 0.40]
    
    else:
        raise ValueError(
            "Errore: per questo numero di periodi non è stata definita "
            "una strategia manuale di riferimento."
        )
    
    return nome_strategia, percorso_sconti

# funzione per estrarre una raccomandazione già calcolata
def estrai_raccomandazione(df_raccomandazioni, nome_prodotto, nome_scenario):
    
    riga = df_raccomandazioni.loc[
        (df_raccomandazioni["prodotto"] == nome_prodotto)
        & (df_raccomandazioni["scenario"] == nome_scenario)
    ]
    
    assert len(riga) == 1, (
        "Errore: non è stata trovata una sola raccomandazione "
        "per la combinazione prodotto-scenario."
    )
    
    return riga.iloc[0]

# funzione per confrontare i tre approcci su prodotto e scenario
def confronta_approcci_markdown(
    nome_prodotto,
    nome_scenario,
    numero_simulazioni=100,
    seed=123
):
    
    nome_manuale, percorso_manuale = definisci_strategia_manuale_riferimento(
        nome_prodotto
    )
    
    raccomandazione_deterministica = estrai_raccomandazione(
        df_raccomandazioni_casi_studio,
        nome_prodotto,
        nome_scenario
    )
    
    raccomandazione_stocastica = estrai_raccomandazione(
        df_raccomandazioni_montecarlo_casi_studio,
        nome_prodotto,
        nome_scenario
    )
    
    strategie_da_confrontare = [
        {
            "approccio": "Strategia manuale",
            "descrizione_approccio": nome_manuale,
            "strategia_id_originale": "MANUALE",
            "percorso_sconti": percorso_manuale
        },
        {
            "approccio": "Raccomandazione deterministica",
            "descrizione_approccio": "Strategia scelta massimizzando il risultato economico atteso",
            "strategia_id_originale": raccomandazione_deterministica["strategia_id"],
            "percorso_sconti": raccomandazione_deterministica["percorso_sconti"]
        },
        {
            "approccio": "Raccomandazione stocastica",
            "descrizione_approccio": "Strategia scelta considerando risultato medio e rischio di spreco",
            "strategia_id_originale": raccomandazione_stocastica["strategia_id"],
            "percorso_sconti": raccomandazione_stocastica["percorso_sconti"]
        }
    ]
    
    risultati = []
    
    for strategia in strategie_da_confrontare:
        
        df_montecarlo = montecarlo_singola_strategia(
            nome_prodotto=nome_prodotto,
            nome_scenario=nome_scenario,
            percorso_sconti=strategia["percorso_sconti"],
            numero_simulazioni=numero_simulazioni,
            seed=seed,
            strategia_id=strategia["approccio"]
        )
        
        df_indicatori = calcola_indicatori_rischio_montecarlo(
            df_montecarlo
        )
        
        df_indicatori["approccio"] = strategia["approccio"]
        df_indicatori["descrizione_approccio"] = strategia["descrizione_approccio"]
        df_indicatori["strategia_id_originale"] = strategia["strategia_id_originale"]
        
        risultati.append(df_indicatori)
    
    df_confronto = pd.concat(
        risultati,
        ignore_index=True
    )
    
    df_confronto = df_confronto[
        [
            "prodotto",
            "scenario",
            "approccio",
            "descrizione_approccio",
            "strategia_id_originale",
            "percorso_sconti",
            "sconto_medio_percentuale",
            "numero_simulazioni",
            "risultato_economico_medio",
            "risultato_economico_minimo",
            "risultato_economico_massimo",
            "vendite_medie",
            "sell_through_medio_percentuale",
            "invenduto_medio",
            "costo_spreco_medio",
            "probabilita_spreco_percentuale"
        ]
    ]
    
    return df_confronto

# test del confronto tra approcci sul Sushi Roll in scenario Ordinario
df_confronto_approcci_test = confronta_approcci_markdown(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario",
    numero_simulazioni=NUM_SIMULAZIONI_TEST,
    seed=123
)
assert len(df_confronto_approcci_test) == 3, (
    "Errore: il confronto dovrebbe contenere tre approcci."
)
assert df_confronto_approcci_test["approccio"].nunique() == 3, (
    "Errore: dovrebbero essere presenti tre approcci distinti."
)
assert df_confronto_approcci_test["probabilita_spreco_percentuale"].between(0, 100).all(), (
    "Errore: la probabilità di spreco deve essere compresa tra 0% e 100%."
)

# funzione per confrontare gli approcci su tutti i casi studio
def confronta_approcci_casi_studio(
    lista_prodotti,
    lista_scenari,
    numero_simulazioni=100,
    seed=123
):
    
    risultati = []
    
    for prodotto in lista_prodotti:
        
        for scenario in lista_scenari:
            
            df_confronto = confronta_approcci_markdown(
                nome_prodotto=prodotto,
                nome_scenario=scenario,
                numero_simulazioni=numero_simulazioni,
                seed=seed
            )
            
            risultati.append(df_confronto)
    
    df_confronto_finale = pd.concat(
        risultati,
        ignore_index=True
    )
    
    return df_confronto_finale

# confronto finale tra approcci per tutti i prodotti e scenari
df_confronto_approcci_casi_studio = confronta_approcci_casi_studio(
    lista_prodotti=prodotti_casi_studio,
    lista_scenari=scenari_casi_studio,
    numero_simulazioni=NUM_SIMULAZIONI_MONTECARLO,
    seed=123
)

assert len(df_confronto_approcci_casi_studio) == 27, (
    "Errore: il confronto dovrebbe contenere 27 righe "
    "3 prodotti x 3 scenari x 3 approcci."
)

assert (
    df_confronto_approcci_casi_studio
    .groupby(["prodotto", "scenario"])["approccio"]
    .nunique()
    .eq(3)
    .all()
), (
    "Errore: per ogni prodotto e scenario dovrebbero esserci tre approcci."
)

assert df_confronto_approcci_casi_studio["risultato_economico_medio"].notna().all(), (
    "Errore: esiste un approccio senza risultato economico medio."
)

assert df_confronto_approcci_casi_studio["probabilita_spreco_percentuale"].between(0, 100).all(), (
    "Errore: la probabilità di spreco deve essere compresa tra 0% e 100%."
)


# ANALISI DI SENSIBILITÀ SU STOCK ED ERRORE PREVISIONALE
# funzione per modificare l'ampiezza dell'incertezza della domanda
def calcola_limiti_incertezza_modificati(
    incertezza_min_base,
    incertezza_moda_base,
    incertezza_max_base,
    fattore_errore_previsionale
):
    
    nuovo_min = (
        incertezza_moda_base
        - (incertezza_moda_base - incertezza_min_base)
        * fattore_errore_previsionale
    )
    
    nuovo_max = (
        incertezza_moda_base
        + (incertezza_max_base - incertezza_moda_base)
        * fattore_errore_previsionale
    )
    
    nuovo_min = max(nuovo_min, 0.01)
    
    return nuovo_min, incertezza_moda_base, nuovo_max

# funzione per calcolare una raccomandazione Monte Carlo con stock e incertezza modificati
def raccomanda_markdown_sensibilita(
    nome_prodotto,
    nome_scenario,
    fattore_stock,
    fattore_errore_previsionale,
    numero_simulazioni=50,
    seed=123
):
    
    indice_prodotto = df_prodotti.index[
        df_prodotti["prodotto"] == nome_prodotto
    ]
    
    assert len(indice_prodotto) == 1, (
        "Errore: il prodotto indicato non è presente nei dati."
    )
    
    indice_prodotto = indice_prodotto[0]
    
    stock_base = df_prodotti.loc[indice_prodotto, "stock_iniziale"]
    incertezza_min_base = df_prodotti.loc[indice_prodotto, "incertezza_min"]
    incertezza_moda_base = df_prodotti.loc[indice_prodotto, "incertezza_moda"]
    incertezza_max_base = df_prodotti.loc[indice_prodotto, "incertezza_max"]
    
    stock_sensibilita = max(
        int(round(stock_base * fattore_stock)),
        1
    )
    
    (
        incertezza_min_sensibilita,
        incertezza_moda_sensibilita,
        incertezza_max_sensibilita
    ) = calcola_limiti_incertezza_modificati(
        incertezza_min_base=incertezza_min_base,
        incertezza_moda_base=incertezza_moda_base,
        incertezza_max_base=incertezza_max_base,
        fattore_errore_previsionale=fattore_errore_previsionale
    )
    
    try:
        df_prodotti.loc[indice_prodotto, "stock_iniziale"] = stock_sensibilita
        df_prodotti.loc[indice_prodotto, "incertezza_min"] = incertezza_min_sensibilita
        df_prodotti.loc[indice_prodotto, "incertezza_moda"] = incertezza_moda_sensibilita
        df_prodotti.loc[indice_prodotto, "incertezza_max"] = incertezza_max_sensibilita
        
        df_raccomandazione = raccomanda_markdown_montecarlo(
            nome_prodotto=nome_prodotto,
            nome_scenario=nome_scenario,
            numero_simulazioni=numero_simulazioni,
            seed=seed
        )
        
        df_raccomandazione["stock_base"] = stock_base
        df_raccomandazione["stock_sensibilita"] = stock_sensibilita
        df_raccomandazione["fattore_stock"] = fattore_stock
        
        df_raccomandazione["incertezza_min_base"] = incertezza_min_base
        df_raccomandazione["incertezza_moda_base"] = incertezza_moda_base
        df_raccomandazione["incertezza_max_base"] = incertezza_max_base
        
        df_raccomandazione["incertezza_min_sensibilita"] = incertezza_min_sensibilita
        df_raccomandazione["incertezza_moda_sensibilita"] = incertezza_moda_sensibilita
        df_raccomandazione["incertezza_max_sensibilita"] = incertezza_max_sensibilita
        df_raccomandazione["fattore_errore_previsionale"] = fattore_errore_previsionale
        
    finally:
        df_prodotti.loc[indice_prodotto, "stock_iniziale"] = stock_base
        df_prodotti.loc[indice_prodotto, "incertezza_min"] = incertezza_min_base
        df_prodotti.loc[indice_prodotto, "incertezza_moda"] = incertezza_moda_base
        df_prodotti.loc[indice_prodotto, "incertezza_max"] = incertezza_max_base
    
    return df_raccomandazione

# funzione per svolgere l'analisi di sensibilità su stock ed errore previsionale
def analisi_sensibilita_stock_errore(
    nome_prodotto,
    nome_scenario,
    fattori_stock=None,
    fattori_errore_previsionale=None,
    numero_simulazioni=50,
    seed=123
):
    
    if fattori_stock is None:
        fattori_stock = [0.80, 1.00, 1.20]
    
    if fattori_errore_previsionale is None:
        fattori_errore_previsionale = [0.75, 1.00, 1.25]
    
    risultati = []
    
    for fattore_stock in fattori_stock:
        
        for fattore_errore in fattori_errore_previsionale:
            
            df_raccomandazione = raccomanda_markdown_sensibilita(
                nome_prodotto=nome_prodotto,
                nome_scenario=nome_scenario,
                fattore_stock=fattore_stock,
                fattore_errore_previsionale=fattore_errore,
                numero_simulazioni=numero_simulazioni,
                seed=seed
            )
            
            risultati.append(df_raccomandazione)
    
    df_sensibilita = pd.concat(
        risultati,
        ignore_index=True
    )
    
    df_sensibilita = df_sensibilita[
        [
            "prodotto",
            "scenario",
            "strategia_id",
            "percorso_sconti",
            "numero_simulazioni",
            "fattore_stock",
            "stock_base",
            "stock_sensibilita",
            "fattore_errore_previsionale",
            "incertezza_min_sensibilita",
            "incertezza_moda_sensibilita",
            "incertezza_max_sensibilita",
            "sconto_medio_percentuale",
            "risultato_economico_medio",
            "vendite_medie",
            "sell_through_medio_percentuale",
            "invenduto_medio",
            "costo_spreco_medio",
            "probabilita_spreco_percentuale"
        ]
    ]
    
    return df_sensibilita

# test dell'analisi di sensibilità sul Sushi Roll in scenario Ordinario
df_sensibilita_sushi_test = analisi_sensibilita_stock_errore(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario",
    numero_simulazioni=NUM_SIMULAZIONI_SENSIBILITA,
    seed=123
)
assert len(df_sensibilita_sushi_test) == 9, (
    "Errore: l'analisi di sensibilità dovrebbe produrre 9 combinazioni "
    "3 livelli di stock x 3 livelli di errore previsionale."
)
assert df_sensibilita_sushi_test["probabilita_spreco_percentuale"].between(0, 100).all(), (
    "Errore: la probabilità di spreco deve essere compresa tra 0% e 100%."
)
assert (df_sensibilita_sushi_test["stock_sensibilita"] > 0).all(), (
    "Errore: lo stock modificato deve essere positivo."
)
assert df_sensibilita_sushi_test["risultato_economico_medio"].notna().all(), (
    "Errore: esiste una riga senza risultato economico medio."
)

# analisi di sensibilità estesa a tutti i casi studio
def analisi_sensibilita_casi_studio(
    lista_prodotti,
    lista_scenari,
    fattori_stock=None,
    fattori_errore_previsionale=None,
    numero_simulazioni=50,
    seed=123
):
    
    risultati = []
    
    for prodotto in lista_prodotti:
        
        for scenario in lista_scenari:
            
            df_sensibilita = analisi_sensibilita_stock_errore(
                nome_prodotto=prodotto,
                nome_scenario=scenario,
                fattori_stock=fattori_stock,
                fattori_errore_previsionale=fattori_errore_previsionale,
                numero_simulazioni=numero_simulazioni,
                seed=seed
            )
            
            risultati.append(df_sensibilita)
    
    df_sensibilita_finale = pd.concat(
        risultati,
        ignore_index=True
    )
    
    return df_sensibilita_finale

# analisi di sensibilità per tutti i prodotti e scenari dei casi studio
# analisi di sensibilità estesa in forma di stress test
# si considerano combinazioni estreme di stock e incertezza per contenere i tempi di esecuzione
df_sensibilita_casi_studio = analisi_sensibilita_casi_studio(
    lista_prodotti=prodotti_casi_studio,
    lista_scenari=scenari_casi_studio,
    fattori_stock=[0.80, 1.20],
    fattori_errore_previsionale=[0.75, 1.25],
    numero_simulazioni=NUM_SIMULAZIONI_SENSIBILITA,
    seed=123
)


# 4. OUTPUT FINALI, TABELLE DI LETTURA E CRUSCOTTO

# confronto tra raccomandazione deterministica e raccomandazione Monte Carlo
df_deterministico = df_raccomandazioni_casi_studio[
    [
        "prodotto",
        "scenario",
        "strategia_id",
        "percorso_sconti",
        "sconto_medio_percentuale",
        "risultato_economico",
        "vendite_totali",
        "sell_through_percentuale",
        "invenduto_finale",
        "tasso_invenduto_percentuale"
    ]
].copy()

df_deterministico = df_deterministico.rename(columns={
    "strategia_id": "strategia_id_deterministica",
    "percorso_sconti": "percorso_sconti_deterministico",
    "sconto_medio_percentuale": "sconto_medio_deterministico",
    "risultato_economico": "risultato_economico_deterministico",
    "vendite_totali": "vendite_deterministiche",
    "sell_through_percentuale": "sell_through_deterministico",
    "invenduto_finale": "invenduto_deterministico",
    "tasso_invenduto_percentuale": "tasso_invenduto_deterministico"
})


df_montecarlo = df_raccomandazioni_montecarlo_casi_studio[
    [
        "prodotto",
        "scenario",
        "strategia_id",
        "percorso_sconti",
        "sconto_medio_percentuale",
        "risultato_economico_medio",
        "risultato_economico_minimo",
        "risultato_economico_massimo",
        "vendite_medie",
        "sell_through_medio_percentuale",
        "invenduto_medio",
        "probabilita_spreco_percentuale"
    ]
].copy()

df_montecarlo = df_montecarlo.rename(columns={
    "strategia_id": "strategia_id_montecarlo",
    "percorso_sconti": "percorso_sconti_montecarlo",
    "sconto_medio_percentuale": "sconto_medio_montecarlo",
    "risultato_economico_medio": "risultato_economico_medio_montecarlo",
    "risultato_economico_minimo": "risultato_economico_minimo_montecarlo",
    "risultato_economico_massimo": "risultato_economico_massimo_montecarlo",
    "vendite_medie": "vendite_medie_montecarlo",
    "sell_through_medio_percentuale": "sell_through_medio_montecarlo",
    "invenduto_medio": "invenduto_medio_montecarlo",
    "probabilita_spreco_percentuale": "probabilita_spreco_montecarlo"
})


df_confronto_deterministico_montecarlo = df_deterministico.merge(
    df_montecarlo,
    on=["prodotto", "scenario"],
    how="inner"
)

df_confronto_deterministico_montecarlo["stessa_strategia"] = (
    df_confronto_deterministico_montecarlo.apply(
        lambda riga: (
            riga["percorso_sconti_deterministico"]
            == riga["percorso_sconti_montecarlo"]
        ),
        axis=1
    )
)

df_confronto_deterministico_montecarlo["differenza_sconto_medio"] = (
    df_confronto_deterministico_montecarlo["sconto_medio_montecarlo"]
    - df_confronto_deterministico_montecarlo["sconto_medio_deterministico"]
)

df_confronto_deterministico_montecarlo = (
    df_confronto_deterministico_montecarlo.round(2)
)

# tabella sintetica del confronto tra approcci
df_sintesi_approcci = df_confronto_approcci_casi_studio[
    [
        "prodotto",
        "scenario",
        "approccio",
        "percorso_sconti",
        "sconto_medio_percentuale",
        "risultato_economico_medio",
        "invenduto_medio",
        "probabilita_spreco_percentuale"
    ]
].copy()

df_sintesi_approcci = df_sintesi_approcci.sort_values(
    by=[
        "prodotto",
        "scenario",
        "risultato_economico_medio"
    ],
    ascending=[
        True,
        True,
        False
    ]
).reset_index(drop=True)

df_sintesi_approcci = df_sintesi_approcci.round(2)

# tabella sintetica dell'analisi di sensibilità estesa
df_sintesi_sensibilita_casi_studio = df_sensibilita_casi_studio[
    [
        "prodotto",
        "scenario",
        "fattore_stock",
        "stock_sensibilita",
        "fattore_errore_previsionale",
        "percorso_sconti",
        "sconto_medio_percentuale",
        "risultato_economico_medio",
        "invenduto_medio",
        "probabilita_spreco_percentuale"
    ]
].copy()

df_sintesi_sensibilita_casi_studio = df_sintesi_sensibilita_casi_studio.sort_values(
    by=[
        "prodotto",
        "scenario",
        "fattore_stock",
        "fattore_errore_previsionale"
    ]
).reset_index(drop=True)

df_sintesi_sensibilita_casi_studio = df_sintesi_sensibilita_casi_studio.round(2)

# controlli 
assert len(df_confronto_deterministico_montecarlo) == 9, (
    "Errore: il confronto deterministico-Monte Carlo dovrebbe contenere 9 righe."
)
assert len(df_sintesi_approcci) == 27, (
    "Errore: la sintesi degli approcci dovrebbe contenere 27 righe."
)
assert len(df_sintesi_sensibilita_casi_studio) == 36, (
    "Errore: la sintesi della sensibilità estesa dovrebbe contenere 36 righe."
)
assert df_confronto_deterministico_montecarlo["probabilita_spreco_montecarlo"].between(0, 100).all(), (
    "Errore: la probabilità di spreco Monte Carlo deve essere compresa tra 0% e 100%."
)
assert df_sintesi_approcci["probabilita_spreco_percentuale"].between(0, 100).all(), (
    "Errore: la probabilità di spreco nel confronto approcci deve essere compresa tra 0% e 100%."
)
assert df_sintesi_sensibilita_casi_studio["probabilita_spreco_percentuale"].between(0, 100).all(), (
    "Errore: la probabilità di spreco nella sensibilità estesa deve essere compresa tra 0% e 100%."
)
assert df_confronto_deterministico_montecarlo["risultato_economico_medio_montecarlo"].notna().all(), (
    "Errore: esiste una riga senza risultato economico medio Monte Carlo."
)
assert df_sintesi_approcci["risultato_economico_medio"].notna().all(), (
    "Errore: esiste una riga senza risultato economico medio nel confronto approcci."
)
assert df_sintesi_sensibilita_casi_studio["risultato_economico_medio"].notna().all(), (
    "Errore: esiste una riga senza risultato economico medio nella sensibilità estesa."
)

# controllo dell'approccio con il risultato economico medio più alto
df_vincitore_approcci = (
    df_sintesi_approcci
    .sort_values(
        by=[
            "prodotto",
            "scenario",
            "risultato_economico_medio",
            "probabilita_spreco_percentuale"
        ],
        ascending=[
            True,
            True,
            False,
            True
        ]
    )
    .groupby(["prodotto", "scenario"])
    .head(1)
    .reset_index(drop=True)
)
assert len(df_vincitore_approcci) == 9, (
    "Errore: dovrebbe esserci un approccio vincente per ogni prodotto e scenario."
)

# tabelle finali pulite per l'analisi dei risultati
df_output_raccomandazioni = df_confronto_deterministico_montecarlo[
    [
        "prodotto",
        "scenario",
        "percorso_sconti_deterministico",
        "percorso_sconti_montecarlo",
        "stessa_strategia",
        "sconto_medio_deterministico",
        "sconto_medio_montecarlo",
        "differenza_sconto_medio",
        "risultato_economico_deterministico",
        "risultato_economico_medio_montecarlo",
        "invenduto_deterministico",
        "invenduto_medio_montecarlo",
        "probabilita_spreco_montecarlo"
    ]
].copy()

df_output_raccomandazioni = df_output_raccomandazioni.round(2)


df_output_approcci = df_sintesi_approcci[
    [
        "prodotto",
        "scenario",
        "approccio",
        "percorso_sconti",
        "sconto_medio_percentuale",
        "risultato_economico_medio",
        "invenduto_medio",
        "probabilita_spreco_percentuale"
    ]
].copy()

df_output_approcci = df_output_approcci.round(2)


df_output_sensibilita = df_sintesi_sensibilita_casi_studio[
    [
        "prodotto",
        "scenario",
        "fattore_stock",
        "stock_sensibilita",
        "fattore_errore_previsionale",
        "percorso_sconti",
        "sconto_medio_percentuale",
        "risultato_economico_medio",
        "invenduto_medio",
        "probabilita_spreco_percentuale"
    ]
].copy()

df_output_sensibilita = df_output_sensibilita.round(2)


# riepilogo finale del modello
df_riepilogo_modello = pd.DataFrame({
    "elemento": [
        "Numero prodotti casi studio",
        "Numero scenari",
        "Strategie Sushi Roll",
        "Strategie Giovanni Rana Sfogliavelo",
        "Strategie FAGE Total 0%",
        "Raccomandazioni deterministiche",
        "Raccomandazioni Monte Carlo",
        "Confronti approcci",
        "Analisi sensibilità"
    ],
    "valore": [
        len(df_prodotti),
        len(df_scenari),
        len(df_strategie_sushi),
        len(df_strategie_ravioli),
        len(df_strategie_fage),
        len(df_raccomandazioni_casi_studio),
        len(df_raccomandazioni_montecarlo_casi_studio),
        len(df_sintesi_approcci),
        len(df_sintesi_sensibilita_casi_studio)
    ]
})

# tabella leggibile della decisione finale del modello -> la raccomandazione finale è basata sulla simulazione Monte Carlo, perché considera anche l'incertezza della domanda.
df_cruscotto_raccomandazioni = df_output_raccomandazioni[
    [
        "prodotto",
        "scenario",
        "percorso_sconti_montecarlo",
        "sconto_medio_montecarlo",
        "risultato_economico_medio_montecarlo",
        "invenduto_medio_montecarlo",
        "probabilita_spreco_montecarlo",
        "percorso_sconti_deterministico",
        "risultato_economico_deterministico",
        "stessa_strategia"
    ]
].copy()

df_cruscotto_raccomandazioni = df_cruscotto_raccomandazioni.rename(
    columns={
        "percorso_sconti_montecarlo": "decisione_finale_modello",
        "sconto_medio_montecarlo": "sconto_medio_percentuale",
        "risultato_economico_medio_montecarlo": "risultato_economico_atteso",
        "invenduto_medio_montecarlo": "invenduto_medio_atteso",
        "probabilita_spreco_montecarlo": "probabilita_spreco_percentuale",
        "percorso_sconti_deterministico": "strategia_deterministica"
    }
)

df_cruscotto_raccomandazioni["criterio_decisione_finale"] = np.where(
    df_cruscotto_raccomandazioni["stessa_strategia"],
    "La simulazione Monte Carlo conferma la raccomandazione deterministica.",
    "La simulazione Monte Carlo viene preferita perché considera anche l'incertezza della domanda."
)

df_cruscotto_raccomandazioni = df_cruscotto_raccomandazioni[
    [
        "prodotto",
        "scenario",
        "decisione_finale_modello",
        "sconto_medio_percentuale",
        "risultato_economico_atteso",
        "invenduto_medio_atteso",
        "probabilita_spreco_percentuale",
        "strategia_deterministica",
        "risultato_economico_deterministico",
        "stessa_strategia",
        "criterio_decisione_finale"
    ]
].round(2)

# controlli sul cruscotto finale
assert len(df_cruscotto_raccomandazioni) == 9, (
    "Errore: il cruscotto dovrebbe contenere 9 righe, "
    "una per ogni combinazione prodotto-scenario."
)
assert (
    df_cruscotto_raccomandazioni
    .groupby(["prodotto", "scenario"])
    .size()
    .eq(1)
    .all()
), (
    "Errore: nel cruscotto dovrebbe esserci una sola raccomandazione "
    "per ogni prodotto e scenario."
)
assert df_cruscotto_raccomandazioni["probabilita_spreco_percentuale"].between(0, 100).all(), (
    "Errore: la probabilità di spreco deve essere compresa tra 0% e 100%."
)
assert df_cruscotto_raccomandazioni["risultato_economico_atteso"].notna().all(), (
    "Errore: esiste una raccomandazione senza risultato economico atteso."
)

# funzione per leggere la raccomandazione di uno specifico prodotto e scenario
def mostra_raccomandazione(nome_prodotto, nome_scenario):
    
    df_risultato = df_cruscotto_raccomandazioni[
        (df_cruscotto_raccomandazioni["prodotto"] == nome_prodotto) &
        (df_cruscotto_raccomandazioni["scenario"] == nome_scenario)
    ].copy()
    
    assert len(df_risultato) == 1, (
        "Errore: la selezione dovrebbe restituire una sola raccomandazione."
    )
    
    return df_risultato.reset_index(drop=True)

# esempi di lettura delle raccomandazioni finali
df_leggi_sushi_ordinario = mostra_raccomandazione(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario"
)

df_leggi_ravioli_ordinario = mostra_raccomandazione(
    nome_prodotto="Giovanni Rana Sfogliavelo",
    nome_scenario="Ordinario"
)

df_leggi_fage_ordinario = mostra_raccomandazione(
    nome_prodotto="FAGE Total 0%",
    nome_scenario="Ordinario"
)


# confronto per un singolo prodotto e scenario di strategia manuale, raccomandazione deterministica, raccomandazione stocastica / Monte Carlo
def mostra_confronto_approcci(nome_prodotto, nome_scenario):
    
    df_risultato = df_output_approcci[
        (df_output_approcci["prodotto"] == nome_prodotto) &
        (df_output_approcci["scenario"] == nome_scenario)
    ].copy()
    
    assert len(df_risultato) == 3, (
        "Errore: il confronto dovrebbe contenere tre approcci."
    )
    
    df_risultato = df_risultato.sort_values(
        by=[
            "risultato_economico_medio",
            "probabilita_spreco_percentuale"
        ],
        ascending=[
            False,
            True
        ]
    ).reset_index(drop=True)
    
    return df_risultato

# esempi di confronto tra approcci
df_confronto_approcci_sushi_ordinario = mostra_confronto_approcci(
    nome_prodotto="Sushi Roll",
    nome_scenario="Ordinario"
)

df_confronto_approcci_ravioli_ordinario = mostra_confronto_approcci(
    nome_prodotto="Giovanni Rana Sfogliavelo",
    nome_scenario="Ordinario"
)

df_confronto_approcci_fage_ordinario = mostra_confronto_approcci(
    nome_prodotto="FAGE Total 0%",
    nome_scenario="Ordinario"
)


# 5. CONTROLLI FINALI DI COERENZA DEL MODELLO

# elenco degli oggetti principali che devono esistere nel modello finale
oggetti_richiesti = [
    "df_prodotti",
    "df_periodi",
    "df_scenari",
    "df_sconti",
    "df_raccomandazioni_casi_studio",
    "df_raccomandazioni_montecarlo_casi_studio",
    "df_confronto_deterministico_montecarlo",
    "df_sintesi_approcci",
    "df_sintesi_sensibilita_casi_studio",
    "df_vincitore_approcci"
]

for oggetto in oggetti_richiesti:
    assert oggetto in globals(), (
        f"Errore: l'oggetto {oggetto} non è presente nel modello."
    )

# controlli dimensionali finali
assert len(df_prodotti) == 3, (
    "Errore: dovrebbero esserci 3 prodotti nei casi studio."
)
assert len(df_scenari) == 3, (
    "Errore: dovrebbero esserci 3 scenari di domanda."
)
assert len(df_raccomandazioni_casi_studio) == 9, (
    "Errore: le raccomandazioni deterministiche dovrebbero essere 9."
)
assert len(df_raccomandazioni_montecarlo_casi_studio) == 9, (
    "Errore: le raccomandazioni Monte Carlo dovrebbero essere 9."
)
assert len(df_confronto_deterministico_montecarlo) == 9, (
    "Errore: il confronto deterministico-Monte Carlo dovrebbe contenere 9 righe."
)

assert len(df_sintesi_approcci) == 27, (
    "Errore: il confronto tra approcci dovrebbe contenere 27 righe."
)
assert len(df_sintesi_sensibilita_casi_studio) == 36, (
    "Errore: l'analisi di sensibilità dovrebbe contenere 36 righe."
)
assert len(df_vincitore_approcci) == 9, (
    "Errore: dovrebbe esserci un approccio vincente per ciascun prodotto e scenario."
)

# controlli economici e operativi finali
assert (df_prodotti["prezzo_netto"] > 0).all(), (
    "Errore: tutti i prezzi netti devono essere positivi."
)
assert (df_prodotti["costo_unitario"] > 0).all(), (
    "Errore: tutti i costi unitari devono essere positivi."
)
assert (df_prodotti["stock_iniziale"] > 0).all(), (
    "Errore: tutti gli stock iniziali devono essere positivi."
)
assert df_raccomandazioni_casi_studio["sell_through_percentuale"].between(0, 100).all(), (
    "Errore: il sell-through deterministico deve essere compreso tra 0% e 100%."
)
assert df_raccomandazioni_montecarlo_casi_studio["probabilita_spreco_percentuale"].between(0, 100).all(), (
    "Errore: la probabilità di spreco Monte Carlo deve essere compresa tra 0% e 100%."
)
assert df_sintesi_approcci["probabilita_spreco_percentuale"].between(0, 100).all(), (
    "Errore: la probabilità di spreco nel confronto approcci deve essere compresa tra 0% e 100%."
)
assert df_sintesi_sensibilita_casi_studio["probabilita_spreco_percentuale"].between(0, 100).all(), (
    "Errore: la probabilità di spreco nella sensibilità deve essere compresa tra 0% e 100%."
)
assert df_confronto_deterministico_montecarlo["risultato_economico_deterministico"].notna().all(), (
    "Errore: esistono risultati deterministici mancanti."
)
assert df_confronto_deterministico_montecarlo["risultato_economico_medio_montecarlo"].notna().all(), (
    "Errore: esistono risultati Monte Carlo mancanti."
)


tempo_fine = time.perf_counter()
tempo_totale = tempo_fine - tempo_inizio

print(f"Tempo totale di esecuzione del modello: {tempo_totale:.2f} secondi")

