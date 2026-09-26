import math
import itertools
import numpy as np
import pandas as pd
import streamlit as st
import modello_markdown_optimization as modello


# CONFIGURAZIONE PAGINA

st.set_page_config(
    page_title="Markdown Optimization Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)


# FUNZIONI DI FORMATTAZIONE

def formatta_percentuale(valore):
    return f"{valore:.1f}%".replace(".", ",")


def formatta_punti_percentuali(valore):
    if valore >= 0:
        return f"+ {valore:.1f}".replace(".", ",") + " punti"
    return f"− {abs(valore):.1f}".replace(".", ",") + " punti"


def formatta_euro(valore):
    return f"€ {valore:.2f}".replace(".", ",")


def formatta_delta_euro(valore):
    if valore >= 0:
        return f"+ € {valore:.2f}".replace(".", ",")
    return f"− € {abs(valore):.2f}".replace(".", ",")


def formatta_numero(valore):
    return f"{valore:.2f}".replace(".", ",")


def formatta_delta_numero(valore):
    if valore >= 0:
        return f"+ {valore:.2f}".replace(".", ",")
    return f"− {abs(valore):.2f}".replace(".", ",")


def formatta_percorso_sconti(percorso):
    return " → ".join([f"{int(round(sconto * 100))}%" for sconto in percorso])


def formatta_lista_fattori(lista_valori):
    return " → ".join([formatta_numero(valore) for valore in lista_valori])


# FUNZIONI DI SUPPORTO INPUT

def arrotonda_per_eccesso_mezzo(valore):
    return math.ceil(valore * 2) / 2


def etichetta_approccio(approccio):
    testo = str(approccio).lower()

    if "manual" in testo:
        return "Strategia manuale"
    if "determin" in testo:
        return "Raccomandazione deterministica"
    if "stoc" in testo or "monte" in testo:
        return "Raccomandazione Monte Carlo"

    return str(approccio)


def punteggio_beta(valore):
    if valore == "Bassa":
        return 1
    if valore == "Media":
        return 2
    return 3


def punteggio_incertezza(valore):
    if valore == "Bassa":
        return 1
    if valore == "Media":
        return 2
    return 3


def adatta_profilo_freschezza(profilo_base, numero_periodi):
    if numero_periodi <= 1:
        return [profilo_base[0]]

    indice_massimo = len(profilo_base) - 1
    valori_adattati = []

    for i in range(numero_periodi):
        posizione = i * indice_massimo / (numero_periodi - 1)
        indice_basso = int(posizione)
        indice_alto = min(indice_basso + 1, indice_massimo)
        peso = posizione - indice_basso

        valore = (
            profilo_base[indice_basso] * (1 - peso)
            + profilo_base[indice_alto] * peso
        )

        valori_adattati.append(round(valore, 2))

    return valori_adattati


def interpreta_fattore_attrattivita(fattore):
    riduzione = (1 - fattore) * 100

    if riduzione <= 0:
        return "Nessuna penalizzazione della domanda per effetto dell'attrattività residua."

    return (
        f"Domanda attesa ridotta di circa {formatta_percentuale(riduzione)} "
        "per effetto della minore attrattività residua."
    )


def genera_pesi_domanda_base(numero_periodi, profilo_temporale):
    if profilo_temporale == "Distribuzione uniforme":
        pesi = [1.00] * numero_periodi

    elif profilo_temporale == "Più alta nei primi periodi":
        if numero_periodi == 1:
            pesi = [1.00]
        else:
            pesi = [
                1.30 - (0.60 * i / (numero_periodi - 1))
                for i in range(numero_periodi)
            ]

    elif profilo_temporale == "Più alta negli ultimi periodi":
        if numero_periodi == 1:
            pesi = [1.00]
        else:
            pesi = [
                0.70 + (0.60 * i / (numero_periodi - 1))
                for i in range(numero_periodi)
            ]

    else:
        pesi = [1.00] * numero_periodi

    somma_pesi = sum(pesi)

    return [
        peso / somma_pesi
        for peso in pesi
    ]


def distribuisci_domanda_base(domanda_totale, numero_periodi, profilo_temporale):
    pesi = genera_pesi_domanda_base(
        numero_periodi=numero_periodi,
        profilo_temporale=profilo_temporale
    )

    valori_grezzi = [
        domanda_totale * peso
        for peso in pesi
    ]

    mezze_unita_totali = int(round(domanda_totale * 2))

    mezze_unita_base = [
        math.floor(valore * 2)
        for valore in valori_grezzi
    ]

    mezze_unita_da_distribuire = (
        mezze_unita_totali
        - sum(mezze_unita_base)
    )

    resti = [
        ((valori_grezzi[i] * 2) - mezze_unita_base[i], i)
        for i in range(numero_periodi)
    ]

    resti_ordinati = sorted(resti, reverse=True)

    for posizione in range(mezze_unita_da_distribuire):
        indice_periodo = resti_ordinati[posizione][1]
        mezze_unita_base[indice_periodo] += 1

    domanda_per_periodo = [
        mezze_unita / 2
        for mezze_unita in mezze_unita_base
    ]

    return domanda_per_periodo


# MOTORE CUSTOM DI OTTIMIZZAZIONE

def genera_percorsi_markdown(numero_periodi):
    livelli_sconto = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]

    percorsi = list(
        itertools.combinations_with_replacement(
            livelli_sconto,
            numero_periodi
        )
    )

    return percorsi


def simula_percorso_custom(
    percorso_sconti,
    prezzo_netto,
    costo_unitario,
    costo_smaltimento,
    stock_iniziale,
    domanda_base_periodi,
    fattori_attrattivita,
    fattore_scenario,
    beta,
    moltiplicatori_incertezza=None
):
    stock_residuo = float(stock_iniziale)
    ricavi_netto = 0.0
    vendite_totali = 0.0
    dettagli_periodo = []

    if moltiplicatori_incertezza is None:
        moltiplicatori_incertezza = [1.00] * len(percorso_sconti)

    for indice_periodo, sconto in enumerate(percorso_sconti):
        domanda_base = domanda_base_periodi[indice_periodo]
        fattore_attrattivita = fattori_attrattivita[indice_periodo]
        moltiplicatore_incertezza = moltiplicatori_incertezza[indice_periodo]

        domanda_attesa = (
            domanda_base
            * fattore_scenario
            * fattore_attrattivita
            * (1 + beta * sconto)
            * moltiplicatore_incertezza
        )

        domanda_attesa = max(domanda_attesa, 0)

        vendite_periodo = min(stock_residuo, domanda_attesa)

        prezzo_netto_post_sconto = prezzo_netto * (1 - sconto)
        ricavi_periodo = vendite_periodo * prezzo_netto_post_sconto

        stock_residuo -= vendite_periodo
        ricavi_netto += ricavi_periodo
        vendite_totali += vendite_periodo

        dettagli_periodo.append({
            "periodo": indice_periodo + 1,
            "sconto": sconto,
            "domanda_base": domanda_base,
            "fattore_attrattivita": fattore_attrattivita,
            "moltiplicatore_incertezza": moltiplicatore_incertezza,
            "domanda_attesa": domanda_attesa,
            "vendite_periodo": vendite_periodo,
            "stock_residuo": stock_residuo,
            "prezzo_netto_post_sconto": prezzo_netto_post_sconto
        })

    costo_acquisto_totale = stock_iniziale * costo_unitario
    costo_smaltimento_totale = stock_residuo * costo_smaltimento

    risultato_economico = (
        ricavi_netto
        - costo_acquisto_totale
        - costo_smaltimento_totale
    )

    sell_through = vendite_totali / stock_iniziale * 100

    return {
        "risultato_economico": risultato_economico,
        "ricavi_netto": ricavi_netto,
        "costo_acquisto_totale": costo_acquisto_totale,
        "costo_smaltimento_totale": costo_smaltimento_totale,
        "vendite_totali": vendite_totali,
        "invenduto_finale": stock_residuo,
        "sell_through_percentuale": sell_through,
        "dettagli_periodo": dettagli_periodo
    }


def ottimizza_markdown_custom(
    prezzo_netto,
    prezzo_lordo,
    costo_unitario,
    costo_smaltimento,
    stock_iniziale,
    numero_periodi,
    domanda_base_periodi,
    fattori_attrattivita,
    fattore_scenario,
    beta,
    incertezza_min,
    incertezza_moda,
    incertezza_max,
    numero_simulazioni
):
    percorsi = genera_percorsi_markdown(numero_periodi)

    righe_deterministiche = []

    for percorso in percorsi:
        risultato = simula_percorso_custom(
            percorso_sconti=percorso,
            prezzo_netto=prezzo_netto,
            costo_unitario=costo_unitario,
            costo_smaltimento=costo_smaltimento,
            stock_iniziale=stock_iniziale,
            domanda_base_periodi=domanda_base_periodi,
            fattori_attrattivita=fattori_attrattivita,
            fattore_scenario=fattore_scenario,
            beta=beta,
            moltiplicatori_incertezza=[1.00] * numero_periodi
        )

        righe_deterministiche.append({
            "percorso_sconti": percorso,
            "risultato_economico": risultato["risultato_economico"],
            "ricavi_netto": risultato["ricavi_netto"],
            "costo_acquisto_totale": risultato["costo_acquisto_totale"],
            "costo_smaltimento_totale": risultato["costo_smaltimento_totale"],
            "vendite_totali": risultato["vendite_totali"],
            "invenduto_finale": risultato["invenduto_finale"],
            "sell_through_percentuale": risultato["sell_through_percentuale"]
        })

    df_deterministico = pd.DataFrame(righe_deterministiche)

    riga_deterministica = df_deterministico.sort_values(
        "risultato_economico",
        ascending=False
    ).iloc[0]

    rng = np.random.default_rng(123)

    scenari_incertezza = rng.triangular(
        left=incertezza_min,
        mode=incertezza_moda,
        right=incertezza_max,
        size=(numero_simulazioni, numero_periodi)
    )

    righe_montecarlo = []

    for percorso in percorsi:
        risultati = []
        ricavi = []
        costi_smaltimento = []
        vendite = []
        invenduti = []
        sell_through = []

        for indice_simulazione in range(numero_simulazioni):
            moltiplicatori = scenari_incertezza[indice_simulazione]

            risultato = simula_percorso_custom(
                percorso_sconti=percorso,
                prezzo_netto=prezzo_netto,
                costo_unitario=costo_unitario,
                costo_smaltimento=costo_smaltimento,
                stock_iniziale=stock_iniziale,
                domanda_base_periodi=domanda_base_periodi,
                fattori_attrattivita=fattori_attrattivita,
                fattore_scenario=fattore_scenario,
                beta=beta,
                moltiplicatori_incertezza=moltiplicatori
            )

            risultati.append(risultato["risultato_economico"])
            ricavi.append(risultato["ricavi_netto"])
            costi_smaltimento.append(risultato["costo_smaltimento_totale"])
            vendite.append(risultato["vendite_totali"])
            invenduti.append(risultato["invenduto_finale"])
            sell_through.append(risultato["sell_through_percentuale"])

        righe_montecarlo.append({
            "percorso_sconti": percorso,
            "risultato_economico_medio": float(np.mean(risultati)),
            "risultato_economico_std": float(np.std(risultati)),
            "risultato_economico_p5": float(np.percentile(risultati, 5)),
            "risultato_economico_p95": float(np.percentile(risultati, 95)),
            "ricavi_netto_medi": float(np.mean(ricavi)),
            "costo_acquisto_totale": float(stock_iniziale * costo_unitario),
            "costo_smaltimento_medio": float(np.mean(costi_smaltimento)),
            "vendite_totali_medie": float(np.mean(vendite)),
            "invenduto_medio": float(np.mean(invenduti)),
            "sell_through_medio_percentuale": float(np.mean(sell_through)),
            "probabilita_stock_residuo_percentuale": float((np.array(invenduti) > 0).mean() * 100)
        })

    df_montecarlo = pd.DataFrame(righe_montecarlo)

    riga_montecarlo = df_montecarlo.sort_values(
        "risultato_economico_medio",
        ascending=False
    ).iloc[0]

    percorso_finale = riga_montecarlo["percorso_sconti"]

    risultato_centrale_finale = simula_percorso_custom(
        percorso_sconti=percorso_finale,
        prezzo_netto=prezzo_netto,
        costo_unitario=costo_unitario,
        costo_smaltimento=costo_smaltimento,
        stock_iniziale=stock_iniziale,
        domanda_base_periodi=domanda_base_periodi,
        fattori_attrattivita=fattori_attrattivita,
        fattore_scenario=fattore_scenario,
        beta=beta,
        moltiplicatori_incertezza=[1.00] * numero_periodi
    )

    righe_dettaglio = []

    for dettaglio in risultato_centrale_finale["dettagli_periodo"]:
        periodo = dettaglio["periodo"]
        sconto = dettaglio["sconto"]

        righe_dettaglio.append({
            "Periodo": periodo,
            "Descrizione periodo": f"Periodo {periodo}",
            "Sconto consigliato": f"{int(round(sconto * 100))}%",
            "Sconto (%)": int(round(sconto * 100)),
            "Prezzo lordo post-sconto": prezzo_lordo * (1 - sconto),
            "Domanda base": dettaglio["domanda_base"],
            "Fattore attrattività": dettaglio["fattore_attrattivita"],
            "Domanda attesa centrale": dettaglio["domanda_attesa"],
            "Vendite attese centrali": dettaglio["vendite_periodo"],
            "Stock residuo stimato": dettaglio["stock_residuo"]
        })

    df_dettaglio_finale = pd.DataFrame(righe_dettaglio)

    return {
        "df_deterministico": df_deterministico,
        "df_montecarlo": df_montecarlo,
        "riga_deterministica": riga_deterministica,
        "riga_montecarlo": riga_montecarlo,
        "df_dettaglio_finale": df_dettaglio_finale
    }


# FUNZIONI PER ESEMPIO PRECOMPILATO

def crea_tabella_sconti_periodo(nome_prodotto, percorso_sconti):
    periodi_prodotto = (
        modello.df_periodi[
            modello.df_periodi["prodotto"] == nome_prodotto
        ]
        .sort_values("periodo")
        .copy()
    )

    prezzo_lordo = modello.df_prodotti.loc[
        modello.df_prodotti["prodotto"] == nome_prodotto,
        "prezzo_lordo"
    ].iloc[0]

    df_tabella = pd.DataFrame({
        "Periodo": periodi_prodotto["periodo"].astype(int).tolist(),
        "Descrizione periodo": periodi_prodotto["descrizione_periodo"].tolist(),
        "Sconto consigliato": [
            f"{int(round(sconto * 100))}%"
            for sconto in percorso_sconti
        ],
        "Sconto (%)": [
            int(round(sconto * 100))
            for sconto in percorso_sconti
        ],
        "Prezzo lordo post-sconto": [
            prezzo_lordo * (1 - sconto)
            for sconto in percorso_sconti
        ],
        "Fattore attrattività": periodi_prodotto["fattore_freschezza"].tolist(),
        "Domanda base": periodi_prodotto["domanda_base"].tolist()
    })

    return df_tabella


def crea_timeline_sconti(df_sconti_periodo):
    html = ['<div class="timeline-grid">']

    for _, riga in df_sconti_periodo.iterrows():
        html.append(
            '<div class="period-card">'
            f'<div class="period-label">Periodo {int(riga["Periodo"])}</div>'
            f'<div class="period-name">{riga["Descrizione periodo"]}</div>'
            f'<div class="period-discount">{riga["Sconto consigliato"]}</div>'
            f'<div class="period-detail">Prezzo cliente: {formatta_euro(riga["Prezzo lordo post-sconto"])} IVA inclusa</div>'
            '</div>'
        )

    html.append('</div>')

    return "".join(html)


@st.cache_data(show_spinner=False)
def calcola_benchmark_no_sconto(
    nome_prodotto,
    nome_scenario,
    numero_periodi,
    stock_iniziale,
    numero_simulazioni
):
    percorso_no_sconto = tuple([0.0] * int(numero_periodi))

    df_no_sconto = modello.montecarlo_singola_strategia(
        nome_prodotto=nome_prodotto,
        nome_scenario=nome_scenario,
        percorso_sconti=percorso_no_sconto,
        numero_simulazioni=numero_simulazioni,
        seed=123,
        strategia_id="no_markdown"
    )

    risultato_medio = df_no_sconto["risultato_economico"].mean()
    invenduto_medio = df_no_sconto["invenduto_finale"].mean()
    vendite_medie = df_no_sconto["vendite_totali"].mean()

    probabilita_stock_residuo = (
        (df_no_sconto["invenduto_finale"] > 0).mean() * 100
    )

    sell_through_medio = vendite_medie / stock_iniziale * 100

    return {
        "percorso_sconti": percorso_no_sconto,
        "risultato_economico_medio": risultato_medio,
        "invenduto_medio": invenduto_medio,
        "vendite_medie": vendite_medie,
        "sell_through_medio_percentuale": sell_through_medio,
        "probabilita_stock_residuo_percentuale": probabilita_stock_residuo
    }


# STILE GRAFICO

st.markdown(
    """
    <style>
    :root {
        --background: #FAF7F2;
        --surface: #FFFFFF;
        --text-main: #1F2937;
        --text-muted: #6B7280;
        --border: #E4D8CD;
        --accent: #C46A2B;
        --accent-dark: #8F451C;
        --accent-soft: #F5E4D5;
    }

    html, body, [class*="css"] {
        font-family: "Avenir Next", "Avenir", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }

    * {
        accent-color: var(--accent);
    }

    .stApp {
        background-color: var(--background);
    }

    header[data-testid="stHeader"] {
        background-color: rgba(250, 247, 242, 0.92);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.8rem;
        max-width: 1120px;
    }

    section[data-testid="stSidebar"] {
        background-color: #F3EDE6;
        border-right: 1px solid var(--border);
    }

    .sidebar-caption {
        font-size: 0.86rem;
        color: var(--text-muted);
        line-height: 1.5;
        margin-top: 0.15rem;
        margin-bottom: 0.7rem;
    }

    .sidebar-note {
        font-size: 0.82rem;
        color: var(--text-muted);
        line-height: 1.48;
        margin-top: 0.7rem;
    }

    .hero-box {
        background-color: var(--surface);
        padding: 1.72rem 2.05rem;
        border: 1px solid var(--border);
        border-radius: 22px;
        box-shadow: 0 8px 24px rgba(31, 41, 55, 0.04);
        margin-bottom: 1.05rem;
    }

    .eyebrow {
        color: var(--accent-dark);
        font-size: 0.72rem;
        font-weight: 650;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .hero-title {
        font-size: 2.22rem;
        font-weight: 540;
        line-height: 1.07;
        letter-spacing: -0.045em;
        color: var(--text-main);
        margin-bottom: 0.65rem;
    }

    .hero-title-accent {
        color: var(--accent-dark);
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: var(--text-muted);
        max-width: 860px;
        line-height: 1.55;
        margin-bottom: 0.85rem;
    }

    .title-line {
        width: 62px;
        height: 2px;
        background-color: var(--accent);
        margin-top: 0.2rem;
        margin-bottom: 0.85rem;
    }

    .pill {
        display: inline-block;
        padding: 0.28rem 0.62rem;
        background-color: var(--accent-soft);
        color: var(--accent-dark);
        border: 1px solid #E2BEA1;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 520;
        margin-right: 0.32rem;
        margin-bottom: 0.28rem;
    }

    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.85rem;
        margin-top: 0.95rem;
        margin-bottom: 1.35rem;
    }

    .kpi-item {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 17px;
        box-shadow: 0 6px 16px rgba(31, 41, 55, 0.03);
        padding: 0.82rem 0.85rem;
        min-height: 98px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }

    .kpi-value {
        font-size: 1.08rem;
        font-weight: 570;
        color: var(--text-main);
        letter-spacing: -0.02em;
        margin-bottom: 0.16rem;
    }

    .kpi-label {
        font-size: 0.79rem;
        color: var(--text-muted);
        line-height: 1.35;
    }

    .section-title {
        font-size: 1.14rem;
        font-weight: 540;
        color: var(--text-main);
        margin-top: 1.35rem;
        margin-bottom: 0.68rem;
        letter-spacing: -0.02em;
    }

    .section-subtitle {
        font-size: 0.93rem;
        color: var(--text-muted);
        line-height: 1.55;
        margin-bottom: 0.9rem;
    }

    .info-card-title {
        font-size: 1rem;
        font-weight: 540;
        color: var(--text-main);
        margin-bottom: 0.55rem;
        letter-spacing: -0.01em;
    }

    .small-muted {
        font-size: 0.91rem;
        color: var(--text-muted);
        line-height: 1.56;
    }

    .custom-input-card,
    .custom-summary-card,
    .strategy-box,
    .comparison-card,
    .comparison-card-highlight,
    .period-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(31, 41, 55, 0.03);
    }

    .custom-input-card {
        padding: 1.2rem 1.25rem;
        margin-top: 0.8rem;
        margin-bottom: 1.2rem;
    }

    .interpretation-box {
        background-color: #FFF7ED;
        border: 1px solid #E2BEA1;
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        color: var(--text-main);
        line-height: 1.55;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }

    .custom-summary-grid,
    .comparison-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 1rem;
        margin-bottom: 1.35rem;
    }

    .custom-summary-card {
        padding: 1rem 1rem;
        min-height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .custom-summary-value {
        font-size: 1.35rem;
        font-weight: 540;
        color: var(--accent-dark);
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }

    .custom-summary-label {
        font-size: 0.84rem;
        color: var(--text-muted);
        line-height: 1.35;
    }

    .strategy-box {
        padding: 1.25rem 1.35rem;
        margin-top: 0.8rem;
        margin-bottom: 1.2rem;
    }

    .strategy-main {
        font-size: 1.5rem;
        font-weight: 520;
        color: var(--accent-dark);
        letter-spacing: -0.02em;
        margin-top: 0.2rem;
        margin-bottom: 0.35rem;
    }

    .comparison-card,
    .comparison-card-highlight {
        padding: 1rem 1rem;
        min-height: 155px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .comparison-card-highlight {
        background-color: #FFF7ED;
        border: 1px solid #E2BEA1;
    }

    .comparison-title {
        font-size: 0.84rem;
        color: var(--text-muted);
        line-height: 1.35;
        margin-bottom: 0.35rem;
    }

    .comparison-main {
        font-size: 1.42rem;
        font-weight: 560;
        color: var(--text-main);
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
    }

    .comparison-main-accent {
        font-size: 1.52rem;
        font-weight: 580;
        color: var(--accent-dark);
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
    }

    .comparison-detail {
        font-size: 0.78rem;
        color: var(--text-muted);
        line-height: 1.35;
    }

    .timeline-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 1rem;
        margin-top: 0.9rem;
        margin-bottom: 1.4rem;
    }

    .period-card {
        padding: 1rem 1rem;
        min-height: 145px;
    }

    .period-label {
        color: var(--text-muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }

    .period-name {
        color: var(--text-main);
        font-size: 0.95rem;
        font-weight: 520;
        margin-bottom: 0.55rem;
    }

    .period-discount {
        color: var(--accent-dark);
        font-size: 1.55rem;
        font-weight: 540;
        margin-bottom: 0.45rem;
    }

    .period-detail {
        color: var(--text-muted);
        font-size: 0.84rem;
        line-height: 1.35;
    }

    .stButton > button {
        background-color: var(--accent);
        color: white;
        border-radius: 999px;
        border: 1px solid var(--accent);
        font-weight: 520;
        padding: 0.6rem 1.15rem;
    }

    .stButton > button:hover {
        background-color: var(--accent-dark);
        color: white;
        border: 1px solid var(--accent-dark);
    }

    @media (max-width: 900px) {
        .kpi-strip,
        .comparison-grid,
        .custom-summary-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        .hero-title {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# SIDEBAR

with st.sidebar:
    st.markdown("## Control Panel")

    st.markdown(
        """
        <div class="sidebar-caption">
        Scegli la modalità operativa del tool.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Modalità")

    modalita_avvio = st.radio(
        label="Modalità operativa",
        options=[
            "Nuova simulazione",
            "Esempio precompilato"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div class="sidebar-note">
        La nuova simulazione configura un prodotto personalizzato.
        L'esempio precompilato usa i casi studio della tesi.
        </div>
        """,
        unsafe_allow_html=True
    )


# HERO

st.markdown(
    """
    <div class="hero-box">
        <div class="eyebrow">Retail analytics · Markdown optimization</div>
        <div class="hero-title">
            Markdown Optimization<br>
            <span class="hero-title-accent">Decision Support Tool</span>
        </div>
        <div class="title-line"></div>
        <div class="hero-subtitle">
            Strumento applicativo per simulare decisioni di markdown nella Grande Distribuzione Organizzata,
            integrando stock residuo, domanda attesa, incertezza e rischio di invenduto.
        </div>
        <span class="pill">Markdown optimization</span>
        <span class="pill">Monte Carlo simulation</span>
        <span class="pill">Retail decision support</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="kpi-strip">
        <div class="kpi-item">
            <div class="kpi-value">Prodotto</div>
            <div class="kpi-label">prezzo, costo, stock</div>
        </div>
        <div class="kpi-item">
            <div class="kpi-value">Domanda</div>
            <div class="kpi-label">scenario e incertezza</div>
        </div>
        <div class="kpi-item">
            <div class="kpi-value">Markdown</div>
            <div class="kpi-label">percorsi alternativi di sconto</div>
        </div>
        <div class="kpi-item">
            <div class="kpi-value">Raccomandazione</div>
            <div class="kpi-label">decisione finale e indicatori</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# MODALITÀ ESEMPIO PRECOMPILATO

if modalita_avvio == "Esempio precompilato":

    st.markdown('<div class="section-title">Simulazione da esempio precompilato</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-subtitle">
        Seleziona uno dei casi studio della tesi e una condizione di domanda.
        I parametri sono già definiti nel modello.
        </div>
        """,
        unsafe_allow_html=True
    )

    col_input_1, col_input_2 = st.columns(2)

    with col_input_1:
        prodotto_selezionato = st.selectbox(
            "Prodotto",
            options=modello.df_prodotti["prodotto"].tolist(),
            index=0
        )

    with col_input_2:
        scenario_selezionato = st.selectbox(
            "Condizione di domanda",
            options=modello.df_scenari["scenario"].tolist(),
            index=1
        )

    dati_prodotto = modello.df_prodotti[
        modello.df_prodotti["prodotto"] == prodotto_selezionato
    ].copy()

    stock_iniziale = dati_prodotto["stock_iniziale"].iloc[0]
    numero_periodi = int(dati_prodotto["numero_periodi"].iloc[0])
    prezzo_lordo = dati_prodotto["prezzo_lordo"].iloc[0]
    iva = dati_prodotto["iva"].iloc[0]
    prezzo_netto = prezzo_lordo / (1 + iva)

    df_parametri_prodotto = dati_prodotto[
        [
            "prodotto",
            "formato_grammi",
            "numero_periodi",
            "prezzo_lordo",
            "iva",
            "beta",
            "stock_iniziale",
            "incertezza_min",
            "incertezza_moda",
            "incertezza_max"
        ]
    ].copy()

    df_parametri_prodotto["prezzo_netto_calcolato"] = prezzo_netto

    df_parametri_prodotto = df_parametri_prodotto.rename(columns={
        "prodotto": "Prodotto",
        "formato_grammi": "Formato (g)",
        "numero_periodi": "Numero periodi",
        "prezzo_lordo": "Prezzo cliente iniziale IVA inclusa",
        "prezzo_netto_calcolato": "Prezzo netto usato nei calcoli",
        "iva": "IVA",
        "beta": "Beta",
        "stock_iniziale": "Stock iniziale",
        "incertezza_min": "Incertezza min",
        "incertezza_moda": "Incertezza moda",
        "incertezza_max": "Incertezza max"
    })

    with st.expander("Mostra i parametri caricati per il caso studio"):
        st.dataframe(df_parametri_prodotto, width="stretch", hide_index=True)

    raccomandazione = modello.mostra_raccomandazione(
        nome_prodotto=prodotto_selezionato,
        nome_scenario=scenario_selezionato
    ).iloc[0]

    percorso_finale = raccomandazione["decisione_finale_modello"]
    invenduto_medio_atteso = raccomandazione["invenduto_medio_atteso"]
    vendite_medie_attese = stock_iniziale - invenduto_medio_atteso
    sell_through_atteso = vendite_medie_attese / stock_iniziale * 100
    sell_through_atteso = max(min(sell_through_atteso, 100), 0)

    benchmark_no_sconto = calcola_benchmark_no_sconto(
        nome_prodotto=prodotto_selezionato,
        nome_scenario=scenario_selezionato,
        numero_periodi=numero_periodi,
        stock_iniziale=stock_iniziale,
        numero_simulazioni=modello.NUM_SIMULAZIONI_MONTECARLO
    )

    risultato_consigliato = raccomandazione["risultato_economico_atteso"]
    risultato_no_sconto = benchmark_no_sconto["risultato_economico_medio"]
    differenza_vs_no_sconto = risultato_consigliato - risultato_no_sconto

    vendite_medie_no_sconto = benchmark_no_sconto["vendite_medie"]
    differenza_vendite = vendite_medie_attese - vendite_medie_no_sconto

    sell_through_no_sconto = benchmark_no_sconto["sell_through_medio_percentuale"]
    differenza_sell_through = sell_through_atteso - sell_through_no_sconto

    stock_residuo_no_sconto = benchmark_no_sconto["invenduto_medio"]
    differenza_stock_residuo = invenduto_medio_atteso - stock_residuo_no_sconto

    if differenza_vs_no_sconto >= 0:
        nota_differenza = "guadagno stimato in più rispetto a non applicare sconti"
    else:
        nota_differenza = "risultato stimato inferiore rispetto a non applicare sconti"

    st.markdown('<div class="section-title">Raccomandazione del modello</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="strategy-box">
            <div class="small-muted">Strategia di markdown consigliata</div>
            <div class="strategy-main">{formatta_percorso_sconti(percorso_finale)}</div>
            <div class="small-muted">
                La decisione finale è basata sulla simulazione Monte Carlo, che valuta più possibili
                realizzazioni della domanda a partire dai livelli di incertezza impostati.
                <br><br>
                I prezzi visualizzati nell'interfaccia sono prezzi cliente IVA inclusa. I risultati economici
                del modello sono calcolati su valori netti IVA esclusa.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Impatto rispetto a non applicare sconti</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="comparison-grid">
            <div class="comparison-card-highlight">
                <div class="comparison-title">Differenza economica</div>
                <div class="comparison-main-accent">{formatta_delta_euro(differenza_vs_no_sconto)}</div>
                <div class="comparison-detail">
                    {nota_differenza}<br>
                    Strategia: {formatta_euro(risultato_consigliato)} · Nessuno sconto: {formatta_euro(risultato_no_sconto)}
                </div>
            </div>
            <div class="comparison-card">
                <div class="comparison-title">Unità vendute medie</div>
                <div class="comparison-main">{formatta_delta_numero(differenza_vendite)}</div>
                <div class="comparison-detail">
                    Strategia: {formatta_numero(vendite_medie_attese)} unità<br>
                    Nessuno sconto: {formatta_numero(vendite_medie_no_sconto)} unità
                </div>
            </div>
            <div class="comparison-card">
                <div class="comparison-title">Sell-through atteso</div>
                <div class="comparison-main">{formatta_punti_percentuali(differenza_sell_through)}</div>
                <div class="comparison-detail">
                    Strategia: {formatta_percentuale(sell_through_atteso)}<br>
                    Nessuno sconto: {formatta_percentuale(sell_through_no_sconto)}
                </div>
            </div>
            <div class="comparison-card">
                <div class="comparison-title">Stock residuo medio</div>
                <div class="comparison-main">{formatta_delta_numero(differenza_stock_residuo)}</div>
                <div class="comparison-detail">
                    Strategia: {formatta_numero(invenduto_medio_atteso)} unità<br>
                    Nessuno sconto: {formatta_numero(stock_residuo_no_sconto)} unità
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="interpretation-box">
            <b>Come leggere la raccomandazione:</b><br>
            Il modello massimizza il risultato economico atteso, non forza necessariamente lo stock residuo a zero.
            Può quindi suggerire una strategia che lascia una parte di invenduto se vendere ulteriori unità richiederebbe
            sconti troppo elevati e una perdita di margine superiore al beneficio ottenuto.
        </div>
        """,
        unsafe_allow_html=True
    )

    df_sconti_periodo = crea_tabella_sconti_periodo(
        nome_prodotto=prodotto_selezionato,
        percorso_sconti=percorso_finale
    )

    st.markdown('<div class="section-title">Strategia consigliata per periodo</div>', unsafe_allow_html=True)
    st.markdown(crea_timeline_sconti(df_sconti_periodo), unsafe_allow_html=True)

    with st.expander("Mostra tabella dettagliata degli sconti"):
        st.dataframe(
            df_sconti_periodo[
                [
                    "Periodo",
                    "Descrizione periodo",
                    "Sconto consigliato",
                    "Prezzo lordo post-sconto",
                    "Fattore attrattività",
                    "Domanda base"
                ]
            ],
            width="stretch",
            hide_index=True
        )

    st.markdown('<div class="section-title">Confronto tra approcci</div>', unsafe_allow_html=True)

    df_confronto = modello.mostra_confronto_approcci(
        nome_prodotto=prodotto_selezionato,
        nome_scenario=scenario_selezionato
    ).copy()

    df_confronto_output = df_confronto[
        [
            "approccio",
            "percorso_sconti",
            "risultato_economico_medio",
            "invenduto_medio",
            "probabilita_spreco_percentuale"
        ]
    ].copy()

    df_no_sconto = pd.DataFrame([{
        "approccio": "Nessuno sconto",
        "percorso_sconti": benchmark_no_sconto["percorso_sconti"],
        "risultato_economico_medio": benchmark_no_sconto["risultato_economico_medio"],
        "invenduto_medio": benchmark_no_sconto["invenduto_medio"],
        "probabilita_spreco_percentuale": benchmark_no_sconto["probabilita_stock_residuo_percentuale"]
    }])

    df_confronto_output = pd.concat([df_no_sconto, df_confronto_output], ignore_index=True)

    df_confronto_output["Approccio"] = df_confronto_output["approccio"].apply(etichetta_approccio)
    df_confronto_output["Percorso sconti"] = df_confronto_output["percorso_sconti"].apply(formatta_percorso_sconti)
    df_confronto_output["Differenza vs nessuno sconto"] = (
        df_confronto_output["risultato_economico_medio"]
        - risultato_no_sconto
    )
    df_confronto_output["Sell-through medio"] = (
        (stock_iniziale - df_confronto_output["invenduto_medio"])
        / stock_iniziale
        * 100
    )

    df_confronto_output_finale = pd.DataFrame({
        "Approccio": df_confronto_output["Approccio"],
        "Percorso sconti": df_confronto_output["Percorso sconti"],
        "Risultato economico medio": df_confronto_output["risultato_economico_medio"].apply(formatta_euro),
        "Differenza vs nessuno sconto": df_confronto_output["Differenza vs nessuno sconto"].apply(formatta_delta_euro),
        "Sell-through medio": df_confronto_output["Sell-through medio"].apply(formatta_percentuale),
        "Unità medie residue": df_confronto_output["invenduto_medio"].apply(formatta_numero),
        "Probabilità stock residuo": df_confronto_output["probabilita_spreco_percentuale"].apply(formatta_percentuale)
    })

    st.dataframe(df_confronto_output_finale, width="stretch", hide_index=True)

    if raccomandazione["stessa_strategia"]:
        testo_interpretazione = (
            "In questo caso la simulazione Monte Carlo conferma la raccomandazione deterministica. "
            "L'inclusione dell'incertezza non modifica il percorso di sconti scelto, ma consente "
            "di stimare anche la variabilità del risultato e la probabilità di stock residuo."
        )
    else:
        testo_interpretazione = (
            "In questo caso la raccomandazione Monte Carlo non coincide con quella deterministica. "
            "La differenza deriva dal fatto che il modello stocastico valuta più possibili realizzazioni "
            "della domanda: la strategia finale è quindi quella che, nelle simulazioni, offre il miglior "
            "risultato economico medio considerando anche lo stock residuo."
        )

    st.markdown(
        f"""
        <div class="interpretation-box">
            <b>Lettura manageriale del risultato:</b><br>
            {testo_interpretazione}
        </div>
        """,
        unsafe_allow_html=True
    )


# MODALITÀ NUOVA SIMULAZIONE

else:

    st.markdown('<div class="section-title">Nuova simulazione</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-subtitle">
        Configura un prodotto personalizzato inserendo i parametri economici, operativi e previsionali.
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. DATI PRODOTTO

    st.markdown(
        """
        <div class="custom-input-card">
            <div class="info-card-title">1. Dati prodotto</div>
            <div class="small-muted">
            Inserisci le informazioni principali del prodotto. Il prezzo è inteso come prezzo cliente
            IVA inclusa; il modello utilizza il prezzo netto per i calcoli economici.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_prodotto_1, col_prodotto_2 = st.columns([1.4, 0.8])

    with col_prodotto_1:
        nome_prodotto_custom = st.text_input(
            "Nome prodotto",
            value="Prodotto personalizzato"
        )

    with col_prodotto_2:
        formato_custom = st.number_input(
            "Formato prodotto, opzionale (grammi)",
            min_value=0,
            max_value=5000,
            value=250,
            step=10
        )

    col_econ_1, col_econ_2, col_econ_3 = st.columns(3)

    with col_econ_1:
        prezzo_lordo_custom = st.number_input(
            "Prezzo cliente iniziale IVA inclusa (€)",
            min_value=0.10,
            value=3.99,
            step=0.10
        )

    with col_econ_2:
        iva_scelta = st.selectbox(
            "Aliquota IVA",
            options=["4%", "10%", "22%", "Personalizzata"],
            index=1
        )

        if iva_scelta == "4%":
            iva_custom = 0.04
        elif iva_scelta == "10%":
            iva_custom = 0.10
        elif iva_scelta == "22%":
            iva_custom = 0.22
        else:
            iva_custom = st.number_input(
                "IVA personalizzata",
                min_value=0.00,
                max_value=0.50,
                value=0.10,
                step=0.01
            )

    with col_econ_3:
        costo_unitario_custom = st.number_input(
            "Costo unitario netto (€)",
            min_value=0.01,
            value=1.80,
            step=0.10
        )

    col_stock_1, col_stock_2, col_stock_3 = st.columns(3)

    with col_stock_1:
        stock_iniziale_custom = st.number_input(
            "Stock iniziale disponibile",
            min_value=1,
            value=20,
            step=1
        )

    with col_stock_2:
        costo_smaltimento_custom = st.number_input(
            "Costo smaltimento unitario (€)",
            min_value=0.00,
            value=0.00,
            step=0.01
        )

    with col_stock_3:
        numero_periodi_custom = st.slider(
            "Numero di periodi decisionali",
            min_value=2,
            max_value=5,
            value=3,
            step=1
        )

    prezzo_netto_custom = prezzo_lordo_custom / (1 + iva_custom)
    costo_acquisto_totale_custom = stock_iniziale_custom * costo_unitario_custom

    st.markdown(
        f"""
        <div class="interpretation-box">
            <b>Nota su IVA e risultato economico:</b><br>
            Il prezzo cliente iniziale è pari a <b>{formatta_euro(prezzo_lordo_custom)} IVA inclusa</b>.
            Il prezzo netto usato nei calcoli è pari a <b>{formatta_euro(prezzo_netto_custom)}</b>.
            <br><br>
            Il risultato economico viene calcolato al netto IVA, considerando i ricavi netti generati dalle vendite,
            il costo di acquisto dell'intero stock iniziale e il costo di smaltimento dell'eventuale stock residuo.
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. SCENARIO

    st.markdown(
        """
        <div class="custom-input-card">
            <div class="info-card-title">2. Condizione generale della domanda</div>
            <div class="small-muted">
            La condizione di domanda modifica la previsione centrale: può rappresentare un contesto
            debole, ordinario o favorevole rispetto alla domanda standard.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_scenario_1, col_scenario_2 = st.columns([1, 1])

    with col_scenario_1:
        scenario_custom = st.selectbox(
            "Condizione di domanda",
            options=[
                "Sfavorevole",
                "Ordinaria",
                "Favorevole",
                "Personalizzata"
            ],
            index=1
        )

        if scenario_custom == "Sfavorevole":
            fattore_scenario_custom = 0.85
        elif scenario_custom == "Ordinaria":
            fattore_scenario_custom = 1.00
        elif scenario_custom == "Favorevole":
            fattore_scenario_custom = 1.15
        else:
            fattore_scenario_custom = st.number_input(
                "Fattore scenario personalizzato",
                min_value=0.60,
                max_value=1.50,
                value=1.00,
                step=0.05
            )

    with col_scenario_2:
        if scenario_custom == "Sfavorevole":
            descrizione_scenario = "La domanda attesa viene ridotta del 15%."
        elif scenario_custom == "Ordinaria":
            descrizione_scenario = "La domanda attesa resta invariata."
        elif scenario_custom == "Favorevole":
            descrizione_scenario = "La domanda attesa viene aumentata del 15%."
        else:
            variazione_domanda = (fattore_scenario_custom - 1) * 100

            if variazione_domanda >= 0:
                descrizione_scenario = (
                    f"La domanda attesa viene aumentata di circa {formatta_percentuale(variazione_domanda)}."
                )
            else:
                descrizione_scenario = (
                    f"La domanda attesa viene ridotta di circa {formatta_percentuale(abs(variazione_domanda))}."
                )

        st.markdown(
            f"""
            <div class="custom-input-card">
                <div class="custom-summary-value">{formatta_numero(fattore_scenario_custom)}</div>
                <div class="custom-summary-label">fattore scenario applicato alla domanda attesa</div>
                <br>
                <div class="small-muted">{descrizione_scenario}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    numero_simulazioni_custom = st.slider(
        "Numero simulazioni Monte Carlo",
        min_value=50,
        max_value=5000,
        value=500,
        step=50
    )

    st.markdown(
        """
        <div class="interpretation-box">
            <b>Nota sul Monte Carlo:</b><br>
            Tutte le strategie vengono confrontate sugli stessi scenari casuali di domanda.
            In questo modo il confronto è più equo: una strategia non viene avvantaggiata solo perché
            valutata su simulazioni casualmente più favorevoli.
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. BETA

    st.markdown(
        """
        <div class="custom-input-card">
            <div class="info-card-title">3. Sensibilità allo sconto</div>
            <div class="small-muted">
            Il beta rappresenta quanto la domanda reagisce allo sconto. Un beta più basso indica
            minore risposta al markdown; un beta più alto indica maggiore risposta.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    modalita_beta_custom = st.radio(
        "Come vuoi impostare il beta?",
        options=[
            "Usa stima guidata",
            "Inserisci beta manualmente"
        ],
        index=0,
        horizontal=True,
        key="modalita_beta_custom"
    )

    if modalita_beta_custom == "Usa stima guidata":
        col_beta_1, col_beta_2, col_beta_3 = st.columns(3)

        with col_beta_1:
            sostituibilita_custom = st.selectbox(
                "Sostituibilità",
                options=["Bassa", "Media", "Alta"],
                index=1,
                key="sostituibilita_custom"
            )

        with col_beta_2:
            rilevanza_prezzo_custom = st.selectbox(
                "Rilevanza prezzo",
                options=["Bassa", "Media", "Alta"],
                index=1,
                key="rilevanza_prezzo_custom"
            )

        with col_beta_3:
            reazione_promozioni_custom = st.selectbox(
                "Reazione promozioni",
                options=["Bassa", "Media", "Alta"],
                index=1,
                key="reazione_promozioni_custom"
            )

        punteggio_totale_beta = (
            punteggio_beta(sostituibilita_custom)
            + punteggio_beta(rilevanza_prezzo_custom)
            + punteggio_beta(reazione_promozioni_custom)
        )

        if punteggio_totale_beta == 3:
            beta_custom = 0.80
            classe_beta_custom = "Bassa sensibilità allo sconto"
            descrizione_beta_custom = "Il prodotto sembra poco sensibile al markdown."
        elif punteggio_totale_beta == 4:
            beta_custom = 1.00
            classe_beta_custom = "Sensibilità medio-bassa allo sconto"
            descrizione_beta_custom = "Il prodotto presenta una sensibilità contenuta."
        elif punteggio_totale_beta <= 6:
            beta_custom = 1.20
            classe_beta_custom = "Sensibilità media allo sconto"
            descrizione_beta_custom = "Il prodotto presenta una sensibilità media al markdown."
        elif punteggio_totale_beta <= 8:
            beta_custom = 1.50
            classe_beta_custom = "Sensibilità medio-alta allo sconto"
            descrizione_beta_custom = "Il prodotto sembra reagire in modo significativo al markdown."
        else:
            beta_custom = 2.00
            classe_beta_custom = "Alta sensibilità allo sconto"
            descrizione_beta_custom = "Il prodotto sembra molto sensibile allo sconto."

        beta_suggerito_custom = beta_custom

        st.markdown(
            f"""
            <div class="interpretation-box">
                <b>Beta stimato:</b> {formatta_numero(beta_custom)}<br>
                <b>Classe:</b> {classe_beta_custom}<br>
                {descrizione_beta_custom}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        sostituibilita_custom = "Non utilizzata"
        rilevanza_prezzo_custom = "Non utilizzata"
        reazione_promozioni_custom = "Non utilizzata"
        punteggio_totale_beta = "Non applicato"
        beta_suggerito_custom = "Non applicato"
        classe_beta_custom = "Beta manuale"

        beta_custom = st.slider(
            "Beta manuale",
            min_value=0.80,
            max_value=2.00,
            value=1.20,
            step=0.05,
            key="slider_beta_manuale"
        )

        st.markdown(
            f"""
            <div class="interpretation-box">
                <b>Beta manuale impostato:</b> {formatta_numero(beta_custom)}.
            </div>
            """,
            unsafe_allow_html=True
        )

    # 4. ATTRATTIVITÀ RESIDUA

    st.markdown(
        """
        <div class="custom-input-card">
            <div class="info-card-title">4. Attrattività residua del prodotto</div>
            <div class="small-muted">
            Il fattore di attrattività residua riduce la domanda attesa nei diversi periodi
            quando il prodotto perde appeal commerciale, freschezza percepita o vicinanza alla scadenza.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    profili_freschezza = {
        "Stabile nell'orizzonte": [1.00, 1.00, 1.00, 1.00, 1.00],
        "Calo graduale moderato": [1.00, 0.95, 0.90, 0.80, 0.70],
        "Tenuta iniziale, calo finale marcato": [1.00, 1.00, 0.95, 0.85, 0.65],
        "Calo rapido e marcato": [1.00, 0.90, 0.75, 0.60, 0.45]
    }

    profilo_freschezza_custom = st.selectbox(
        "Profilo di attrattività residua",
        options=[
            "Stabile nell'orizzonte",
            "Calo graduale moderato",
            "Tenuta iniziale, calo finale marcato",
            "Calo rapido e marcato",
            "Personalizzato"
        ],
        index=1,
        key="profilo_attrattivita_residua"
    )

    if profilo_freschezza_custom == "Personalizzato":
        valori_base_personalizzati = adatta_profilo_freschezza(
            profili_freschezza["Calo graduale moderato"],
            numero_periodi_custom
        )

        fattori_freschezza_custom = []
        colonne_freschezza = st.columns(numero_periodi_custom)

        for indice_periodo in range(numero_periodi_custom):
            with colonne_freschezza[indice_periodo]:
                fattore_periodo = st.slider(
                    f"Periodo {indice_periodo + 1}",
                    min_value=0.30,
                    max_value=1.00,
                    value=float(valori_base_personalizzati[indice_periodo]),
                    step=0.05,
                    key=f"slider_attrattivita_{indice_periodo + 1}"
                )

                fattori_freschezza_custom.append(fattore_periodo)

    else:
        fattori_freschezza_custom = adatta_profilo_freschezza(
            profili_freschezza[profilo_freschezza_custom],
            numero_periodi_custom
        )

    df_freschezza_custom = pd.DataFrame({
        "Periodo": [f"Periodo {i + 1}" for i in range(numero_periodi_custom)],
        "Fattore attrattività residua": fattori_freschezza_custom,
        "Interpretazione": [
            interpreta_fattore_attrattivita(fattore)
            for fattore in fattori_freschezza_custom
        ]
    })

    st.markdown(
        f"""
        <div class="interpretation-box">
            <b>Fattori di attrattività residua impostati:</b><br>
            {formatta_lista_fattori(fattori_freschezza_custom)}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(df_freschezza_custom, width="stretch", hide_index=True)
   
    # 5. INCERTEZZA

    st.markdown(
        """
        <div class="custom-input-card">
            <div class="info-card-title">5. Incertezza della domanda</div>
            <div class="small-muted">
            L'incertezza misura quanto la domanda effettiva può oscillare rispetto alla previsione centrale.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    modalita_incertezza_custom = st.radio(
        "Come vuoi impostare l'incertezza?",
        options=[
            "Usa stima guidata",
            "Inserisci intervallo manualmente"
        ],
        index=0,
        horizontal=True,
        key="modalita_incertezza_custom"
    )

    if modalita_incertezza_custom == "Usa stima guidata":
        col_inc_1, col_inc_2, col_inc_3 = st.columns(3)

        with col_inc_1:
            variabilita_domanda_custom = st.selectbox(
                "Variabilità storica della domanda",
                options=["Bassa", "Media", "Alta"],
                index=1,
                key="variabilita_domanda_custom",
                help="Indica quanto le vendite del prodotto tendono normalmente a variare rispetto alla previsione."
            )

        with col_inc_2:
            dipendenza_fattori_esterni_custom = st.selectbox(
                "Esposizione a fattori esterni",
                options=["Limitata", "Media", "Elevata"],
                index=1,
                key="esposizione_fattori_esterni_custom",
                help="Indica quanto elementi come meteo, eventi, traffico nel punto vendita o promozioni concorrenti possono influenzare la domanda."
            )

        with col_inc_3:
            difficolta_previsione_custom = st.selectbox(
                "Prevedibilità a breve termine",
                options=["Bassa", "Media", "Alta"],
                index=1,
                key="difficolta_previsione_custom",
                help="Indica quanto è facile prevedere le vendite del prodotto nel breve periodo."
            )

        punteggio_variabilita = {
            "Bassa": 1,
            "Media": 2,
            "Alta": 3
        }

        punteggio_fattori_esterni = {
            "Limitata": 1,
            "Media": 2,
            "Elevata": 3
        }

        punteggio_prevedibilita = {
            "Alta": 1,
            "Media": 2,
            "Bassa": 3
        }

        punteggio_totale_incertezza = (
            punteggio_variabilita[variabilita_domanda_custom]
            + punteggio_fattori_esterni[dipendenza_fattori_esterni_custom]
            + punteggio_prevedibilita[difficolta_previsione_custom]
        )

        metodo_incertezza_custom = "Stima guidata tramite domande"

        if punteggio_totale_incertezza <= 4:
            incertezza_min_custom = 0.90
            incertezza_moda_custom = 1.00
            incertezza_max_custom = 1.10
            classe_incertezza_custom = "Bassa incertezza"
            descrizione_incertezza_custom = "La domanda appare relativamente stabile e prevedibile."
        elif punteggio_totale_incertezza <= 6:
            incertezza_min_custom = 0.80
            incertezza_moda_custom = 1.00
            incertezza_max_custom = 1.20
            classe_incertezza_custom = "Media incertezza"
            descrizione_incertezza_custom = "La domanda può variare in modo moderato."
        elif punteggio_totale_incertezza <= 8:
            incertezza_min_custom = 0.70
            incertezza_moda_custom = 1.00
            incertezza_max_custom = 1.30
            classe_incertezza_custom = "Alta incertezza"
            descrizione_incertezza_custom = "La domanda presenta una variabilità significativa."
        else:
            incertezza_min_custom = 0.60
            incertezza_moda_custom = 1.00
            incertezza_max_custom = 1.40
            classe_incertezza_custom = "Molto alta incertezza"
            descrizione_incertezza_custom = "La domanda è fortemente instabile e difficilmente prevedibile."

    else:
        variabilita_domanda_custom = "Non utilizzata"
        dipendenza_fattori_esterni_custom = "Non utilizzata"
        difficolta_previsione_custom = "Non utilizzata"
        punteggio_totale_incertezza = "Non applicato"
        classe_incertezza_custom = "Intervallo manuale"
        metodo_incertezza_custom = "Intervallo inserito manualmente"

        col_manual_inc_1, col_manual_inc_2 = st.columns(2)

        with col_manual_inc_1:
            incertezza_min_custom = st.slider(
                "Scenario peggiore plausibile",
                min_value=0.50,
                max_value=0.95,
                value=0.80,
                step=0.05,
                key="slider_incertezza_min"
            )

        incertezza_moda_custom = 1.00

        with col_manual_inc_2:
            incertezza_max_custom = st.slider(
                "Scenario migliore plausibile",
                min_value=1.05,
                max_value=1.50,
                value=1.20,
                step=0.05,
                key="slider_incertezza_max"
            )

        descrizione_incertezza_custom = "L'intervallo di incertezza è stato definito manualmente."

    riduzione_domanda_min = (1 - incertezza_min_custom) * 100
    aumento_domanda_max = (incertezza_max_custom - 1) * 100

    st.markdown(
        f"""
        <div class="interpretation-box">
            <b>Incertezza della domanda impostata:</b><br>
            Metodo: <b>{metodo_incertezza_custom}</b><br>
            Classe: <b>{classe_incertezza_custom}</b><br>
            Intervallo: <b>{formatta_numero(incertezza_min_custom)} - {formatta_numero(incertezza_moda_custom)} - {formatta_numero(incertezza_max_custom)}</b>
            <br><br>
            {descrizione_incertezza_custom}
            <br><br>
            La domanda potrà oscillare da circa <b>-{formatta_percentuale(riduzione_domanda_min)}</b>
            a circa <b>+{formatta_percentuale(aumento_domanda_max)}</b> rispetto alla previsione centrale.
        </div>
        """,
        unsafe_allow_html=True
    )

    # 6. VENDITE A PREZZO PIENO

    st.markdown(
        """
        <div class="custom-input-card">
            <div class="info-card-title">6. Vendite attese a prezzo pieno</div>
            <div class="small-muted">
            Questa sezione stima quante unità potrebbero essere vendute a prezzo pieno prima dell'applicazione
            dei markdown. Nel modello, questo valore viene trasformato nella domanda base per periodo.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    modalita_domanda_base_custom = st.radio(
        "Come vuoi impostare le vendite attese a prezzo pieno?",
        options=[
            "Usa stima guidata",
            "Inserisci domanda per periodo manualmente"
        ],
        index=0,
        horizontal=True,
        key="modalita_domanda_base_custom"
    )

    if modalita_domanda_base_custom == "Usa stima guidata":
        quota_stock_prezzo_pieno_custom = st.slider(
            "Quota dello stock vendibile a prezzo pieno",
            min_value=0,
            max_value=100,
            value=70,
            step=5,
            key="quota_stock_prezzo_pieno_custom"
        )

        domanda_base_totale_grezza_custom = (
            stock_iniziale_custom
            * quota_stock_prezzo_pieno_custom
            / 100
        )

        domanda_base_totale_custom = arrotonda_per_eccesso_mezzo(
            domanda_base_totale_grezza_custom
        )

        andamento_domanda_base_custom = st.radio(
            "Vuoi modificare l'andamento della domanda tra i periodi?",
            options=[
                "No, distribuzione uniforme",
                "Sì, più alta nei primi periodi",
                "Sì, più alta negli ultimi periodi"
            ],
            index=0,
            key="andamento_domanda_base_custom"
        )

        if andamento_domanda_base_custom == "No, distribuzione uniforme":
            profilo_temporale_domanda_custom = "Distribuzione uniforme"
            descrizione_andamento_domanda = "La domanda base viene distribuita in modo uniforme tra i periodi."
        elif andamento_domanda_base_custom == "Sì, più alta nei primi periodi":
            profilo_temporale_domanda_custom = "Più alta nei primi periodi"
            descrizione_andamento_domanda = "La domanda base viene concentrata maggiormente nei primi periodi."
        else:
            profilo_temporale_domanda_custom = "Più alta negli ultimi periodi"
            descrizione_andamento_domanda = "La domanda base viene concentrata maggiormente negli ultimi periodi."

        domanda_base_periodi_custom = distribuisci_domanda_base(
            domanda_totale=domanda_base_totale_custom,
            numero_periodi=numero_periodi_custom,
            profilo_temporale=profilo_temporale_domanda_custom
        )

        metodo_domanda_base_custom = "Stima guidata tramite quota di stock"

    else:
        quota_stock_prezzo_pieno_custom = "Non applicata"
        profilo_temporale_domanda_custom = "Manuale"
        andamento_domanda_base_custom = "Manuale"
        metodo_domanda_base_custom = "Domanda inserita manualmente per periodo"
        descrizione_andamento_domanda = "La domanda base è stata inserita manualmente periodo per periodo."

        domanda_base_periodi_custom = []
        colonne_domanda_base = st.columns(numero_periodi_custom)

        valore_default_domanda_periodo = arrotonda_per_eccesso_mezzo(
            stock_iniziale_custom * 0.70 / numero_periodi_custom
        )

        for indice_periodo in range(numero_periodi_custom):
            with colonne_domanda_base[indice_periodo]:
                domanda_periodo = st.number_input(
                    f"Periodo {indice_periodo + 1}",
                    min_value=0.00,
                    value=float(valore_default_domanda_periodo),
                    step=0.50,
                    key=f"domanda_base_periodo_{indice_periodo + 1}"
                )

                domanda_base_periodi_custom.append(domanda_periodo)

        domanda_base_totale_custom = round(sum(domanda_base_periodi_custom), 2)

    domanda_base_media_periodo_custom = (
        sum(domanda_base_periodi_custom)
        / len(domanda_base_periodi_custom)
    )

    rapporto_domanda_stock_custom = (
        domanda_base_totale_custom
        / stock_iniziale_custom
        * 100
    )

    st.markdown(
        f"""
        <div class="interpretation-box">
            <b>Vendite attese a prezzo pieno impostate:</b><br>
            Metodo: <b>{metodo_domanda_base_custom}</b><br>
            Domanda base totale: <b>{formatta_numero(domanda_base_totale_custom)} unità</b><br>
            Incidenza sullo stock iniziale: <b>{formatta_percentuale(rapporto_domanda_stock_custom)}</b><br>
            Andamento temporale: <b>{profilo_temporale_domanda_custom}</b>
            <br><br>
            Domanda base per periodo: <b>{formatta_lista_fattori(domanda_base_periodi_custom)}</b>
            <br><br>
            {descrizione_andamento_domanda}
            <br><br>
            Nota: se la quota è pari a 0%, il modello interpreta una domanda base nulla.
            Per simulare un prodotto che a prezzo pieno vende pochissimo ma non zero, usa valori come 5% o 10%.
        </div>
        """,
        unsafe_allow_html=True
    )

    df_domanda_base_custom = pd.DataFrame({
        "Periodo": [f"Periodo {i + 1}" for i in range(numero_periodi_custom)],
        "Domanda base": domanda_base_periodi_custom
    })

    st.dataframe(df_domanda_base_custom, width="stretch", hide_index=True)

    # ANTEPRIMA PARAMETRI

    fattore_freschezza_iniziale = fattori_freschezza_custom[0]
    fattore_freschezza_finale = fattori_freschezza_custom[-1]
    riduzione_freschezza_finale = (1 - fattore_freschezza_finale) * 100

    st.markdown('<div class="section-title">Anteprima parametri inseriti</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="custom-summary-grid">
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_euro(prezzo_lordo_custom)}</div>
                <div class="custom-summary-label">prezzo cliente iniziale IVA inclusa</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_euro(prezzo_netto_custom)}</div>
                <div class="custom-summary-label">prezzo netto usato nei calcoli economici</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{int(stock_iniziale_custom)}</div>
                <div class="custom-summary-label">unità di stock iniziale</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{numero_periodi_custom}</div>
                <div class="custom-summary-label">periodi decisionali di markdown</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="custom-summary-grid">
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_euro(costo_unitario_custom)}</div>
                <div class="custom-summary-label">costo unitario netto</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_euro(costo_acquisto_totale_custom)}</div>
                <div class="custom-summary-label">costo dell'intero stock iniziale</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{scenario_custom}</div>
                <div class="custom-summary-label">condizione generale di domanda</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_numero(beta_custom)}</div>
                <div class="custom-summary-label">beta utilizzato nella simulazione</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="custom-summary-grid">
            <div class="custom-summary-card">
                <div class="custom-summary-value">{profilo_freschezza_custom}</div>
                <div class="custom-summary-label">profilo di attrattività residua</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_numero(fattore_freschezza_iniziale)}</div>
                <div class="custom-summary-label">fattore attrattività iniziale</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_numero(fattore_freschezza_finale)}</div>
                <div class="custom-summary-label">fattore attrattività finale</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_percentuale(riduzione_freschezza_finale)}</div>
                <div class="custom-summary-label">riduzione domanda finale per attrattività</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="custom-summary-grid">
            <div class="custom-summary-card">
                <div class="custom-summary-value">{classe_incertezza_custom}</div>
                <div class="custom-summary-label">classe di incertezza</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_numero(incertezza_min_custom)}</div>
                <div class="custom-summary-label">scenario peggiore plausibile</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_numero(incertezza_moda_custom)}</div>
                <div class="custom-summary-label">scenario centrale</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_numero(incertezza_max_custom)}</div>
                <div class="custom-summary-label">scenario migliore plausibile</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="custom-summary-grid">
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_numero(domanda_base_totale_custom)}</div>
                <div class="custom-summary-label">vendite attese totali a prezzo pieno</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_percentuale(rapporto_domanda_stock_custom)}</div>
                <div class="custom-summary-label">vendite attese rispetto allo stock</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{formatta_numero(domanda_base_media_periodo_custom)}</div>
                <div class="custom-summary-label">domanda base media per periodo</div>
            </div>
            <div class="custom-summary-card">
                <div class="custom-summary-value">{profilo_temporale_domanda_custom}</div>
                <div class="custom-summary-label">andamento domanda base</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    df_input_base_custom = pd.DataFrame([{
        "Nome prodotto": nome_prodotto_custom,
        "Formato (g)": formato_custom,
        "Prezzo cliente iniziale IVA inclusa": prezzo_lordo_custom,
        "IVA": iva_custom,
        "Prezzo netto usato nei calcoli": prezzo_netto_custom,
        "Costo unitario netto": costo_unitario_custom,
        "Costo acquisto intero stock iniziale": costo_acquisto_totale_custom,
        "Costo smaltimento unitario": costo_smaltimento_custom,
        "Stock iniziale": stock_iniziale_custom,
        "Numero periodi": numero_periodi_custom,
        "Condizione domanda": scenario_custom,
        "Fattore scenario": fattore_scenario_custom,
        "Numero simulazioni": numero_simulazioni_custom,
        "Sostituibilità": sostituibilita_custom,
        "Rilevanza prezzo": rilevanza_prezzo_custom,
        "Reazione promozioni": reazione_promozioni_custom,
        "Punteggio beta": punteggio_totale_beta,
        "Beta suggerito": beta_suggerito_custom,
        "Beta utilizzato": beta_custom,
        "Profilo attrattività residua": profilo_freschezza_custom,
        "Fattori attrattività residua": formatta_lista_fattori(fattori_freschezza_custom),
        "Variabilità storica domanda": variabilita_domanda_custom,
        "Dipendenza da fattori esterni": dipendenza_fattori_esterni_custom,
        "Difficoltà previsione": difficolta_previsione_custom,
        "Punteggio incertezza": punteggio_totale_incertezza,
        "Classe incertezza": classe_incertezza_custom,
        "Incertezza min": incertezza_min_custom,
        "Incertezza moda": incertezza_moda_custom,
        "Incertezza max": incertezza_max_custom,
        "Modalità domanda base": modalita_domanda_base_custom,
        "Metodo domanda base": metodo_domanda_base_custom,
        "Quota stock vendibile a prezzo pieno": quota_stock_prezzo_pieno_custom,
        "Domanda base totale": domanda_base_totale_custom,
        "Domanda base media periodo": domanda_base_media_periodo_custom,
        "Domanda base rispetto stock (%)": rapporto_domanda_stock_custom,
        "Andamento domanda base": profilo_temporale_domanda_custom,
        "Domanda base per periodo": formatta_lista_fattori(domanda_base_periodi_custom)
    }])

    with st.expander("Mostra tabella tecnica degli input base"):
        st.dataframe(df_input_base_custom, width="stretch", hide_index=True)

    # OTTIMIZZAZIONE

    st.markdown('<div class="section-title">Raccomandazione per il prodotto personalizzato</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="custom-input-card">
            <div class="info-card-title">7. Ottimizzazione della strategia di markdown</div>
            <div class="small-muted">
            Il tool genera percorsi alternativi di sconto progressivo e li valuta prima in modo deterministico,
            usando la domanda centrale, e poi tramite simulazione Monte Carlo, considerando l'incertezza della domanda.
            <br><br>
            La raccomandazione finale corrisponde al percorso di sconti con il miglior risultato economico medio
            nella simulazione Monte Carlo.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Genera raccomandazione di markdown"):
        st.session_state["esegui_ottimizzazione_custom"] = True

    if st.session_state.get("esegui_ottimizzazione_custom", False):

        with st.spinner("Calcolo della raccomandazione in corso..."):
            risultati_custom = ottimizza_markdown_custom(
                prezzo_netto=prezzo_netto_custom,
                prezzo_lordo=prezzo_lordo_custom,
                costo_unitario=costo_unitario_custom,
                costo_smaltimento=costo_smaltimento_custom,
                stock_iniziale=stock_iniziale_custom,
                numero_periodi=numero_periodi_custom,
                domanda_base_periodi=domanda_base_periodi_custom,
                fattori_attrattivita=fattori_freschezza_custom,
                fattore_scenario=fattore_scenario_custom,
                beta=beta_custom,
                incertezza_min=incertezza_min_custom,
                incertezza_moda=incertezza_moda_custom,
                incertezza_max=incertezza_max_custom,
                numero_simulazioni=numero_simulazioni_custom
            )

        df_montecarlo_custom = risultati_custom["df_montecarlo"]
        riga_montecarlo_custom = risultati_custom["riga_montecarlo"]
        riga_deterministica_custom = risultati_custom["riga_deterministica"]
        df_dettaglio_finale_custom = risultati_custom["df_dettaglio_finale"]

        percorso_finale_custom = riga_montecarlo_custom["percorso_sconti"]
        percorso_deterministico_custom = riga_deterministica_custom["percorso_sconti"]
        percorso_no_sconto_custom = tuple([0.00] * numero_periodi_custom)

        riga_no_sconto_custom = df_montecarlo_custom[
            df_montecarlo_custom["percorso_sconti"].apply(
                lambda x: tuple(x) == percorso_no_sconto_custom
            )
        ].iloc[0]

        risultato_consigliato_custom = riga_montecarlo_custom["risultato_economico_medio"]
        risultato_no_sconto_custom = riga_no_sconto_custom["risultato_economico_medio"]

        differenza_vs_no_sconto_custom = (
            risultato_consigliato_custom
            - risultato_no_sconto_custom
        )

        differenza_vendite_custom = (
            riga_montecarlo_custom["vendite_totali_medie"]
            - riga_no_sconto_custom["vendite_totali_medie"]
        )

        differenza_sell_through_custom = (
            riga_montecarlo_custom["sell_through_medio_percentuale"]
            - riga_no_sconto_custom["sell_through_medio_percentuale"]
        )

        differenza_invenduto_custom = (
            riga_montecarlo_custom["invenduto_medio"]
            - riga_no_sconto_custom["invenduto_medio"]
        )

        if differenza_vs_no_sconto_custom > 0.01: 
            nota_differenza_custom = "risultato economico medio superiore rispetto a non applicare sconti"
        elif differenza_vs_no_sconto_custom < -0.01:
            nota_differenza_custom = "risultato economico medio inferiore rispetto a non applicare sconti"
        else: 
            nota_differenza_custom = "risultato economico medio equivalente rispetto a non applicare sconti"

        st.markdown(
            f"""
            <div class="strategy-box">
                <div class="small-muted">Strategia di markdown consigliata per il prodotto configurato</div>
                <div class="strategy-main">{formatta_percorso_sconti(percorso_finale_custom)}</div>
                <div class="small-muted">
                    La strategia è stata scelta confrontando percorsi progressivi di sconto e selezionando
                    quello con il miglior risultato economico medio nella simulazione Monte Carlo.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-title">Impatto rispetto a non applicare sconti</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="comparison-grid">
                <div class="comparison-card-highlight">
                    <div class="comparison-title">Differenza economica</div>
                    <div class="comparison-main-accent">{formatta_delta_euro(differenza_vs_no_sconto_custom)}</div>
                    <div class="comparison-detail">
                        {nota_differenza_custom}<br>
                        Strategia: {formatta_euro(risultato_consigliato_custom)} · Nessuno sconto: {formatta_euro(risultato_no_sconto_custom)}
                    </div>
                </div>
                <div class="comparison-card">
                    <div class="comparison-title">Unità vendute medie</div>
                    <div class="comparison-main">{formatta_delta_numero(differenza_vendite_custom)}</div>
                    <div class="comparison-detail">
                        Strategia: {formatta_numero(riga_montecarlo_custom["vendite_totali_medie"])} unità<br>
                        Nessuno sconto: {formatta_numero(riga_no_sconto_custom["vendite_totali_medie"])} unità
                    </div>
                </div>
                <div class="comparison-card">
                    <div class="comparison-title">Sell-through medio</div>
                    <div class="comparison-main">{formatta_punti_percentuali(differenza_sell_through_custom)}</div>
                    <div class="comparison-detail">
                        Strategia: {formatta_percentuale(riga_montecarlo_custom["sell_through_medio_percentuale"])}<br>
                        Nessuno sconto: {formatta_percentuale(riga_no_sconto_custom["sell_through_medio_percentuale"])}
                    </div>
                </div>
                <div class="comparison-card">
                    <div class="comparison-title">Stock residuo medio</div>
                    <div class="comparison-main">{formatta_delta_numero(differenza_invenduto_custom)}</div>
                    <div class="comparison-detail">
                        Strategia: {formatta_numero(riga_montecarlo_custom["invenduto_medio"])} unità<br>
                        Nessuno sconto: {formatta_numero(riga_no_sconto_custom["invenduto_medio"])} unità
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="interpretation-box">
                <b>Come leggere la raccomandazione:</b><br>
                Il modello massimizza il risultato economico atteso, non forza necessariamente lo stock residuo a zero.
                Una strategia può quindi lasciare una parte di invenduto se vendere ulteriori unità richiederebbe
                sconti troppo elevati e una perdita di margine superiore al beneficio ottenuto.
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("Mostra composizione tecnica del risultato economico"):
            df_breakdown_custom = pd.DataFrame([
                {
                    "Approccio": "Nessuno sconto",
                    "Percorso sconti": formatta_percorso_sconti(percorso_no_sconto_custom),
                    "Ricavi netti medi da vendite": riga_no_sconto_custom["ricavi_netto_medi"],
                    "Costo acquisto stock iniziale": riga_no_sconto_custom["costo_acquisto_totale"],
                    "Costo smaltimento medio": riga_no_sconto_custom["costo_smaltimento_medio"],
                    "Risultato economico medio": riga_no_sconto_custom["risultato_economico_medio"]
                },
                {
                    "Approccio": "Raccomandazione Monte Carlo",
                    "Percorso sconti": formatta_percorso_sconti(percorso_finale_custom),
                    "Ricavi netti medi da vendite": riga_montecarlo_custom["ricavi_netto_medi"],
                    "Costo acquisto stock iniziale": riga_montecarlo_custom["costo_acquisto_totale"],
                    "Costo smaltimento medio": riga_montecarlo_custom["costo_smaltimento_medio"],
                    "Risultato economico medio": riga_montecarlo_custom["risultato_economico_medio"]
                }
            ])

            df_breakdown_output = pd.DataFrame({
                "Approccio": df_breakdown_custom["Approccio"],
                "Percorso sconti": df_breakdown_custom["Percorso sconti"],
                "Ricavi netti medi da vendite": df_breakdown_custom["Ricavi netti medi da vendite"].apply(formatta_euro),
                "Costo acquisto stock iniziale": df_breakdown_custom["Costo acquisto stock iniziale"].apply(lambda x: "− " + formatta_euro(x)),
                "Costo smaltimento medio": df_breakdown_custom["Costo smaltimento medio"].apply(lambda x: "− " + formatta_euro(x)),
                "Risultato economico medio": df_breakdown_custom["Risultato economico medio"].apply(formatta_euro)
            })

            st.dataframe(
                df_breakdown_output,
                width="stretch",
                hide_index=True
            )

        st.markdown('<div class="section-title">Strategia consigliata per periodo</div>', unsafe_allow_html=True)

        st.markdown(
            crea_timeline_sconti(df_dettaglio_finale_custom),
            unsafe_allow_html=True
        )

        with st.expander("Mostra dettaglio della strategia consigliata"):
            st.dataframe(
                df_dettaglio_finale_custom[
                    [
                        "Periodo",
                        "Sconto consigliato",
                        "Prezzo lordo post-sconto",
                        "Domanda base",
                        "Fattore attrattività",
                        "Domanda attesa centrale",
                        "Vendite attese centrali",
                        "Stock residuo stimato"
                    ]
                ],
                width="stretch",
                hide_index=True
            )

        st.markdown('<div class="section-title">Confronto tra approcci</div>', unsafe_allow_html=True)

        riga_deterministica_mc = df_montecarlo_custom[
            df_montecarlo_custom["percorso_sconti"].apply(
                lambda x: tuple(x) == tuple(percorso_deterministico_custom)
            )
        ].iloc[0]

        df_confronto_custom = pd.DataFrame([
            {
                "Approccio": "Nessuno sconto",
                "Percorso sconti": formatta_percorso_sconti(percorso_no_sconto_custom),
                "Risultato economico medio": riga_no_sconto_custom["risultato_economico_medio"],
                "Differenza vs nessuno sconto": 0.00,
                "Sell-through medio": riga_no_sconto_custom["sell_through_medio_percentuale"],
                "Unità medie residue": riga_no_sconto_custom["invenduto_medio"],
                "Probabilità stock residuo": riga_no_sconto_custom["probabilita_stock_residuo_percentuale"]
            },
            {
                "Approccio": "Raccomandazione deterministica",
                "Percorso sconti": formatta_percorso_sconti(percorso_deterministico_custom),
                "Risultato economico medio": riga_deterministica_mc["risultato_economico_medio"],
                "Differenza vs nessuno sconto": riga_deterministica_mc["risultato_economico_medio"] - risultato_no_sconto_custom,
                "Sell-through medio": riga_deterministica_mc["sell_through_medio_percentuale"],
                "Unità medie residue": riga_deterministica_mc["invenduto_medio"],
                "Probabilità stock residuo": riga_deterministica_mc["probabilita_stock_residuo_percentuale"]
            },
            {
                "Approccio": "Raccomandazione Monte Carlo",
                "Percorso sconti": formatta_percorso_sconti(percorso_finale_custom),
                "Risultato economico medio": riga_montecarlo_custom["risultato_economico_medio"],
                "Differenza vs nessuno sconto": differenza_vs_no_sconto_custom,
                "Sell-through medio": riga_montecarlo_custom["sell_through_medio_percentuale"],
                "Unità medie residue": riga_montecarlo_custom["invenduto_medio"],
                "Probabilità stock residuo": riga_montecarlo_custom["probabilita_stock_residuo_percentuale"]
            }
        ])

        df_confronto_custom_output = pd.DataFrame({
            "Approccio": df_confronto_custom["Approccio"],
            "Percorso sconti": df_confronto_custom["Percorso sconti"],
            "Risultato economico medio": df_confronto_custom["Risultato economico medio"].apply(formatta_euro),
            "Differenza vs nessuno sconto": df_confronto_custom["Differenza vs nessuno sconto"].apply(formatta_delta_euro),
            "Sell-through medio": df_confronto_custom["Sell-through medio"].apply(formatta_percentuale),
            "Unità medie residue": df_confronto_custom["Unità medie residue"].apply(formatta_numero),
            "Probabilità stock residuo": df_confronto_custom["Probabilità stock residuo"].apply(formatta_percentuale)
        })

        st.dataframe(
            df_confronto_custom_output,
            width="stretch",
            hide_index=True
        )

        stessa_strategia_custom = (
            tuple(percorso_finale_custom)
            == tuple(percorso_deterministico_custom)
        )

        if stessa_strategia_custom:
            testo_interpretazione_custom = (
                "In questa simulazione, la raccomandazione Monte Carlo conferma la strategia deterministica. "
                "L'incertezza della domanda non modifica quindi il percorso di markdown scelto, ma consente "
                "di stimare il risultato medio, la probabilità di stock residuo e la variabilità potenziale."
            )
        else:
            testo_interpretazione_custom = (
                "In questa simulazione, la raccomandazione Monte Carlo differisce dalla strategia deterministica. "
                "Questo accade perché il modello stocastico considera più possibili realizzazioni della domanda: "
                "la strategia finale è quindi quella che offre il miglior risultato economico medio tenendo conto "
                "dell'incertezza e del rischio di invenduto."
            )

        st.markdown(
            f"""
            <div class="interpretation-box">
                <b>Lettura manageriale del risultato:</b><br>
                {testo_interpretazione_custom}
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("Mostra le migliori strategie simulate"):
            df_top_strategie_custom = (
                df_montecarlo_custom
                .sort_values("risultato_economico_medio", ascending=False)
                .head(10)
                .copy()
            )

            df_top_strategie_custom_output = pd.DataFrame({
                "Percorso sconti": df_top_strategie_custom["percorso_sconti"].apply(formatta_percorso_sconti),
                "Risultato economico medio": df_top_strategie_custom["risultato_economico_medio"].apply(formatta_euro),
                "Risultato nel 5% peggiore dei casi": df_top_strategie_custom["risultato_economico_p5"].apply(formatta_euro),
                "Risultato nel 5% migliore dei casi": df_top_strategie_custom["risultato_economico_p95"].apply(formatta_euro),
                "Vendite totali medie": df_top_strategie_custom["vendite_totali_medie"].apply(formatta_numero),
                "Stock residuo medio": df_top_strategie_custom["invenduto_medio"].apply(formatta_numero),
                "Probabilità stock residuo": df_top_strategie_custom["probabilita_stock_residuo_percentuale"].apply(formatta_percentuale)
            })

            st.dataframe(
                df_top_strategie_custom_output,
                width="stretch",
                hide_index=True
            )