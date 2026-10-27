"""Carrega as abas da planilha de dados brutos e aplica a tipagem."""
import pandas as pd

ABAS = ("molas", "dimensoes", "ensaio", "falha")

COLUNAS_NUMERICAS = {
    "molas": ["n", "massa_g"],
    "dimensoes": ["valor_mm"],
    "ensaio": ["ponto", "carga_kgf", "alongamento_mm"],
    "falha": ["carga_min_kgf", "carga_max_kgf", "alongamento_min_mm", "alongamento_max_mm"],
}


def carregar(caminho):
    planilha = pd.read_excel(caminho, sheet_name=None)
    abas_faltando = [aba for aba in ABAS if aba not in planilha]
    if abas_faltando:
        raise ValueError(f"Abas ausentes na planilha: {abas_faltando}")
    dados = {aba: planilha[aba].dropna(how="all") for aba in ABAS}  # a aba LEIA é ignorada

    for tabela in dados.values():
        tabela["id"] = tabela["id"].astype(int)
        tabela["observacao"] = tabela["observacao"].fillna("")

    # Texto digitado em coluna numérica precisa quebrar aqui, não virar NaN silencioso
    for aba, colunas in COLUNAS_NUMERICAS.items():
        dados[aba][colunas] = dados[aba][colunas].apply(pd.to_numeric, errors="raise")

    molas = dados["molas"]
    molas["ensaiada"] = molas["ensaiada"].str.strip().str.lower() == "sim"

    dimensoes = dados["dimensoes"]
    dimensoes["grandeza"] = dimensoes["grandeza"].str.strip()
    dimensoes["posicao"] = dimensoes["posicao"].astype(int)

    dados["falha"]["modo"] = dados["falha"]["modo"].str.strip()
    return dados
