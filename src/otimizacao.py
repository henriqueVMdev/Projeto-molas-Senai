"""Projeto da mola nova (AGENTS.md 8): grade em h, n, Ltot; t resolvido para K_alvo; Pareto massa × Fmax.

O problema sem restrições é degenerado (com K fixo, Fmax/m cresce quando n e Ltot caem), então cada
variável fica presa à faixa [mín, máx] das molas ensaiadas, que é a região de confiança dos modelos.
"""
import numpy as np
import pandas as pd

from src.modelos import resolver_t

BORDA = 0.05  # posição relativa na faixa abaixo/acima da qual a mola é avisada como "na borda"
CRITERIOS = ("teto_massa", "fmax_por_massa", "ponderado")


def faixas(ensaiadas):
    """[mín, máx] de cada variável no lote ensaiado. Calculado dos dados, nunca fixo no código."""
    return ensaiadas[["h_ef", "t", "n", "Ltot"]].agg(["min", "max"])


def grade(ensaiadas, modelo_K, modelos_F, modelo_massa, K_alvo, pontos=80):
    """Toda candidata bate K_alvo por construção. Uma mola nova herda o perfil de h do lote:
    h_crit/h_ef e h_mu/h_ef = mediana das molas ensaiadas (os defeitos de impressão continuam lá)."""
    limites = faixas(ensaiadas)
    candidatas = pd.MultiIndex.from_product([
        np.linspace(*limites["h_ef"], pontos),
        np.arange(limites.at["min", "n"], limites.at["max", "n"] + 1),
        np.linspace(*limites["Ltot"], pontos),
    ], names=["h_ef", "n", "Ltot"]).to_frame(index=False)
    candidatas["h_crit"] = candidatas["h_ef"] * (ensaiadas["h_crit"] / ensaiadas["h_ef"]).median()
    candidatas["h_mu"] = candidatas["h_ef"] * (ensaiadas["h_mu"] / ensaiadas["h_ef"]).median()
    candidatas["t"] = resolver_t(modelo_K, candidatas, K_alvo)
    candidatas = candidatas[candidatas["t"].between(*limites["t"])].reset_index(drop=True)
    candidatas["massa_g"] = modelo_massa["prever"](candidatas)
    for versao, modelo_F in modelos_F.items():
        candidatas[f"Fmax_{versao}"] = modelo_F["prever"](candidatas)
    return candidatas


def pareto(candidatas, coluna_fmax):
    """Ordena por massa e mantém só as molas com Fmax maior que o de todas as mais leves."""
    ordenadas = candidatas.sort_values(["massa_g", coluna_fmax], ascending=[True, False])
    melhor_anterior = ordenadas[coluna_fmax].cummax().shift(fill_value=-np.inf)
    return ordenadas[ordenadas[coluna_fmax] > melhor_anterior]


def pontuacao(fmax, massa, criterio, teto_massa, alfa):
    """Valor que o critério maximiza (maior = melhor; −inf = acima do teto de massa)."""
    if criterio == "teto_massa":
        return np.where(massa <= teto_massa, fmax, -np.inf)
    return fmax ** (1 if criterio == "fmax_por_massa" else alfa) / massa


def escolher(fronteira, coluna_fmax, criterio, teto_massa, alfa):
    nota = pd.Series(pontuacao(fronteira[coluna_fmax], fronteira["massa_g"], criterio, teto_massa, alfa),
                     index=fronteira.index)
    return None if np.isneginf(nota.max()) else fronteira.loc[nota.idxmax()]


def avisos_borda(mola, limites):
    avisos = []
    for nome in limites.columns:
        minimo, maximo = limites[nome]
        posicao = (mola[nome] - minimo) / (maximo - minimo) if maximo > minimo else 0.5
        if posicao <= BORDA or posicao >= 1 - BORDA:
            avisos.append(f"{nome} = {mola[nome]:g} na borda da faixa testada [{minimo:.2f}, {maximo:.2f}]")
    return avisos


def mesma_mola(mola_a, mola_b, tolerancia=0.05):
    return mola_a["n"] == mola_b["n"] and all(
        np.isclose(mola_a[nome], mola_b[nome], rtol=tolerancia) for nome in ("h_ef", "Ltot", "t"))


if __name__ == "__main__":
    exemplo = pd.DataFrame(dict(massa_g=[1, 2, 3, 4, 5], F=[1, 0.5, 2, 2, 3]))
    fronteira = pareto(exemplo, "F")
    assert fronteira["massa_g"].tolist() == [1, 3, 5]
    assert escolher(fronteira, "F", "teto_massa", 4, 2)["massa_g"] == 3
    assert escolher(fronteira, "F", "fmax_por_massa", 0, 2)["massa_g"] == 1
    assert escolher(fronteira, "F", "ponderado", 0, 2)["massa_g"] == 5  # 9/5 > 4/3 > 1/1
    print("otimizacao ok")
