"""Tratamento das dimensões (AGENTS.md 5.1): estatística de h por mola e médias de t/L/Ltot/C.

A normal é ASSUMIDA para h, não provada: com 5 a 9 medidas por mola o Shapiro-Wilk tem pouco poder.
"""
import numpy as np
import pandas as pd
from scipy import stats


def marcar_exclusoes(dimensoes, exclusoes):
    """Adiciona as colunas `excluida` e `motivo_exclusao`. Nada é apagado."""
    dimensoes = dimensoes.copy()
    dimensoes["excluida"] = False
    dimensoes["motivo_exclusao"] = ""
    for id_mola, grandeza, posicao, motivo in exclusoes:
        linha = ((dimensoes["id"] == id_mola) & (dimensoes["grandeza"] == grandeza)
                 & (dimensoes["posicao"] == posicao))
        if not linha.any():
            raise ValueError(f"Exclusão sem medida correspondente: {(id_mola, grandeza, posicao)}")
        dimensoes.loc[linha, ["excluida", "motivo_exclusao"]] = [True, motivo]
    return dimensoes


def h_efetivo(espessuras):
    """Média de potência −3: os trechos flexionam em série (a flexibilidade soma 1/h³),
    então os trechos finos dominam."""
    return stats.pmean(espessuras, -3)


def medidas_h(dimensoes, z_flag):
    """Medidas de h com o z dentro da própria mola. |z| > z_flag é marcado, nunca excluído."""
    medidas = dimensoes[(dimensoes["grandeza"] == "h") & ~dimensoes["excluida"]].copy()
    medidas["z"] = medidas.groupby("id")["valor_mm"].transform(lambda h: stats.zscore(h, ddof=1))
    medidas["fora_do_z"] = medidas["z"].abs() > z_flag
    return medidas


def resumo(dimensoes, z_flag, metodo_h_crit, dispersao_max):
    """Uma linha por mola: h_mu, h_sigma, h_ef, h_min, h_crit e médias de t/L/Ltot/C, mais avisos."""
    medidas = medidas_h(dimensoes, z_flag)
    h_por_mola = medidas.groupby("id")["valor_mm"]
    tabela = h_por_mola.agg(n_h="count", h_mu="mean", h_sigma="std", h_ef=h_efetivo, h_min="min")
    tabela["pos_h_min"] = medidas.loc[h_por_mola.idxmin()].set_index("id")["posicao"]

    usar_mu_menos_2sigma = (metodo_h_crit == "mu_menos_2sigma") & tabela["h_sigma"].notna()
    tabela["h_crit"] = tabela["h_min"].where(~usar_mu_menos_2sigma, tabela["h_mu"] - 2 * tabela["h_sigma"])

    tabela["avisos"] = [[] for _ in range(len(tabela))]
    for id_mola, linha in tabela.iterrows():
        if linha["n_h"] < 3:
            linha["avisos"].append(f"só {linha['n_h']} medida(s) de h")
        if metodo_h_crit == "mu_menos_2sigma" and np.isnan(linha["h_sigma"]):
            linha["avisos"].append("σ indefinido: h_crit = h_min")
    for _, medida in medidas[medidas["fora_do_z"]].iterrows():
        lado = "baixo (possível defeito proposital)" if medida["z"] < 0 else "alto (possível erro de leitura)"
        tabela.at[medida["id"], "avisos"].append(
            f"h pos {medida['posicao']} = {medida['valor_mm']:.2f} com z = {medida['z']:+.1f}: {lado}")

    outras = dimensoes[(dimensoes["grandeza"] != "h") & ~dimensoes["excluida"]]
    estatisticas = outras.groupby(["grandeza", "id"])["valor_mm"].agg(["mean", "min", "max", np.ptp])
    for grandeza, est in estatisticas.groupby(level="grandeza"):
        est = est.droplevel("grandeza")
        tabela[grandeza] = est["mean"]
        dispersao_relativa = est["ptp"] / est["mean"]
        for id_mola, dispersao in dispersao_relativa[dispersao_relativa > dispersao_max].items():
            tabela.at[id_mola, "avisos"].append(
                f"{grandeza} disperso: {est.at[id_mola, 'min']:.2f} a {est.at[id_mola, 'max']:.2f} ({dispersao:.0%})")
    return tabela


if __name__ == "__main__":
    assert np.isclose(h_efetivo([2.0, 2.0, 2.0]), 2.0)
    assert h_efetivo([1.5, 3.0]) < np.mean([1.5, 3.0])  # o trecho fino puxa h_ef para baixo
    exemplo = pd.DataFrame(dict(id=[1] * 5, grandeza=["h"] * 4 + ["t"], posicao=[1, 2, 3, 4, 1],
                                valor_mm=[3.0, 3.1, 2.9, 9.0, 5.0], observacao=""))
    exemplo = marcar_exclusoes(exemplo, [(1, "h", 4, "teste")])
    tabela = resumo(exemplo, 2.0, "min", 0.04)
    assert np.isclose(tabela.at[1, "h_mu"], 3.0) and tabela.at[1, "n_h"] == 3 and tabela.at[1, "t"] == 5.0
    print("geometria ok")
