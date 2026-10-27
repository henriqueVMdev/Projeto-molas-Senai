"""Rigidez K, limite elástico e falha censurada (AGENTS.md 5.2 e 5.3)."""
import numpy as np
import pandas as pd
from scipy import stats


def rigidez_pela_origem(alongamento, carga):
    """Mínimos quadrados forçado pela origem (carga = K · alongamento, sem intercepto)."""
    coeficientes, *_ = np.linalg.lstsq(np.asarray(alongamento, float)[:, None], np.asarray(carga, float))
    return coeficientes[0]


def rigidez_por_trecho(alongamento, carga):
    """ΔF/Δx entre pontos consecutivos, começando na origem."""
    return np.diff(np.r_[0, carga]) / np.diff(np.r_[0, alongamento])


def pontos_ate_queda(alongamento, carga, corte):
    """Fallback sem linha em `falha`: a região elástica termina antes do primeiro trecho
    cuja rigidez cai mais que `corte` em relação ao primeiro trecho. Devolve quantos pontos ficam."""
    rigidez = rigidez_por_trecho(alongamento, carga)
    trechos_com_queda = np.flatnonzero(rigidez < rigidez[0] * (1 - corte))
    return trechos_com_queda[0] if len(trechos_com_queda) else len(alongamento)


def classificar_r2(r2, n_pontos):
    if n_pontos == 2:
        return "pouco confiável (2 pontos)"
    if r2 >= 0.98:
        return "confiável"
    return "suspeito" if r2 >= 0.90 else "sem sentido"


def analisar(ensaio, falha, comprimento_C, corte, tolerancia):
    """Uma linha por mola ensaiada. `comprimento_C` é a série de C0 por id (para a ductilidade)."""
    falha = falha.set_index("id")
    linhas, trechos = [], []
    for id_mola, pontos in ensaio.sort_values(["id", "ponto"]).groupby("id"):
        alongamento, carga = pontos["alongamento_mm"].to_numpy(), pontos["carga_kgf"].to_numpy()
        avisos = []
        resultado = dict(id=id_mola, modo="sem_registro", fmax=np.nan, fmax_incerteza=np.nan,
                         alongamento_falha=np.nan)

        for i, rigidez in enumerate(rigidez_por_trecho(alongamento, carga)):
            trechos.append(dict(id=id_mola, trecho=f"{i}-{i + 1}", rigidez=rigidez))

        if id_mola in falha.index:
            registro_falha = falha.loc[id_mola]
            resultado["modo"] = registro_falha["modo"]
            if registro_falha["modo"] == "nao_falhou":
                alongamento_elastico, carga_elastica = alongamento, carga
            else:
                # Pontos acima da falha foram medidos com a mola já plastificada.
                antes_da_falha = carga <= registro_falha["carga_min_kgf"]
                faixa_carga = registro_falha[["carga_min_kgf", "carga_max_kgf"]]
                resultado["fmax"] = faixa_carga.mean()
                resultado["fmax_incerteza"] = np.ptp(faixa_carga) / 2
                resultado["alongamento_falha"] = registro_falha[["alongamento_min_mm", "alongamento_max_mm"]].mean()
                alongamento_elastico = np.r_[alongamento[antes_da_falha], resultado["alongamento_falha"]]
                carga_elastica = np.r_[carga[antes_da_falha], resultado["fmax"]]
                if (~antes_da_falha).any():
                    avisos.append(f"{(~antes_da_falha).sum()} ponto(s) acima da falha fora do K")
        else:
            pontos_elasticos = pontos_ate_queda(alongamento, carga, corte)
            alongamento_elastico, carga_elastica = alongamento[:pontos_elasticos], carga[:pontos_elasticos]
            avisos.append(f"sem linha em `falha`: limite elástico pelo corte de {corte:.0%} na rigidez por trecho")

        resultado["n_pontos_K"] = len(alongamento_elastico)
        if len(alongamento_elastico) < 2:
            resultado["K"] = np.nan
            avisos.append("menos de 2 pontos elásticos: K indefinido")
        else:
            resultado["K"] = rigidez_pela_origem(alongamento_elastico, carga_elastica)
            regressao = stats.linregress(alongamento_elastico, carga_elastica)
            resultado.update(inclinacao_linregress=regressao.slope, intercepto=regressao.intercept,
                             R2=regressao.rvalue ** 2)
            resultado["confianca_K"] = classificar_r2(resultado["R2"], len(alongamento_elastico))
            if resultado["confianca_K"] != "confiável":
                avisos.append(f"K {resultado['confianca_K']}")

        if not np.isnan(resultado["fmax"]):
            resultado["ductilidade_pct"] = 100 * resultado["alongamento_falha"] / comprimento_C.get(id_mola, np.nan)
            # Se a mola foi elástica até falhar, K·x_falha cai no intervalo de Fmax.
            carga_pelo_K = resultado["K"] * resultado["alongamento_falha"]
            resultado["K_vezes_alongamento_falha"] = carga_pelo_K
            minimo, maximo = registro_falha["carga_min_kgf"], registro_falha["carga_max_kgf"]
            if not minimo * (1 - tolerancia) <= carga_pelo_K <= maximo * (1 + tolerancia):
                avisos.append(f"K·x_falha = {carga_pelo_K:.2f} kgf fora do intervalo de falha "
                              f"{minimo:g}–{maximo:g} (±{tolerancia:.0%})")
        resultado["avisos"] = avisos
        linhas.append(resultado)
    return pd.DataFrame(linhas).set_index("id"), pd.DataFrame(trechos)


if __name__ == "__main__":
    assert np.isclose(rigidez_pela_origem([1, 2, 3], [2, 4, 6]), 2.0)
    # Mola 12 da planilha: 4 pontos + ponto de fratura → K ≈ 0,0480
    ensaio_12 = pd.DataFrame(dict(id=12, ponto=[1, 2, 3, 4], carga_kgf=[0.25, 0.5, 1.0, 2.0],
                                  alongamento_mm=[4.68, 9.24, 19.56, 40.74]))
    falha_12 = pd.DataFrame([dict(id=12, carga_min_kgf=2.275, carga_max_kgf=2.275, alongamento_min_mm=48.95,
                                  alongamento_max_mm=48.95, modo="fratura")])
    tabela, _ = analisar(ensaio_12, falha_12, pd.Series({12: 138.47}), 0.2, 0.1)
    assert abs(tabela.at[12, "K"] - 0.0480) < 5e-4, tabela.at[12, "K"]
    # Fallback: rigidez cai 50% no 3º trecho → só 2 pontos ficam
    assert pontos_ate_queda([1, 2, 4], [1, 2, 3], 0.2) == 2
    print("ensaio ok")
