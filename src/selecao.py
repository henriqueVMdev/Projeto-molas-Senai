"""Escolha das próximas molas a ensaiar (AGENTS.md 6): quebrar a correlação t × Ltot."""
import numpy as np
import pandas as pd

LIMITE_CORRELACAO = 0.7
ALONGAMENTOS_CARGA_MM = (10, 20, 30, 40)  # cargas = K previsto × estes alongamentos; a 5ª leva à falha
H_EVITAR = 1.8  # AGENTS.md 6: não pegar mais molas com h ~1,6


def correlacao(ensaiadas):
    """Correlação entre ln t, ln h_ef, ln n, ln Ltot no conjunto ensaiado (meta |r| < 0,7)."""
    return np.log(ensaiadas[["t", "h_ef", "n", "Ltot"]]).corr()


def pares_correlacionados(matriz):
    acima_da_diagonal = matriz.where(np.triu(np.ones(matriz.shape, bool), k=1)).stack()
    return acima_da_diagonal[acima_da_diagonal.abs() >= LIMITE_CORRELACAO]


def ganho_d_otimo(ensaiadas, candidatas):
    """Quanto cada candidata aumenta log det(XᵀX) do modelo de Fmax (ln t, ln h_crit, ln Ltot).
    Maior = informa mais sobre os expoentes (critério D-ótimo)."""
    def matriz_projeto(tabela):
        return np.column_stack([np.ones(len(tabela)), np.log(tabela[["t", "h_crit", "Ltot"]].to_numpy(float))])

    informacao_atual = matriz_projeto(ensaiadas).T @ matriz_projeto(ensaiadas)
    _, log_det_atual = np.linalg.slogdet(informacao_atual)
    ganhos = [np.linalg.slogdet(informacao_atual + np.outer(linha, linha))[1] - log_det_atual
              for linha in matriz_projeto(candidatas)]
    return pd.Series(ganhos, index=candidatas.index)


def sugerir(geometria, ensaiadas, modelo_K, modelo_F, quantidade=4):
    """Candidatas = molas medidas e não ensaiadas. Pega as de maior e menor t/Ltot (metade cada),
    evitando h ~1,6. Retorna a tabela de candidatas com K e Fmax previstos e as cargas sugeridas."""
    candidatas = geometria[~geometria["ensaiada"]].dropna(subset=["t", "h_ef", "h_crit", "Ltot", "n"]).copy()
    if candidatas.empty:
        return candidatas
    candidatas["t_sobre_Ltot"] = candidatas["t"] / candidatas["Ltot"]
    candidatas["K_previsto"] = modelo_K["prever"](candidatas)
    candidatas["Fmax_previsto"] = modelo_F["prever"](candidatas)
    candidatas["ganho_d_otimo"] = ganho_d_otimo(ensaiadas, candidatas)
    candidatas["evitar"] = candidatas["h_ef"] < H_EVITAR
    permitidas = candidatas[~candidatas["evitar"]].sort_values("t_sobre_Ltot")
    metade = quantidade // 2
    escolhidas = pd.concat([permitidas.tail(metade), permitidas.head(quantidade - metade)]).index.unique()
    candidatas["sugerida"] = candidatas.index.isin(escolhidas)
    candidatas["cargas_kgf"] = [", ".join(f"{K * alongamento:.2f}" for alongamento in ALONGAMENTOS_CARGA_MM)
                                + " → até a falha" for K in candidatas["K_previsto"]]
    return candidatas.sort_values(["sugerida", "ganho_d_otimo"], ascending=False)


if __name__ == "__main__":
    ensaiadas = pd.DataFrame(dict(t=[4, 10, 8, 5], h_crit=[2, 3, 2.5, 2.2], Ltot=[27, 50, 45, 30]))
    candidatas = pd.DataFrame(dict(t=[10, 7], h_crit=[2.5, 2.6], Ltot=[28, 40]))  # t alto + Ltot curto quebra a correlação
    ganhos = ganho_d_otimo(ensaiadas, candidatas)
    assert ganhos.iloc[0] > ganhos.iloc[1], ganhos
    print("selecao ok")
