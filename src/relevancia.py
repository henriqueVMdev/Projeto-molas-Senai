"""Relevância dos parâmetros (AGENTS.md 5.6): o que mais pesa em K, Fmax e massa.

Os expoentes de uma lei de potência são elasticidades: expoente 3 em h → +10% em h dá ~+33% em K.
Para modelos que não são lei de potência (massa empírica), a elasticidade d ln y / d ln x é
calculada numericamente com scipy.differentiate.derivative.
"""
import numpy as np
import pandas as pd
from scipy.differentiate import derivative

from src.modelos import resolver_t

VARIAVEIS = ("t", "h", "n", "Ltot")
COLUNAS_H = ("h_ef", "h_crit", "h_mu")  # "h" escala as três juntas (mesmo perfil de espessura)
COLUNAS_GEOMETRIA = ["t", *COLUNAS_H, "n", "Ltot"]


def _escalar(mola, variavel, fator):
    mola = mola.copy()
    for coluna in (COLUNAS_H if variavel == "h" else (variavel,)):
        mola[coluna] = mola[coluna] * fator
    return mola


def mola_mediana(ensaiadas):
    """Geometria de referência: mediana do lote ensaiado (uma linha)."""
    return ensaiadas[COLUNAS_GEOMETRIA].median().to_frame().T


def elasticidade(modelo, mola, variavel):
    def log_saida(log_fator):
        fatores = np.exp(np.ravel(log_fator))
        variacoes = pd.concat([_escalar(mola, variavel, fator) for fator in fatores], ignore_index=True)
        return np.log(modelo["prever"](variacoes)).reshape(np.shape(log_fator))
    return float(derivative(log_saida, 0.0).df)


def tabela_expoentes(modelos, correlacionadas=()):
    """Expoente ± erro-padrão por grandeza e versão. Marca expoentes mal determinados (|e| < 2·ep)
    e os de variáveis correlacionadas entre si no lote (efeitos não separáveis)."""
    linhas = []
    for (alvo, versao), modelo in modelos.items():
        if "expoentes" not in modelo:
            continue
        for nome, expoente, erro in zip(modelo["variaveis"], modelo["expoentes"], modelo["erro_padrao"]):
            if erro == 0:
                situacao = "fixo (teoria)"
            elif abs(expoente) < 2 * erro:
                situacao = "mal determinado"
            elif nome in correlacionadas:
                situacao = "incerto (variável correlacionada)"
            else:
                situacao = "ok"
            linhas.append(dict(grandeza=alvo, versao=versao, variavel=nome, expoente=expoente,
                               erro_padrao=erro, situacao=situacao))
    return pd.DataFrame(linhas)


def contribuicao(modelos_usados, ensaiadas):
    """|elasticidade| × desvio-padrão de ln(variável) no lote: quanto cada variável explicou
    das diferenças entre as molas ensaiadas. Em % do total por grandeza."""
    referencia = mola_mediana(ensaiadas)
    desvio_log = {nome: np.log(ensaiadas["h_ef" if nome == "h" else nome]).std() for nome in VARIAVEIS}
    absoluta = pd.DataFrame({alvo: {nome: abs(elasticidade(modelo, referencia, nome)) * desvio_log[nome]
                                    for nome in VARIAVEIS}
                             for alvo, modelo in modelos_usados.items()})
    return 100 * absoluta / absoluta.sum()


def tornado(modelos_usados, ensaiadas, delta=0.10):
    """Variação % de K, Fmax e m com ±delta em cada variável, a partir da mola mediana do lote."""
    referencia = mola_mediana(ensaiadas)
    linhas = []
    for alvo, modelo in modelos_usados.items():
        valor_base = modelo["prever"](referencia)[0]
        variacao = lambda nome, fator: 100 * (modelo["prever"](_escalar(referencia, nome, fator))[0] / valor_base - 1)
        for nome in VARIAVEIS:
            linhas.append(dict(grandeza=alvo, variavel=nome,
                               efeito_reducao_pct=variacao(nome, 1 - delta),
                               efeito_aumento_pct=variacao(nome, 1 + delta)))
    return pd.DataFrame(linhas)


def efeito_k_fixo(modelos_usados, ensaiadas, K_alvo, delta=0.10):
    """Variando h, n e Ltot com t recalculado para manter K_alvo: como Fmax e m respondem.
    É isso que decide a mola ótima. (n tratado como contínuo aqui, só para sensibilidade.)
    Referência: a mola ensaiada com K mais próximo do alvo, para t ficar perto da faixa testada."""
    modelo_K, modelo_F, modelo_m = modelos_usados["K"], modelos_usados["Fmax"], modelos_usados["massa_g"]
    id_referencia = (np.log(ensaiadas["K"]) - np.log(K_alvo)).abs().idxmin()
    referencia = ensaiadas.loc[[id_referencia], COLUNAS_GEOMETRIA].reset_index(drop=True)
    referencia["t"] = resolver_t(modelo_K, referencia, K_alvo)
    fmax_base, massa_base = modelo_F["prever"](referencia)[0], modelo_m["prever"](referencia)[0]
    linhas = []
    for nome in ("h", "n", "Ltot"):
        for fator in (1 - delta, 1 + delta):
            mola = _escalar(referencia, nome, fator)
            mola["t"] = resolver_t(modelo_K, mola, K_alvo)
            fmax, massa = modelo_F["prever"](mola)[0], modelo_m["prever"](mola)[0]
            linhas.append(dict(variavel=nome, variacao_pct=100 * (fator - 1), t_recalculado_mm=mola["t"].iat[0],
                               variacao_fmax_pct=100 * (fmax / fmax_base - 1),
                               variacao_massa_pct=100 * (massa / massa_base - 1),
                               variacao_fmax_por_massa_pct=100 * ((fmax / massa) / (fmax_base / massa_base) - 1)))
    return id_referencia, referencia, pd.DataFrame(linhas)


if __name__ == "__main__":
    modelo = dict(prever=lambda mola: np.asarray(2 * mola["t"] * mola["h_ef"] ** 3, float))
    mola = pd.DataFrame(dict(t=[5.0], h_ef=[2.0], h_crit=[1.9], h_mu=[2.1], n=[4.0], Ltot=[40.0]))
    assert np.isclose(elasticidade(modelo, mola, "h"), 3) and np.isclose(elasticidade(modelo, mola, "n"), 0)
    print("relevancia ok")
