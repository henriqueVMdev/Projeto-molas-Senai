"""Modelos de K, Fmax e massa (AGENTS.md 5.4): versão teórica × livre, LOO e resíduos.

Lei de potência y = c · Π xᵢ^eᵢ ajustada em log: ln y = ln c + Σ eᵢ ln xᵢ.
- teórica: expoentes fixos da física, só c é ajustado;
- livre: todos os expoentes ajustados.
Os ajustes usam scipy.optimize.curve_fit, que também devolve a covariância (erro-padrão) dos parâmetros.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

MARGEM_LOO = 0.8  # a livre só vence se o erro LOO dela for < 80% do da teórica

EXPOENTES_TEORICOS = {
    "K": dict(t=1, h_ef=3, n=-1, Ltot=-3),
    "Fmax": dict(t=1, h_crit=2, Ltot=-1),
}


def coeficiente_r2(medido, previsto):
    residuo = ((medido - previsto) ** 2).sum()
    total = ((medido - medido.mean()) ** 2).sum()
    return 1 - residuo / total


def _ajustar_potencia(variaveis, alvo, expoentes_fixos):
    """Ajuste em log. `expoentes_fixos` = dict (versão teórica) ou None (versão livre)."""
    log_variaveis = np.log(variaveis.to_numpy(float))
    log_alvo = np.log(alvo.to_numpy(float))
    if expoentes_fixos is not None:
        expoentes = np.array([expoentes_fixos[nome] for nome in variaveis.columns], float)
        (ln_c,), covariancia = curve_fit(lambda log_x, ln_c: ln_c + log_x @ expoentes,
                                         log_variaveis, log_alvo, p0=[0.0])
        return dict(ln_c=ln_c, expoentes=expoentes, erro_padrao=np.zeros_like(expoentes),
                    erro_padrao_ln_c=np.sqrt(covariancia[0, 0]))
    chute_inicial = np.zeros(1 + log_variaveis.shape[1])
    parametros, covariancia = curve_fit(lambda log_x, ln_c, *expoentes: ln_c + log_x @ np.array(expoentes),
                                        log_variaveis, log_alvo, p0=chute_inicial)
    erro_padrao = np.sqrt(np.diag(covariancia))
    return dict(ln_c=parametros[0], expoentes=parametros[1:], erro_padrao=erro_padrao[1:],
                erro_padrao_ln_c=erro_padrao[0])


def _prever_potencia(modelo, tabela):
    return np.exp(modelo["ln_c"] + np.log(tabela[modelo["variaveis"]].to_numpy(float)) @ modelo["expoentes"])


def _massa_empirica(tabela, c3, c4):
    """m = c3 · t·h·n·Ltot + c4 · t (volume da serpentina + blocos das pontas)."""
    volume_serpentina = tabela["t"] * tabela["h_mu"] * tabela["n"] * tabela["Ltot"]
    return np.asarray(c3 * volume_serpentina + c4 * tabela["t"], float)


def _ajustar_massa(tabela):
    (c3, c4), covariancia = curve_fit(_massa_empirica, tabela, tabela["massa_g"].to_numpy(float), p0=[1e-3, 1.0])
    return dict(coeficientes=(c3, c4), erro_padrao_coeficientes=np.sqrt(np.diag(covariancia)))


def ajustar(tabela, alvo, variaveis, versao):
    """versao: 'teorica', 'livre' ou 'empirica' (só massa). Inclui LOO e resíduos por mola."""
    tabela = tabela.dropna(subset=variaveis + [alvo])

    def ajustar_em(dados):
        if versao == "empirica":
            return _ajustar_massa(dados) | dict(variaveis=variaveis)
        expoentes_fixos = EXPOENTES_TEORICOS[alvo] if versao == "teorica" else None
        return _ajustar_potencia(dados[variaveis], dados[alvo], expoentes_fixos) | dict(variaveis=variaveis)

    def prever(modelo, dados):
        if versao == "empirica":
            return _massa_empirica(dados, *modelo["coeficientes"])
        return _prever_potencia(modelo, dados)

    modelo = ajustar_em(tabela) | dict(alvo=alvo, versao=versao, n_amostras=len(tabela))
    medido = tabela[alvo].to_numpy(float)
    previsto = prever(modelo, tabela)
    # Leave-one-out: cada mola é prevista pelo modelo ajustado sem ela
    previsto_loo = np.array([prever(ajustar_em(tabela.drop(index=id_mola)), tabela.loc[[id_mola]])[0]
                             for id_mola in tabela.index])

    modelo["R2_log"] = coeficiente_r2(np.log(medido), np.log(previsto))
    modelo["erro_loo"] = np.sqrt(np.mean(np.log(previsto_loo / medido) ** 2))  # ≈ erro relativo típico
    modelo["residuos"] = pd.DataFrame({
        "medido": medido, "previsto": previsto, "residuo_pct": 100 * (previsto / medido - 1),
        "previsto_loo": previsto_loo, "residuo_loo_pct": 100 * (previsto_loo / medido - 1),
    }, index=tabela.index)
    if versao == "teorica":
        constante_por_mola = medido / np.exp(np.log(tabela[variaveis].to_numpy(float)) @ modelo["expoentes"])
        modelo["cv_constante"] = stats.variation(constante_por_mola, ddof=1)
    modelo["prever"] = lambda dados: prever(modelo, dados)
    return modelo


def resolver_t(modelo_K, tabela, K_alvo):
    """t que faz o modelo de K (lei de potência) dar exatamente K_alvo:
    ln t = (ln K_alvo − ln c − Σ_{v≠t} e_v ln x_v) / e_t."""
    expoentes = dict(zip(modelo_K["variaveis"], modelo_K["expoentes"]))
    soma_outras = sum(expoente * np.log(np.asarray(tabela[nome], float))
                      for nome, expoente in expoentes.items() if nome != "t")
    return np.exp((np.log(K_alvo) - modelo_K["ln_c"] - soma_outras) / expoentes["t"])


def escolher(teorico, livre, metodo):
    if metodo == "teorico":
        return teorico
    if metodo == "livre":
        return livre
    return livre if livre["erro_loo"] < MARGEM_LOO * teorico["erro_loo"] else teorico


def ajustar_todos(ensaiadas):
    """ensaiadas: uma linha por mola ensaiada com t, h_ef, h_crit, h_mu, n, Ltot, K, fmax, massa_g."""
    ensaiadas = ensaiadas.rename(columns={"fmax": "Fmax"})
    return {
        ("K", "teorica"): ajustar(ensaiadas, "K", ["t", "h_ef", "n", "Ltot"], "teorica"),
        ("K", "livre"): ajustar(ensaiadas, "K", ["t", "h_ef", "n", "Ltot"], "livre"),
        ("Fmax", "teorica"): ajustar(ensaiadas, "Fmax", ["t", "h_crit", "Ltot"], "teorica"),
        ("Fmax", "livre"): ajustar(ensaiadas, "Fmax", ["t", "h_crit", "Ltot"], "livre"),
        ("Fmax", "livre_com_n"): ajustar(ensaiadas, "Fmax", ["t", "h_crit", "n", "Ltot"], "livre"),
        ("massa_g", "empirica"): ajustar(ensaiadas, "massa_g", ["t", "h_mu", "n", "Ltot"], "empirica"),
        ("massa_g", "livre"): ajustar(ensaiadas, "massa_g", ["t", "h_mu", "n", "Ltot"], "livre"),
    }


def tabela_resumo(modelos):
    """Uma linha por modelo: constante, expoentes ± erro-padrão, R², CV de c e erro LOO."""
    linhas = []
    for (alvo, versao), modelo in modelos.items():
        linha = dict(grandeza=alvo, versao=versao, amostras=modelo["n_amostras"], R2_log=modelo["R2_log"],
                     erro_loo_pct=100 * modelo["erro_loo"], CV_constante_pct=100 * modelo.get("cv_constante", np.nan))
        if "coeficientes" in modelo:
            c3, c4 = modelo["coeficientes"]
            linha["constante"] = f"c3={c3:.4g}, c4={c4:.3g}"
        else:
            linha["constante"] = f"{np.exp(modelo['ln_c']):.4g}"
            for nome, expoente, erro in zip(modelo["variaveis"], modelo["expoentes"], modelo["erro_padrao"]):
                linha[nome] = f"{expoente:+.2f}" + (f" ± {erro:.2f}" if erro > 0 else "")
        linhas.append(linha)
    return pd.DataFrame(linhas).fillna("")


if __name__ == "__main__":
    gerador = np.random.default_rng(0)
    exemplo = pd.DataFrame(dict(t=gerador.uniform(4, 10, 8), h_ef=gerador.uniform(1.5, 3.5, 8),
                                n=gerador.integers(3, 6, 8).astype(float), Ltot=gerador.uniform(26, 52, 8)))
    exemplo["K"] = 100 * exemplo["t"] * exemplo["h_ef"] ** 3 / (exemplo["n"] * exemplo["Ltot"] ** 3)
    variaveis = ["t", "h_ef", "n", "Ltot"]
    teorico, livre = ajustar(exemplo, "K", variaveis, "teorica"), ajustar(exemplo, "K", variaveis, "livre")
    assert np.isclose(np.exp(teorico["ln_c"]), 100) and teorico["erro_loo"] < 1e-6
    assert np.allclose(livre["expoentes"], [1, 3, -1, -3], atol=1e-4)
    assert np.allclose(teorico["prever"](exemplo.assign(t=resolver_t(teorico, exemplo, 0.05))), 0.05)
    print("modelos ok")
