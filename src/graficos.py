"""Gráficos PNG da apresentação. Paleta categórica fixa (no máximo 3 cores por gráfico de dispersão),
identidade nunca só pela cor (marcador + legenda), grade e eixos discretos."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VERMELHO, CINZA_NEUTRO = "#e34948", "#f0efec"
TINTA, TINTA_SECUNDARIA, TINTA_APAGADA = "#0b0b0b", "#52514e", "#898781"
GRADE, EIXO, SUPERFICIE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
MARCADOR_MODO = {"plastificacao": "o", "fratura": "^", "nao_falhou": "s", "sem_registro": "D"}
MAPA_DIVERGENTE = LinearSegmentedColormap.from_list("divergente", [VERMELHO, CINZA_NEUTRO, AZUL])

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": EIXO, "axes.labelcolor": TINTA_SECUNDARIA, "axes.titlecolor": TINTA,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRADE, "grid.linewidth": 0.6, "xtick.color": TINTA_APAGADA, "ytick.color": TINTA_APAGADA,
    "text.color": TINTA, "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
    "font.size": 10, "axes.axisbelow": True, "axes.titlesize": 11, "figure.titlesize": 13, "axes.formatter.use_locale": False,
})


def _grade_de_paineis(quantidade, colunas=4, largura=3.4, altura=2.8):
    linhas = int(np.ceil(quantidade / colunas))
    figura, eixos = plt.subplots(linhas, colunas, figsize=(largura * colunas, altura * linhas), squeeze=False)
    for eixo in eixos.flat[quantidade:]:
        eixo.set_visible(False)
    return figura, eixos.flat


def _legenda_abaixo(figura, eixos, colunas):
    """Uma legenda só para todos os painéis, abaixo da figura, sem rótulos repetidos."""
    alcas = {}
    for eixo in eixos:
        for alca, rotulo in zip(*eixo.get_legend_handles_labels()):
            alcas.setdefault(rotulo, alca)
    figura.legend(alcas.values(), alcas.keys(), loc="upper center", ncol=colunas, bbox_to_anchor=(0.5, 0.0))


def _salvar(figura, pasta, nome, legenda=None):
    figura.tight_layout()
    if legenda:
        _legenda_abaixo(figura, *legenda)
    figura.savefig(pasta / nome, dpi=150, bbox_inches="tight")
    plt.close(figura)


def curvas_ensaio(pasta, dados, tabela_ensaio):
    """Carga × alongamento por mola: pontos elásticos, ponto/intervalo de falha, reta do K."""
    falha = dados["falha"].set_index("id")
    ids = tabela_ensaio.sort_values("K", ascending=False).index
    figura, eixos = _grade_de_paineis(len(ids))
    eixos = list(eixos)
    for eixo, id_mola in zip(eixos, ids):
        pontos = dados["ensaio"][dados["ensaio"]["id"] == id_mola]
        resultado = tabela_ensaio.loc[id_mola]
        limite = falha.at[id_mola, "carga_min_kgf"] if id_mola in falha.index else np.inf
        elasticos, depois = pontos[pontos["carga_kgf"] <= limite], pontos[pontos["carga_kgf"] > limite]
        eixo.scatter(elasticos["alongamento_mm"], elasticos["carga_kgf"], color=AZUL, zorder=3,
                     edgecolor=SUPERFICIE, linewidth=1, label="regime elástico")
        eixo.scatter(depois["alongamento_mm"], depois["carga_kgf"], facecolor="none", edgecolor=TINTA_APAGADA,
                     zorder=3, label="após a falha (fora do K)")
        if not np.isnan(resultado["fmax"]):
            registro = falha.loc[id_mola]
            eixo.errorbar(resultado["alongamento_falha"], resultado["fmax"], yerr=resultado["fmax_incerteza"],
                          xerr=np.ptp(registro[["alongamento_min_mm", "alongamento_max_mm"]].astype(float)) / 2,
                          fmt=MARCADOR_MODO.get(resultado["modo"], "o"), color=LARANJA, capsize=3, zorder=4,
                          markeredgecolor=SUPERFICIE, label=f"falha ({resultado['modo']})")
        x_reta = np.array([0, np.nanmax([elasticos["alongamento_mm"].max(), resultado["alongamento_falha"]])])
        eixo.plot(x_reta, resultado["K"] * x_reta, color=AZUL, linewidth=1.5, alpha=0.7)
        eixo.set_title(f"Mola {id_mola} · K = {resultado['K']:.4f} kgf/mm")
        eixo.set_xlabel("alongamento (mm)")
        eixo.set_ylabel("carga (kgf)")
        eixo.set_xlim(left=0)
        eixo.set_ylim(bottom=0)
    figura.suptitle("Rigidez K de cada mola: reta pela origem nos pontos elásticos, até a falha")
    _salvar(figura, pasta, "01_curvas_ensaio.png", legenda=(eixos, 4))


def espessura_ao_longo(pasta, dimensoes, tabela_geometria, z_flag):
    """h por posição em cada mola: média, faixa μ ± 2σ, h efetivo, pontos fora do z e excluídos."""
    medidas = dimensoes[dimensoes["grandeza"] == "h"]
    ids = tabela_geometria.index
    figura, eixos = _grade_de_paineis(len(ids))
    eixos = list(eixos)
    for eixo, id_mola in zip(eixos, ids):
        da_mola = medidas[medidas["id"] == id_mola]
        validas, excluidas = da_mola[~da_mola["excluida"]], da_mola[da_mola["excluida"]]
        linha = tabela_geometria.loc[id_mola]
        if not np.isnan(linha["h_sigma"]):
            eixo.axhspan(linha["h_mu"] - z_flag * linha["h_sigma"], linha["h_mu"] + z_flag * linha["h_sigma"],
                         color=CINZA_NEUTRO, label=f"μ ± {z_flag:g}σ")
        eixo.axhline(linha["h_mu"], color=TINTA_SECUNDARIA, linewidth=1, label="μ")
        eixo.axhline(linha["h_ef"], color=AZUL, linewidth=1, linestyle="--", label="h efetivo (rigidez)")
        fora = (validas["valor_mm"] - linha["h_mu"]).abs() > z_flag * linha["h_sigma"]
        eixo.scatter(validas.loc[~fora, "posicao"], validas.loc[~fora, "valor_mm"], color=AZUL, zorder=3,
                     edgecolor=SUPERFICIE, label="medida")
        eixo.scatter(validas.loc[fora, "posicao"], validas.loc[fora, "valor_mm"], color=LARANJA, marker="D",
                     zorder=4, edgecolor=SUPERFICIE, label=f"|z| > {z_flag:g} (marcada, mantida)")
        eixo.scatter(excluidas["posicao"], excluidas["valor_mm"], color=TINTA_APAGADA, marker="x", zorder=4,
                     label="excluída (config)")
        for _, medida in excluidas.iterrows():
            eixo.annotate(f"{medida['valor_mm']:g} excluída", (medida["posicao"], medida["valor_mm"]),
                          textcoords="offset points", xytext=(6, -3), fontsize=8, color=TINTA_SECUNDARIA)
        if len(excluidas):  # o valor excluído fica fora da escala para não esmagar as outras medidas
            margem = 0.15
            eixo.set_ylim(validas["valor_mm"].min() - margem, validas["valor_mm"].max() + margem)
            eixo.annotate("↑ valor excluído fora da escala", (0.02, 0.92), xycoords="axes fraction",
                          fontsize=8, color=TINTA_SECUNDARIA)
        eixo.set_title(f"Mola {id_mola} · {int(linha['n_h'])} medida(s)")
        eixo.xaxis.set_major_locator(MaxNLocator(integer=True))
        eixo.set_xlim(0.5, max(da_mola["posicao"].max(), 2) + 0.5)
        eixo.set_xlabel("posição (a partir da ponta do ID)")
        eixo.set_ylabel("h (mm)")
    figura.suptitle("Espessura h ao longo de cada mola: trechos finos (defeitos) mandam na rigidez e na falha")
    _salvar(figura, pasta, "02_espessura_ao_longo.png", legenda=(eixos, 6))


def previsto_vs_medido(pasta, modelos_ajustados, ensaiadas):
    """Um painel por grandeza, log-log: as duas versões de cada modelo contra a reta y = x."""
    grandezas = [("K", "K (kgf/mm)"), ("Fmax", "Fmax (kgf)"), ("massa_g", "massa (g)")]
    figura, eixos = plt.subplots(1, 3, figsize=(14, 4.6))
    for eixo, (alvo, rotulo) in zip(eixos, grandezas):
        versoes = [(versao, modelo) for (grandeza, versao), modelo in modelos_ajustados.items()
                   if grandeza == alvo and versao != "livre_com_n"]
        extremos = []
        for cor, (versao, modelo) in zip((AZUL, LARANJA), versoes):
            residuos = modelo["residuos"]
            for modo, marcador in MARCADOR_MODO.items():
                ids_modo = residuos.index[ensaiadas.loc[residuos.index, "modo"] == modo]
                if len(ids_modo):
                    eixo.scatter(residuos.loc[ids_modo, "medido"], residuos.loc[ids_modo, "previsto"], color=cor,
                                 marker=marcador, edgecolor=SUPERFICIE, zorder=3)
            eixo.scatter([], [], color=cor, label=f"{versao} (erro LOO {100 * modelo['erro_loo']:.0f}%)")
            extremos += [residuos["medido"].min(), residuos["medido"].max(), residuos["previsto"].min(),
                         residuos["previsto"].max()]
            for id_mola, linha in residuos[residuos["residuo_pct"].abs() > 25].iterrows():
                eixo.annotate(str(id_mola), (linha["medido"], linha["previsto"]), textcoords="offset points",
                              xytext=(5, 3), fontsize=8, color=TINTA_SECUNDARIA)
        limites = [min(extremos) / 1.3, max(extremos) * 1.3]
        eixo.plot(limites, limites, color=TINTA_APAGADA, linewidth=1, label="previsto = medido")
        _escala_log(eixo)
        eixo.set_xlim(limites)
        eixo.set_ylim(limites)
        eixo.set_title(rotulo)
        eixo.set_xlabel("medido")
        eixo.set_ylabel("previsto")
        modos = [Line2D([], [], linestyle="", marker=MARCADOR_MODO[modo], color=TINTA_APAGADA, label=modo)
                 for modo in ensaiadas["modo"].unique()]
        eixo.legend(handles=eixo.get_legend_handles_labels()[0] + modos, fontsize=8, loc="upper left")
    figura.suptitle("Os modelos acertam? Previsto × medido (na diagonal = acerto; número = mola com erro > 25%)")
    _salvar(figura, pasta, "03_previsto_vs_medido.png")


def _escala_log(eixo):
    """Log-log com rótulos em número comum (2, 5, 10), sem notação 10ⁿ."""
    eixo.set_xscale("log")
    eixo.set_yscale("log")
    for eixo_xy in (eixo.xaxis, eixo.yaxis):
        eixo_xy.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:g}"))
        eixo_xy.set_minor_formatter(NullFormatter())


def correlacao(pasta, matriz):
    figura, eixo = plt.subplots(figsize=(5.2, 4.4))
    imagem = eixo.imshow(matriz.to_numpy(), cmap=MAPA_DIVERGENTE, vmin=-1, vmax=1)
    rotulos = [f"ln {nome}" for nome in matriz.columns]
    eixo.set_xticks(range(len(rotulos)), rotulos)
    eixo.set_yticks(range(len(rotulos)), rotulos)
    eixo.grid(False)
    for (i, j), valor in np.ndenumerate(matriz.to_numpy()):
        eixo.text(j, i, f"{valor:+.2f}", ha="center", va="center", fontsize=9,
                  color="#ffffff" if abs(valor) > 0.7 else TINTA, fontweight="bold" if abs(valor) >= 0.7 else None)
    figura.colorbar(imagem, ax=eixo, label="r")
    figura.suptitle("Dimensões que variaram juntas no lote\n(|r| ≥ 0,7: os efeitos delas não se separam)")
    _salvar(figura, pasta, "04_correlacao.png")


def tornado(pasta, tabela):
    grandezas = [("K", "K"), ("Fmax", "Fmax"), ("massa_g", "massa")]
    figura, eixos = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for eixo, (alvo, titulo) in zip(eixos, grandezas):
        dados = tabela[tabela["grandeza"] == alvo].iloc[::-1]
        posicoes = np.arange(len(dados))
        eixo.barh(posicoes, dados["efeito_reducao_pct"], color=LARANJA, height=0.6, label="variável −10%")
        eixo.barh(posicoes, dados["efeito_aumento_pct"], color=AZUL, height=0.6, label="variável +10%")
        eixo.axvline(0, color=EIXO, linewidth=1)
        eixo.set_yticks(posicoes, dados["variavel"])
        eixo.set_title(f"Efeito em {titulo} (%)")
        eixo.grid(axis="y", visible=False)
        for posicao, (reducao, aumento) in zip(posicoes, dados[["efeito_reducao_pct", "efeito_aumento_pct"]].to_numpy()):
            for valor in (reducao, aumento):
                if abs(valor) >= 1:
                    eixo.annotate(f"{valor:+.0f}", (valor, posicao), textcoords="offset points",
                                  xytext=(4 if valor > 0 else -4, 0), ha="left" if valor > 0 else "right",
                                  va="center", fontsize=8, color=TINTA_SECUNDARIA)
        eixo.margins(x=0.25)
    figura.suptitle("Quanto K, Fmax e massa mudam com ±10% em cada dimensão (mola mediana do lote)")
    _salvar(figura, pasta, "05_tornado.png", legenda=(eixos[:1], 2))


def efeito_k_fixo(pasta, tabela, K_alvo_texto):
    """+10% em h, n, Ltot com t recalculado para manter K: o que acontece com Fmax, massa e Fmax/m."""
    aumentos = tabela[tabela["variacao_pct"] > 0].set_index("variavel")
    series = [("variacao_fmax_pct", "Fmax", AZUL), ("variacao_massa_pct", "massa", LARANJA),
              ("variacao_fmax_por_massa_pct", "Fmax / massa", AQUA)]
    figura, eixo = plt.subplots(figsize=(7.5, 4))
    posicoes = np.arange(len(aumentos))
    altura = 0.26
    for deslocamento, (coluna, rotulo, cor) in zip((-altura, 0, altura), series):
        barras = eixo.barh(posicoes + deslocamento, aumentos[coluna], height=altura - 0.03, color=cor, label=rotulo)
        eixo.bar_label(barras, fmt="%+.0f%%", padding=3, fontsize=8, color=TINTA_SECUNDARIA)
    eixo.axvline(0, color=EIXO, linewidth=1)
    eixo.set_yticks(posicoes, [f"{nome} +10%" for nome in aumentos.index])
    eixo.grid(axis="y", visible=False)
    eixo.margins(x=0.2)
    eixo.invert_yaxis()
    eixo.set_xlabel("variação (%)")
    eixo.set_title(f"Mantendo K = {K_alvo_texto}, com t recalculado:\n"
                   "mexer em qual dimensão dá mais força com menos massa?")
    _salvar(figura, pasta, "06_efeito_k_fixo.png", legenda=([eixo], 3))


def pareto(pasta, candidatas, fronteiras, escolhas, mola_final, ensaiadas, versao_final, config):
    """Peça central: massa × Fmax de todas as candidatas com K = K_alvo, fronteiras e escolhas."""
    figura, eixos = plt.subplots(1, 2, figsize=(14, 5.4), sharey=False)
    cores = {"teorica": AZUL, "livre": LARANJA}
    marcadores = {"teto_massa": "s", "fmax_por_massa": "D", "ponderado": "*"}
    perto_do_alvo = ensaiadas[(ensaiadas["K"] / config.K_ALVO - 1).abs() <= config.TOLERANCIA_K]
    for eixo, versao in zip(eixos, fronteiras):
        coluna = f"Fmax_{versao}"
        eixo.scatter(candidatas["massa_g"], candidatas[coluna], s=4, color=EIXO, alpha=0.5, linewidth=0,
                     label=f"candidatas ({len(candidatas)})")
        fronteira = fronteiras[versao]
        eixo.plot(fronteira["massa_g"], fronteira[coluna], color=cores[versao], label="fronteira de Pareto")
        for _, escolha in escolhas[escolhas["modelo_fmax"] == versao].iterrows():
            eixo.scatter(escolha["massa_g"], escolha[coluna], marker=marcadores[escolha["criterio"]],
                         s=170 if escolha["criterio"] == "ponderado" else 80, color=cores[versao],
                         edgecolor=TINTA, linewidth=1, zorder=5, label=f"critério {escolha['criterio']}")
        if versao == versao_final:
            eixo.annotate(f"mola projetada\nh={mola_final['h_ef']:.2f} t={mola_final['t']:.2f} "
                          f"n={int(mola_final['n'])} Ltot={mola_final['Ltot']:.2f}",
                          (mola_final["massa_g"], mola_final[coluna]), textcoords="offset points",
                          xytext=(14, -30), fontsize=8.5, color=TINTA,
                          arrowprops=dict(arrowstyle="-", color=TINTA_SECUNDARIA, linewidth=0.8))
        if len(perto_do_alvo):
            eixo.scatter(perto_do_alvo["massa_g"], perto_do_alvo["fmax"], marker="o", s=60, color=AQUA,
                         edgecolor=SUPERFICIE, zorder=4, label="ensaiada com K ≈ alvo (medido)")
            for id_mola, linha in perto_do_alvo.iterrows():
                eixo.annotate(f"mola {id_mola}", (linha["massa_g"], linha["fmax"]), textcoords="offset points",
                              xytext=(6, 4), fontsize=8, color=TINTA_SECUNDARIA)
        if config.CRITERIO == "teto_massa":
            eixo.axvline(config.TETO_MASSA_G, color=TINTA_APAGADA, linestyle=":", linewidth=1, label="teto de massa")
        titulo = "teórico" if versao == "teorica" else "livre"
        destaque = " · USADO" if versao == versao_final else ""
        eixo.set_title(f"Fmax pelo modelo {titulo}{destaque}")
        eixo.set_xlabel("massa prevista (g)")
        eixo.set_ylabel("Fmax previsto (kgf)")
        eixo.legend(fontsize=8, loc="lower right")
    figura.suptitle(f"Mola projetada: a mais forte e leve com K = {config.K_ALVO_TEXTO}\n"
                    f"(ponto cinza = mola possível; linha = as melhores; critério {config.CRITERIO})")
    _salvar(figura, pasta, "07_pareto.png")


def correlacao_dimensoes(pasta, tabela):
    """Mapa do r (Pearson em log) de cada dimensão com K, Fmax e massa, com o p-valor."""
    ordem = dict(index=tabela["dimensao"].unique(), columns=tabela["grandeza"].unique())
    r = tabela.pivot(index="dimensao", columns="grandeza", values="r").reindex(**ordem)
    p = tabela.pivot(index="dimensao", columns="grandeza", values="p_valor").reindex(**ordem)
    figura, eixo = plt.subplots(figsize=(5.4, 6))
    imagem = eixo.imshow(r.to_numpy(), cmap=MAPA_DIVERGENTE, vmin=-1, vmax=1, aspect="auto")
    eixo.set_xticks(range(len(r.columns)), [f"ln {nome}" for nome in r.columns])
    eixo.set_yticks(range(len(r.index)), [f"ln {nome}" for nome in r.index])
    eixo.xaxis.tick_top()
    eixo.grid(False)
    for (i, j), valor in np.ndenumerate(r.to_numpy()):
        texto_p = f"p = {p.iat[i, j]:.2f}" if p.iat[i, j] >= 0.01 else "p < 0.01"
        eixo.text(j, i, f"{valor:+.2f}\n{texto_p}", ha="center", va="center", fontsize=8.5,
                  color="#ffffff" if abs(valor) > 0.7 else TINTA, fontweight="bold" if p.iat[i, j] < 0.05 else None)
    figura.colorbar(imagem, ax=eixo, label="r (Pearson em log)", shrink=0.8)
    figura.suptitle("Quais dimensões acompanham rigidez, força e massa?\n"
                    "(r de −1 a +1; negrito = p < 0,05, significativo)", fontsize=11)
    _salvar(figura, pasta, "08_correlacao_dimensoes.png")


def dimensoes_vs_saidas(pasta, ensaiadas, tabela):
    """Cada saída (linha) contra cada variável do modelo (coluna), log-log, com a tendência simples e o r.
    h é a versão que cada modelo usa: h_ef na rigidez e na massa, h_crit na força."""
    linhas = [("K", "K", "K (kgf/mm)", "h_ef"), ("fmax", "Fmax", "Fmax (kgf)", "h_crit"),
              ("massa_g", "massa", "massa (g)", "h_ef")]
    estatisticas = tabela.set_index(["dimensao", "grandeza"])
    figura, eixos = plt.subplots(3, 4, figsize=(13, 9.6))
    for linha_eixos, (coluna, grandeza, rotulo, coluna_h) in zip(eixos, linhas):
        for eixo, dimensao in zip(linha_eixos, ("t", coluna_h, "n", "Ltot")):
            dados = ensaiadas[[dimensao, coluna, "modo"]].dropna()
            for modo, marcador in MARCADOR_MODO.items():
                do_modo = dados[dados["modo"] == modo]
                if len(do_modo):
                    eixo.scatter(do_modo[dimensao], do_modo[coluna], color=AZUL, marker=marcador, s=48,
                                 edgecolor=SUPERFICIE, linewidth=1, zorder=3, label=modo)
            linha = estatisticas.loc[(dimensao, grandeza)]
            x_reta = np.geomspace(dados[dimensao].min(), dados[dimensao].max(), 20)
            centro_x, centro_y = np.log(dados[dimensao]).mean(), np.log(dados[coluna]).mean()
            eixo.plot(x_reta, np.exp(centro_y + linha["inclinacao"] * (np.log(x_reta) - centro_x)),
                      color=TINTA_APAGADA, linewidth=1, zorder=2, label="tendência simples (log)")
            for id_mola, ponto in dados.iterrows():
                eixo.annotate(str(id_mola), (ponto[dimensao], ponto[coluna]), textcoords="offset points",
                              xytext=(4, 3), fontsize=7, color=TINTA_SECUNDARIA)
            _escala_log(eixo)
            # faixa estreita (< 1 década): ticks lineares, senão o eixo log fica sem número
            eixo.xaxis.set_major_locator(MaxNLocator(4, integer=dimensao == "n"))
            eixo.set_title(f"r ={linha['r']:+.2f} · inclinação {linha['inclinacao']:+.2f}", fontsize=9.5,
                           fontweight="bold" if linha["p_valor"] < 0.05 else None)
            eixo.set_xlabel(dimensao if dimensao == "n" else f"{dimensao} (mm)")
            eixo.set_ylabel(rotulo)
    figura.suptitle("Cada dimensão contra rigidez, força e massa, mola a mola\n"
                    "(inclinação ≈ % de mudança no resultado para +1% na dimensão)")
    _salvar(figura, pasta, "09_dimensoes_vs_saidas.png", legenda=(list(eixos.flat), 4))


# arquivo: (o que o gráfico mostra, para que serve na decisão)
SECOES = {
    "1. Dados medidos": {
        "01_curvas_ensaio.png": (
            "Carga × alongamento de cada mola, com a reta da rigidez K e o ponto onde ela falhou.",
            "Medir a rigidez e a força de falha de cada mola. São os dados que alimentam todo o resto."),
        "02_espessura_ao_longo.png": (
            "A espessura h medida em vários pontos de cada mola, com a média e a faixa normal (±2σ).",
            "Achar os trechos finos (defeitos de impressão): eles deixam a mola mais mole e é onde ela rompe."),
    },
    "2. Correlações: o que cada dimensão faz": {
        "08_correlacao_dimensoes.png": (
            "Um número (r) por par dimensão × resultado: perto de +1 sobem juntos, perto de −1 um sobe e o outro "
            "desce, perto de 0 não há relação. Negrito = relação confiável (p < 0,05).",
            "Ver de relance quais dimensões acompanham a rigidez, a força e o peso."),
        "09_dimensoes_vs_saidas.png": (
            "Os pontos de cada mola por trás de cada correlação, com a linha de tendência.",
            "Conferir se a correlação é real ou puxada por uma mola só, e ver o tamanho do efeito (inclinação)."),
        "04_correlacao.png": (
            "Quanto as dimensões variaram juntas nas molas ensaiadas.",
            "Alertar quando duas dimensões andam juntas (|r| ≥ 0,7): aí não dá para saber qual das duas causou o "
            "efeito, e a correlação simples engana."),
    },
    "3. Modelos e peso de cada dimensão": {
        "03_previsto_vs_medido.png": (
            "O que os modelos preveem contra o que foi medido, para rigidez, força e massa.",
            "Saber se dá para confiar nos modelos: ponto na diagonal = acerto; longe = erro."),
        "05_tornado.png": (
            "Quanto rigidez, força e massa mudam ao aumentar ou diminuir 10% cada dimensão.",
            "Ranquear as dimensões: barra maior = dimensão que mais pesa naquele resultado."),
        "06_efeito_k_fixo.png": (
            "O efeito de aumentar h, n ou Ltot em 10% ajustando t para a rigidez continuar no alvo.",
            "Mostrar para onde mexer na geometria para ganhar força e perder peso sem sair do K alvo."),
    },
    "4. Resultado": {
        "07_pareto.png": (
            "Todas as molas possíveis com o K alvo (cinza), as melhores — mais força para cada massa — (linha) e a "
            "mola escolhida (estrela).",
            "A resposta do projeto: as dimensões da mola mais forte e mais leve com a rigidez pedida."),
    },
}


def pagina(pasta, mola_final, n_ensaiadas, config):
    """Uma página com todos os gráficos, na ordem da apresentação. Devolve o caminho do HTML."""
    blocos = []
    for secao, graficos in SECOES.items():
        figuras = "".join(
            f'<figure><img src="{arquivo}" alt="{mostra}"><figcaption><p><b>Mostra:</b> {mostra}</p>'
            f'<p><b>Objetivo:</b> {objetivo}</p></figcaption></figure>'
            for arquivo, (mostra, objetivo) in graficos.items() if (pasta / arquivo).exists())
        blocos.append(f"<h2>{secao}</h2>{figuras}")
    aviso = (f'<p class="aviso"><b>K alvo provisório</b> ({config.K_ALVO_PROVISORIO}). O professor ainda não '
             "informou o valor oficial: a mola projetada muda quando ele chegar.</p>"
             if config.K_ALVO_PROVISORIO else "")
    resumo = (f"K alvo = {config.K_ALVO_TEXTO} · {n_ensaiadas} molas ensaiadas · critério {config.CRITERIO}<br>"
              f"<b>Mola projetada:</b> h = {mola_final['h_ef']:.2f} mm · t = {mola_final['t']:.2f} mm · "
              f"n = {int(mola_final['n'])} · Ltot = {mola_final['Ltot']:.2f} mm")
    html = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Molas serpentina</title>
<style>
  body {{ margin: 0 auto; max-width: 1400px; padding: 24px 16px; background: {SUPERFICIE}; color: {TINTA};
         font: 15px/1.5 system-ui, sans-serif; }}
  h1 {{ margin: 0; font-size: 24px; }} .sub {{ color: {TINTA_SECUNDARIA}; margin: 4px 0 12px; }}
  .aviso {{ border-left: 4px solid {LARANJA}; background: #fdf0ea; padding: 8px 12px; margin: 0 0 8px; }}
  h2 {{ margin: 40px 0 12px; font-size: 18px; border-bottom: 1px solid {GRADE}; padding-bottom: 6px; }}
  figure {{ margin: 0 0 32px; }} img {{ max-width: 100%; height: auto; display: block; }}
  figcaption {{ font-size: 14px; margin-top: 6px; max-width: 900px; }}
  figcaption p {{ margin: 2px 0; }} figcaption b {{ color: {TINTA}; }}
</style></head><body>
<h1>Molas serpentina</h1><p class="sub">{resumo}</p>{aviso}
{"".join(blocos)}
</body></html>"""
    caminho = pasta / "graficos.html"
    caminho.write_text(html, encoding="utf-8")
    return caminho


def salvar_todos(pasta, dados, tabela_geometria, dimensoes, tabela_ensaio, ensaiadas, modelos_ajustados,
                 modelos_usados, matriz_correlacao, tabela_tornado, tabela_k_fixo, candidatas, fronteiras,
                 escolhas, mola_final, tabela_correlacao_saidas, config):
    curvas_ensaio(pasta, dados, tabela_ensaio)
    espessura_ao_longo(pasta, dimensoes, tabela_geometria, config.Z_FLAG)
    previsto_vs_medido(pasta, modelos_ajustados, ensaiadas)
    correlacao(pasta, matriz_correlacao)
    tornado(pasta, tabela_tornado)
    efeito_k_fixo(pasta, tabela_k_fixo, config.K_ALVO_TEXTO)
    pareto(pasta, candidatas, fronteiras, escolhas, mola_final, ensaiadas, modelos_usados["Fmax"]["versao"], config)
    correlacao_dimensoes(pasta, tabela_correlacao_saidas)
    dimensoes_vs_saidas(pasta, ensaiadas, tabela_correlacao_saidas)
