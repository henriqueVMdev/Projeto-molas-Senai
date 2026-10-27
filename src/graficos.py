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
    figura.suptitle("Ensaio de tração: K pela origem com os pontos elásticos + ponto de falha")
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
    figura.suptitle("Espessura do fio h ao longo de cada mola (normal assumida, não provada)")
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
        eixo.set_xscale("log")
        eixo.set_yscale("log")
        eixo.set_xlim(limites)
        eixo.set_ylim(limites)
        for eixo_xy in (eixo.xaxis, eixo.yaxis):
            eixo_xy.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:g}"))
            eixo_xy.set_minor_formatter(NullFormatter())
        eixo.set_title(rotulo)
        eixo.set_xlabel("medido")
        eixo.set_ylabel("previsto")
        modos = [Line2D([], [], linestyle="", marker=MARCADOR_MODO[modo], color=TINTA_APAGADA, label=modo)
                 for modo in ensaiadas["modo"].unique()]
        eixo.legend(handles=eixo.get_legend_handles_labels()[0] + modos, fontsize=8, loc="upper left")
    figura.suptitle("Previsto × medido, escala log (rótulo = mola com resíduo > 25%)")
    _salvar(figura, pasta, "03_previsto_vs_medido.png")


def correlacao(pasta, matriz):
    mapa = LinearSegmentedColormap.from_list("divergente", [VERMELHO, CINZA_NEUTRO, AZUL])
    figura, eixo = plt.subplots(figsize=(5.2, 4.4))
    imagem = eixo.imshow(matriz.to_numpy(), cmap=mapa, vmin=-1, vmax=1)
    rotulos = [f"ln {nome}" for nome in matriz.columns]
    eixo.set_xticks(range(len(rotulos)), rotulos)
    eixo.set_yticks(range(len(rotulos)), rotulos)
    eixo.grid(False)
    for (i, j), valor in np.ndenumerate(matriz.to_numpy()):
        eixo.text(j, i, f"{valor:+.2f}", ha="center", va="center", fontsize=9,
                  color="#ffffff" if abs(valor) > 0.7 else TINTA, fontweight="bold" if abs(valor) >= 0.7 else None)
    figura.colorbar(imagem, ax=eixo, label="r")
    figura.suptitle("Correlação nas molas ensaiadas (meta |r| < 0,7)")
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
    figura.suptitle("Tornado: ±10% em cada variável, na mola mediana do lote")
    _salvar(figura, pasta, "05_tornado.png", legenda=(eixos[:1], 2))


def efeito_k_fixo(pasta, tabela, K_alvo):
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
    eixo.set_title(f"Com K fixo = {K_alvo} kgf/mm (t recalculado): o que decide a mola ótima")
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
    figura.suptitle(f"Pareto massa × força para K = {config.K_ALVO} kgf/mm "
                    f"(critério final: {config.CRITERIO}, α = {config.ALFA:g})")
    _salvar(figura, pasta, "07_pareto.png")


def salvar_todos(pasta, dados, tabela_geometria, dimensoes, tabela_ensaio, ensaiadas, modelos_ajustados,
                 modelos_usados, matriz_correlacao, tabela_tornado, tabela_k_fixo, candidatas, fronteiras,
                 escolhas, mola_final, config):
    curvas_ensaio(pasta, dados, tabela_ensaio)
    espessura_ao_longo(pasta, dimensoes, tabela_geometria, config.Z_FLAG)
    previsto_vs_medido(pasta, modelos_ajustados, ensaiadas)
    correlacao(pasta, matriz_correlacao)
    tornado(pasta, tabela_tornado)
    efeito_k_fixo(pasta, tabela_k_fixo, config.K_ALVO)
    pareto(pasta, candidatas, fronteiras, escolhas, mola_final, ensaiadas, modelos_usados["Fmax"]["versao"], config)
