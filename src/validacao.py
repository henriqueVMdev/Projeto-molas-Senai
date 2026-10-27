"""Checagens dos dados brutos (AGENTS.md 5.5). Sinaliza, não esconde: nada é removido aqui."""
import pandas as pd

GRANDEZAS_COPIA = ("L", "Ltot", "C")  # h e t repetem por acaso na resolução de 0,01 mm


def massa_do_fio(geometria, densidade):
    """Massa só da serpentina: t · h · (2·n·Ltot) · ρ. Não inclui os blocos das pontas,
    então a massa medida deve ser sempre maior."""
    # ponytail: comprimento do fio ≈ 2·n·Ltot (ignora os arcos); trocar pelo desenvolvido se precisar
    volume_mm3 = geometria["t"] * geometria["h_mu"] * 2 * geometria["n"] * geometria["Ltot"]
    return densidade * volume_mm3 / 1000


def checar(dados, geometria, ensaio, densidade, desvio_massa=0.35):
    """Devolve a tabela longa de avisos: id, origem, aviso."""
    avisos = []

    def avisar(id_mola, origem, mensagem):
        avisos.append(dict(id=int(id_mola), origem=origem, aviso=mensagem))

    # Avisos já levantados pela geometria e pelo ensaio
    for origem, tabela in (("geometria", geometria), ("ensaio", ensaio)):
        for id_mola, mensagens in tabela["avisos"].items():
            for mensagem in mensagens:
                avisar(id_mola, origem, mensagem)

    # Massa medida × massa do fio
    razao_massa = geometria["massa_g"] / massa_do_fio(geometria, densidade)
    mediana = razao_massa.median()
    for id_mola, razao in razao_massa.items():
        if razao < 1:
            avisar(id_mola, "massa", f"massa medida menor que a do fio sozinho (razão {razao:.2f})")
        elif abs(razao / mediana - 1) > desvio_massa:
            avisar(id_mola, "massa", f"massa/massa do fio = {razao:.2f}, longe da mediana do lote ({mediana:.2f})")

    # Valores idênticos em molas diferentes
    dimensoes = dados["dimensoes"]
    candidatas = dimensoes[dimensoes["grandeza"].isin(GRANDEZAS_COPIA)]
    molas_por_valor = candidatas.groupby(["grandeza", "valor_mm"])["id"].unique()
    for (grandeza, valor), ids in molas_por_valor[molas_por_valor.str.len() > 1].items():
        for id_mola in ids:
            avisar(id_mola, "copia", f"{grandeza} = {valor:g} idêntico nas molas {sorted(ids.tolist())}: provável cópia")
    molas = dados["molas"]
    for valor, repetidas in molas[molas["massa_g"].duplicated(keep=False)].groupby("massa_g"):
        for id_mola in repetidas["id"]:
            avisar(id_mola, "copia", f"massa {valor:g} g idêntica nas molas {repetidas['id'].tolist()}")

    # Alongamento que diminui com a carga
    for id_mola, pontos in dados["ensaio"].sort_values(["id", "carga_kgf"]).groupby("id"):
        if not pontos["alongamento_mm"].is_monotonic_increasing or pontos["alongamento_mm"].duplicated().any():
            avisar(id_mola, "ensaio", "alongamento não cresce com a carga: provável erro de digitação")

    # Se L é o passo, os n passos cabem no comprimento total C
    for id_mola, linha in geometria[geometria["n"] * geometria["L"] > geometria["C"]].iterrows():
        avisar(id_mola, "geometria", f"n·L = {linha['n'] * linha['L']:.1f} > C = {linha['C']:.1f}: "
                                     "L não pode ser o passo (conferir o que foi medido)")

    # Consistência entre abas
    ids_molas = set(molas["id"])
    for aba in ("dimensoes", "ensaio", "falha"):
        for id_mola in set(dados[aba]["id"]) - ids_molas:
            avisar(id_mola, "abas", f"id presente em `{aba}` mas não em `molas`")
    for id_mola in set(molas.loc[molas["ensaiada"], "id"]) - set(dados["ensaio"]["id"]):
        avisar(id_mola, "abas", "marcada como ensaiada sem linhas em `ensaio`")
    falha = dados["falha"]
    for id_mola in falha.loc[falha["carga_min_kgf"] > falha["carga_max_kgf"], "id"]:
        avisar(id_mola, "falha", "carga_min > carga_max")

    return pd.DataFrame(avisos, columns=["id", "origem", "aviso"]).sort_values(["id", "origem"], kind="stable")
