"""Pipeline ponta a ponta: dados brutos → processamento → mola ótima + gráficos."""
import pandas as pd

import config
from src import ensaio, geometria, graficos, leitura, modelos, otimizacao, relevancia, selecao, validacao

pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 120,
              "display.float_format", "{:.4g}".format)


def secao(titulo):
    print(f"\n== {titulo} ==")


def main():
    saida = config.PASTA_SAIDA
    saida.mkdir(exist_ok=True)
    dados = leitura.carregar(config.CAMINHO_DADOS)
    dimensoes = geometria.marcar_exclusoes(dados["dimensoes"], config.EXCLUSOES)

    secao("Exclusões manuais (config.EXCLUSOES)")
    print(dimensoes.loc[dimensoes["excluida"], ["id", "grandeza", "posicao", "valor_mm", "motivo_exclusao"]]
          .to_string(index=False))

    # --- Geometria (5.1) ---
    tabela_geometria = geometria.resumo(dimensoes, config.Z_FLAG, config.H_CRIT_METODO, config.DISPERSAO_MAX)
    tabela_geometria = tabela_geometria.join(dados["molas"].set_index("id")[["n", "massa_g", "ensaiada"]])
    secao(f"Geometria por mola (h_crit = {config.H_CRIT_METODO}; normal de h assumida, não provada)")
    print(tabela_geometria.drop(columns="avisos").to_string())

    # --- Ensaio (5.2, 5.3) ---
    tabela_ensaio, rigidez_trechos = ensaio.analisar(dados["ensaio"], dados["falha"], tabela_geometria["C"],
                                                     config.CORTE_LIMITE_ELASTICO, config.TOLERANCIA_K)
    tabela_ensaio = tabela_ensaio.join(tabela_geometria["massa_g"])
    tabela_ensaio["K_por_massa"] = tabela_ensaio["K"] / tabela_ensaio["massa_g"]
    tabela_ensaio["fmax_por_massa"] = tabela_ensaio["fmax"] / tabela_ensaio["massa_g"]
    secao("Ensaio: K forçado pela origem (pontos elásticos + ponto de falha) e falha censurada")
    print(tabela_ensaio.drop(columns="avisos").sort_values("K", ascending=False).to_string())

    secao("Rigidez por trecho (kgf/mm, diagnóstico)")
    print(rigidez_trechos.pivot(index="id", columns="trecho", values="rigidez").to_string())

    # --- Validação (5.5) ---
    avisos = validacao.checar(dados, tabela_geometria, tabela_ensaio, config.DENSIDADE_PLA)
    secao("Avisos por mola (sinalizar, não esconder)")
    print(avisos.to_string(index=False))

    # --- Modelos (5.4) ---
    ensaiadas = tabela_geometria.join(tabela_ensaio[["K", "fmax", "modo"]], how="inner")
    modelos_ajustados = modelos.ajustar_todos(ensaiadas)
    resumo_modelos = modelos.tabela_resumo(modelos_ajustados)
    secao("Modelos: teórico × livre (erro LOO ≈ erro relativo típico de previsão)")
    print(resumo_modelos.to_string(index=False))

    for alvo in ("K", "Fmax", "massa_g"):
        secao(f"Resíduos por mola: {alvo} (%; previsto/medido − 1)")
        residuos = pd.DataFrame({versao: modelo["residuos"]["residuo_pct"]
                                 for (grandeza, versao), modelo in modelos_ajustados.items() if grandeza == alvo})
        print(residuos.join(ensaiadas["modo"]).to_string())

    secao("Modelos usados na otimização")
    modelos_usados = {
        "K": modelos.escolher(modelos_ajustados[("K", "teorica")], modelos_ajustados[("K", "livre")], "melhor_loo"),
        "Fmax": modelos.escolher(modelos_ajustados[("Fmax", "teorica")], modelos_ajustados[("Fmax", "livre")],
                                 config.MODELO_RESISTENCIA),
        "massa_g": modelos.escolher(modelos_ajustados[("massa_g", "empirica")], modelos_ajustados[("massa_g", "livre")],
                                    "melhor_loo"),
    }
    for alvo, modelo in modelos_usados.items():
        print(f"  {alvo}: {modelo['versao']} (erro LOO {100 * modelo['erro_loo']:.0f}%)")

    # --- Correlação e relevância (5.6, 6) ---
    secao("Correlação entre ln t, ln h_ef, ln n, ln Ltot nas molas ensaiadas (meta |r| < 0,7)")
    matriz_correlacao = selecao.correlacao(ensaiadas)
    print(matriz_correlacao.to_string())
    pares = selecao.pares_correlacionados(matriz_correlacao)
    for (variavel_a, variavel_b), r in pares.items():
        print(f"  ATENÇÃO: r(ln {variavel_a}, ln {variavel_b}) = {r:+.2f}: "
              f"a regressão não separa bem os efeitos de {variavel_a} e {variavel_b}")

    secao("Relevância: expoentes (elasticidades) ± erro-padrão")
    variaveis_correlacionadas = {variavel for par in pares.index for variavel in par}
    expoentes = relevancia.tabela_expoentes(modelos_ajustados, variaveis_correlacionadas)
    print(expoentes.to_string(index=False))

    secao("Relevância: contribuição na variação observada no lote (% por grandeza)")
    contribuicao = relevancia.contribuicao(modelos_usados, ensaiadas)
    print(contribuicao.to_string())

    secao("Relevância: tornado, efeito de ±10% em cada variável (%) na mola mediana do lote")
    tornado = relevancia.tornado(modelos_usados, ensaiadas)
    print(tornado.to_string(index=False))

    id_referencia, referencia, efeito_k_fixo = relevancia.efeito_k_fixo(modelos_usados, ensaiadas, config.K_ALVO)
    secao(f"Relevância: efeito com K fixo = {config.K_ALVO} kgf/mm, t recalculado "
          f"(referência: geometria da mola {id_referencia}, t = {referencia['t'].iat[0]:.2f} mm)")
    print(efeito_k_fixo.to_string(index=False))

    # --- Próximas molas (6) ---
    secao("Próximas molas a ensaiar")
    proximas = selecao.sugerir(tabela_geometria, ensaiadas, modelos_usados["K"], modelos_usados["Fmax"])
    if proximas.empty:
        print("  Nenhuma mola medida e não ensaiada na planilha. Meta para quebrar a correlação: "
              "t alto com Ltot curto e t baixo com Ltot longo (t/Ltot fora de 0,11–0,22).")
    else:
        print(proximas[["t", "Ltot", "h_ef", "n", "t_sobre_Ltot", "K_previsto", "Fmax_previsto", "ganho_d_otimo",
                        "evitar", "sugerida", "cargas_kgf"]].to_string())
        proximas.to_csv(saida / "proximas_molas.csv")

    # --- Otimização (8) ---
    limites = otimizacao.faixas(ensaiadas)
    secao("Espaço de projeto: faixa [mín, máx] das molas ensaiadas")
    print(limites.to_string())

    modelos_fmax = {"teorica": modelos_ajustados[("Fmax", "teorica")], "livre": modelos_ajustados[("Fmax", "livre")]}
    candidatas = otimizacao.grade(ensaiadas, modelos_usados["K"], modelos_fmax, modelos_usados["massa_g"],
                                  config.K_ALVO)
    secao(f"Molas candidatas com K = {config.K_ALVO} kgf/mm (t resolvido; fora da faixa de t descartadas)")
    print(f"  {len(candidatas)} candidatas na grade")
    if candidatas.empty:
        print("  NENHUMA candidata: o K alvo não é atingível dentro da faixa testada. "
              "Ensaiar molas mais próximas do alvo ou ampliar a faixa.")
        return

    fronteiras, escolhas = {}, []
    for versao in modelos_fmax:
        coluna = f"Fmax_{versao}"
        fronteiras[versao] = otimizacao.pareto(candidatas, coluna)
        for criterio in otimizacao.CRITERIOS:
            mola = otimizacao.escolher(fronteiras[versao], coluna, criterio, config.TETO_MASSA_G, config.ALFA)
            if mola is None:
                print(f"  [{versao}] {criterio}: nenhuma mola com m ≤ {config.TETO_MASSA_G} g")
                continue
            escolhas.append(dict(modelo_fmax=versao, criterio=criterio, **mola[["h_ef", "h_crit", "n", "Ltot", "t",
                                 "massa_g", "Fmax_teorica", "Fmax_livre"]].to_dict(),
                                 fmax_por_massa=mola[coluna] / mola["massa_g"]))
    escolhas = pd.DataFrame(escolhas)
    secao("Fronteira de Pareto: mola escolhida por critério e por modelo de resistência")
    print(f"  pontos na fronteira: " + ", ".join(f"{v} = {len(f)}" for v, f in fronteiras.items()))
    print(escolhas.to_string(index=False))

    versao_final = modelos_usados["Fmax"]["versao"]
    selecionada = (escolhas["modelo_fmax"] == versao_final) & (escolhas["criterio"] == config.CRITERIO)
    if not selecionada.any():
        print(f"\n  Critério {config.CRITERIO} sem solução com o modelo {versao_final}.")
        return
    mola_final = escolhas[selecionada].iloc[0].copy()
    mola_final[["h_ef", "h_crit", "Ltot", "t"]] = mola_final[["h_ef", "h_crit", "Ltot", "t"]].astype(float).round(2)
    mola_final["h_mu"] = mola_final["h_ef"] * (ensaiadas["h_mu"] / ensaiadas["h_ef"]).median()
    previsao = {alvo: modelos_usados[alvo]["prever"](mola_final.to_frame().T)[0] for alvo in modelos_usados}

    secao(f"MOLA PROJETADA (critério {config.CRITERIO}, Fmax {versao_final})")
    print(f"  h (espessura do fio)  = {mola_final['h_ef']:.2f} mm (h efetivo; h mínimo esperado "
          f"{mola_final['h_crit']:.2f} mm, mesmo perfil de defeitos do lote)")
    print(f"  t (altura)            = {mola_final['t']:.2f} mm")
    print(f"  n (espiras)           = {int(mola_final['n'])}")
    print(f"  Ltot (largura total)  = {mola_final['Ltot']:.2f} mm")
    for alvo, rotulo, unidade in (("K", "K previsto", "kgf/mm"), ("Fmax", "Fmax previsto", "kgf"),
                                  ("massa_g", "massa prevista", "g")):
        erro = modelos_usados[alvo]["erro_loo"]
        print(f"  {rotulo:<21} = {previsao[alvo]:.4g} {unidade} ± {100 * erro:.0f}% (erro LOO do modelo)")
    print(f"  Fmax/massa            = {previsao['Fmax'] / previsao['massa_g']:.3f} kgf/g")
    if modelos_usados["K"]["erro_loo"] > config.TOLERANCIA_K:
        print(f"  AVISO: erro do modelo de K ({100 * modelos_usados['K']['erro_loo']:.0f}%) maior que a "
              f"tolerância ({100 * config.TOLERANCIA_K:.0f}%): a mola impressa pode sair fora do alvo.")
    for aviso in otimizacao.avisos_borda(mola_final, limites):
        print(f"  AVISO: {aviso}")

    outra_versao = "livre" if versao_final == "teorica" else "teorica"
    alternativa = escolhas[(escolhas["modelo_fmax"] == outra_versao) & (escolhas["criterio"] == config.CRITERIO)]
    if len(alternativa) and not otimizacao.mesma_mola(mola_final, alternativa.iloc[0]):
        alt = alternativa.iloc[0]
        print(f"  ATENÇÃO: com o modelo de Fmax {outra_versao} a mola escolhida seria outra: h = {alt['h_ef']:.2f}, "
              f"t = {alt['t']:.2f}, n = {int(alt['n'])}, Ltot = {alt['Ltot']:.2f} mm. "
              "O modelo de resistência ainda não está definido (seção 5.4).")

    # --- Gráficos ---
    graficos.salvar_todos(saida, dados, tabela_geometria, dimensoes, tabela_ensaio, ensaiadas, modelos_ajustados,
                          modelos_usados, matriz_correlacao, tornado, efeito_k_fixo, candidatas, fronteiras,
                          escolhas, mola_final, config)
    print(f"\nGráficos e tabelas salvos em {saida}/")

    # --- Arquivos ---
    candidatas.to_csv(saida / "candidatas.csv", index=False)
    escolhas.to_csv(saida / "escolhas_pareto.csv", index=False)
    tabela_geometria.assign(avisos=tabela_geometria["avisos"].str.join("; ")).to_csv(saida / "geometria.csv")
    tabela_ensaio.assign(avisos=tabela_ensaio["avisos"].str.join("; ")).to_csv(saida / "ensaio.csv")
    avisos.to_csv(saida / "avisos.csv", index=False)
    resumo_modelos.to_csv(saida / "modelos.csv", index=False)
    expoentes.to_csv(saida / "relevancia_expoentes.csv", index=False)
    contribuicao.to_csv(saida / "relevancia_contribuicao.csv")
    tornado.to_csv(saida / "relevancia_tornado.csv", index=False)
    efeito_k_fixo.to_csv(saida / "relevancia_k_fixo.csv", index=False)


if __name__ == "__main__":
    main()
