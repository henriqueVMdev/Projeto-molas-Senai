# AGENTS.md — Projeto Molas Serpentina (SENAI ADS)

Contexto para agentes de código. Leia inteiro antes de escrever qualquer linha.

## 1. O que é o projeto

Projeto de processamento de dados em Python (curso de ADS, SENAI). Molas serpentina (meandro) impressas em 3D em PLA são medidas e ensaiadas em tração. O programa usa esses dados para **projetar uma mola nova** que:

1. atinja uma rigidez alvo **K_alvo** (kgf/mm), definida pelo professor e ainda não informada;
2. seja a **mais forte** possível (maior carga de falha);
3. seja a **mais leve** possível.

O programa também precisa **mostrar quais parâmetros geométricos (h, t, n, Ltot) são mais relevantes** para rigidez, resistência e massa (seção 5.6). Isso é parte da entrega, não um extra.

A resposta final **não é escolher uma mola do lote**. É gerar as dimensões de uma mola nova a partir dos modelos ajustados aos dados.

**Não há relatório.** A entrega é o próprio programa rodando: dados brutos entram, processamento, resultado (mola ótima + gráficos). O código e os gráficos são a apresentação.

Ponto importante: o professor colocou **falhas propositais** nas molas e na impressão. As dimensões variam ao longo da mesma mola e isso deve ser tratado estatisticamente (distribuição normal), não ignorado.

## 2. Decisões já tomadas (não reabrir sem motivo)

- **A planilha só armazena dados brutos. Todo cálculo é feito no Python.** A planilha não tem nenhuma fórmula, e nenhuma média é calculada fora do código.
- **Formato longo**: uma medida por linha (ver seção 4).
- **h é guardado medida a medida, com a posição.** Nunca use só a média de h (ver seção 5.1).
- **K vem de regressão linear** carga × alongamento, forçada pela origem, usando só os pontos do regime elástico mais o ponto de plastificação (seção 5.2).
- **Falha = primeira perda de elasticidade**: fratura OU plastificação, o que acontecer primeiro. Uma mola que plastificou já falhou.
- **Carga de falha é dado censurado** (intervalo carga_min–carga_max), não um valor exato.
- **Modelos físicos com poucas constantes, não machine learning.** Com 12 amostras, ML genérico decora os dados e não extrapola.
- **Espaço de projeto = faixa testada.** A mola projetada só pode ter cada variável entre o mínimo e o máximo das molas ensaiadas.
- **Resultado em forma de fronteira de Pareto** (massa × carga de falha) para o K alvo, com o critério de escolha como parâmetro.
- **K_alvo é parâmetro de entrada.** Tudo deve funcionar com um valor provisório (por exemplo, o K da mola 73) até o valor oficial chegar.

## 3. Unidades e convenções

- Comprimentos em **mm**, cargas em **kgf**, massa em **g**, K em **kgf/mm**.
- O laboratório anota carga em "kg"; tratar como kgf.
- Célula vazia na planilha = não medido (NaN). Nunca usar "null", "-" ou texto em coluna numérica.
- `id` é inteiro e idêntico em todas as abas.
- Textos de saída, nomes de colunas e gráficos em **português**.

## 4. Dados de entrada — `data/molas_dados_brutos.xlsx`

Ler com `pd.read_excel(path, sheet_name=None)`. A aba `LEIA` contém só instruções e deve ser ignorada.

| Aba | Granularidade | Colunas |
|---|---|---|
| `molas` | 1 linha por mola | `id, n, massa_g, ensaiada (sim/nao), observacao` |
| `dimensoes` | 1 linha por medida | `id, grandeza (h/t/L/Ltot/C), posicao, valor_mm, observacao` |
| `ensaio` | 1 linha por carga | `id, ponto, carga_kgf, alongamento_mm, observacao` |
| `falha` | 1 linha por mola ensaiada | `id, carga_min_kgf, carga_max_kgf, alongamento_min_mm, alongamento_max_mm, modo (fratura/plastificacao/nao_falhou), local, observacao` |

Definições geométricas (diagrama da folha do professor):
- **h**: espessura do fio, largura da fita no plano da mola. Medido várias vezes (6 a 9) ao longo da mola, de ponta a ponta.
- **t**: altura da mola, a profundidade extrudada (fora do plano). 2 medidas.
- **L**: comprimento do passo. 2 medidas. **Não entra nos modelos** (o comprimento que flexiona é Ltot); fica só como dado descritivo.
- **Ltot**: largura total. **C**: comprimento total da mola.
- **n**: número de espiras = número de arcos em cada lado da mola.
- `posicao` numerada sempre a partir da mesma ponta (o bloco com o ID), para relacionar o h mínimo com o local da falha.

O lote tem molas **não ensaiadas** também. As dimensões delas entram na aba `dimensoes` para o programa prever K e ajudar a escolher as próximas a ensaiar.

### Protocolo do ensaio
- **5 cargas por mola** (limitação do laboratório).
- Plano: 4 cargas no regime elástico, entre ~15% e ~60% do Fmax previsto, e a 5ª para levar à falha.
- Na mola 73 houve um salto de 2 → 10 kgf; por isso a falha ficou como intervalo (6–7 kgf).

## 5. Processamento

### 5.1 Tratamento de h (falhas propositais)

Por mola, a partir das medidas individuais:
- μ, σ (distribuição normal assumida; com 6–9 medidas, o Shapiro-Wilk tem pouco poder, então a normal é **assumida, não provada**, e isso deve ser dito).
- **h_ef para rigidez**: `h_ef = mean(h**-3) ** (-1/3)`. Os trechos flexionam em série e os mais finos dominam a flexibilidade.
- **h_crit para resistência**: `h_min` ou `μ − 2σ` (configurável). A mola rompe no ponto mais fraco.
- Pontos com |z| > 2 são **marcados e exibidos, nunca excluídos automaticamente**. Um h baixo pode ser o defeito proposital (informação). Um h alto isolado é mais provavelmente erro de leitura.
- Para t, L, Ltot, C: média das medidas, com aviso quando a dispersão for grande (ex.: t da mola 60 = 10,56 e 10,00).

### 5.2 Rigidez K e limite elástico

1. Pontos elásticos = pontos da aba `ensaio` com carga ≤ `carga_min_kgf` da aba `falha`, **mais o ponto de plastificação** (carga e alongamento da aba `falha`; se for intervalo, usar o ponto médio). Pontos com carga acima da falha ficam fora (foram medidos com a mola já plastificada).
2. K = regressão forçada pela origem: `K = Σ(x·F) / Σ(x²)`. Reportar também `scipy.stats.linregress` (inclinação, intercepto, R²) como diagnóstico.
3. Rigidez por trecho (ΔF/Δx entre pontos consecutivos) é exibida como diagnóstico. Se não houver linha na aba `falha`, usar como fallback o corte de **20%** (`CORTE_LIMITE_ELASTICO`): o primeiro trecho cuja rigidez cai mais que isso em relação ao primeiro trecho encerra a região elástica.
4. R² como diagnóstico: ≥ 0,98 confiável; 0,90–0,98 suspeito; < 0,90 K sem sentido (cortes de referência, não normativos).
5. Mínimo de 2 pontos; com menos, K = NaN e aviso. Com exatamente 2 (caso da 65), calcular mas marcar como pouco confiável.

### 5.3 Resistência

- Fmax = intervalo [carga_min, carga_max]; usar o ponto médio ± metade da largura como incerteza.
- Ductilidade (%) = alongamento na falha / C0.
- Checagem útil: K × alongamento na falha deve cair dentro do intervalo de Fmax se a mola foi elástica até falhar. A 73 bate: 0,1166 × 55 ≈ 6,4 kgf, dentro de 6–7.

### 5.4 Modelos

Todos ajustados por regressão em log (`numpy.linalg.lstsq` ou `scipy.optimize.curve_fit`). Para cada modelo, implementar **duas versões** e comparar:
- **teórica**: expoentes fixos da física, só a constante é ajustada;
- **livre**: todos os expoentes ajustados aos dados.

Comparar por **validação cruzada leave-one-out** (erro de previsão de cada mola com o modelo ajustado sem ela). Com 8–12 amostras, a versão livre tende a superajustar; só usar a livre na otimização se o erro LOO dela for claramente menor.

**Rigidez** — `K = c1 · t^a1 · h_ef^a2 · n^a3 · Ltot^a4`
- Teoria (viga): a = (1, 3, −1, −3). O comprimento que flexiona é o braço (~Ltot), não o passo L: com L, a previsão errava ~10×.
- Com as 8 molas atuais: c1 ≈ 105 (CV 10%); livre dá (1,02; 2,84; −0,85; −2,65), R² 0,999. **Modelo validado.**

**Resistência** — `Fmax = c2 · t^b1 · h_crit^b2 · Ltot^b3` (testar também com n)
- Teoria: b = (1, 2, −1). Com as 8 molas: CV 30% na versão teórica; livre dá (1,68; 2,27; −2,81), R² 0,93.
- **Não confiável ainda**: ln t e ln Ltot têm correlação 0,78 no conjunto ensaiado, então os expoentes de t e Ltot não estão identificados. Reportar a incerteza dos expoentes (erro-padrão) e avisar.
- Dois modos de falha (fratura na 12; plastificação nas outras). Com poucos dados, um modelo só; registrar o modo no gráfico (marcadores diferentes).
- Usar o ponto médio do intervalo de falha; a largura do intervalo entra como incerteza.

**Massa** — empírico: `m = c3 · t · h_med · n · Ltot + c4 · t`
- O primeiro termo é o volume aproximado da serpentina; o segundo representa os blocos das pontas (altura = t).
- Com as 8 molas: c3 ≈ 0,0020, c4 ≈ 0,90, R² 0,97. A 12 (a mais leve, 4,5 g) erra +47%; as demais ficam em ±10%. Como a mola ótima tende a ficar na região da 12, conferir a massa dela e reportar esse resíduo.
- Versão alternativa a testar: comprimento desenvolvido do fio a partir de n, Ltot e do raio dos arcos.

Reportar resíduos por mola em todos os modelos. Resíduo alto = mola suspeita (ver seção 7).

### 5.5 Validação de dados (rodar antes de tudo)

- Massa medida × massa estimada pela geometria (densidade do PLA ~1,24 g/cm³). Sinalizar desvios grandes.
- Valores idênticos em molas diferentes (ex.: C da 71 e da 36) → provável cópia.
- Dispersão grande entre medidas repetidas da mesma grandeza.
- Alongamento que diminui com o aumento da carga → erro de digitação.
- Exclusões manuais (ex.: h = 6,25 da 60) ficam numa lista no `config.py`, cada uma com o motivo, e aparecem na saída.
- Saída: uma tabela de avisos por mola. **Sinalizar, não esconder.** Uma mola suspeita pode entrar no ajuste com peso menor ou ficar de fora, sempre com o motivo explícito.

### 5.6 Relevância dos parâmetros (entrega obrigatória)

Objetivo: responder "o que mais pesa para fazer a mola com rigidez X, mais forte e mais leve".
- **Os expoentes ajustados são elasticidades**: um expoente 3 em h significa que +10% em h dá ~+33% em K. Mostrar uma tabela de expoentes por grandeza (K, Fmax, m) com erro-padrão.
- **Contribuição na variação observada**: |expoente| × desvio-padrão de ln(variável) no lote. Mostra quais variáveis de fato explicaram as diferenças entre as molas ensaiadas.
- **Gráfico tornado**: efeito de ±10% em cada variável sobre K, Fmax e m.
- **Efeito com K fixo**: como t é recalculado para manter K_alvo, mostrar como Fmax e m mudam ao variar h, n e Ltot sob essa restrição. É isso que decide a mola ótima.
- Deixar explícito quando um expoente não está bem determinado (erro-padrão alto, variáveis correlacionadas).

## 6. Escolha das próximas molas a ensaiar

Já ensaiadas 8 de 12. Para as 4 restantes, o critério combinado com o grupo (lista prática levada ao laboratório):
- Calcular t ÷ Ltot de cada mola disponível. Nas 8 ensaiadas, essa razão ficou entre 0,11 e 0,22.
- Pegar 2 com razão > 0,25 (ou a maior disponível) e 2 com razão < 0,10 (ou a menor), para quebrar a correlação t × Ltot.
- Se todas estiverem entre 0,15 e 0,20: 2 repetições da geometria da 12 e 2 molas finas e curtas (h 2–2,5, menor Ltot).
- Não pegar mais molas com h ~1,6.
- Cargas: K previsto ≈ 105·t·h³/(n·Ltot³); aplicar K×10, ×20, ×30, ×40 mm e depois até a falha (quase todas falharam com 45–60 mm).

O programa deve calcular e exibir a correlação entre ln t, ln h_ef, ln n, ln Ltot no conjunto ensaiado (meta |r| < 0,7).

## 7. Estado atual dos dados

Fonte única das dimensões: `data/molas_dados_brutos.xlsx`. Molas medidas: **56, 73, 71, 60, 12, 36, 80, 65**. Ensaiadas: **todas as 8**. Faltam 4 para fechar 12.

Valores suspeitos marcados em laranja na aba `dimensoes` (não corrigidos):
- 60, h posição 1 = 6,25 (provável 3,25).
- 71, C = 260,1 (a foto da folha mostrava 202,2) e 36, C = 260,10: valores idênticos, provável cópia.
- 36 tem só 1 medida de h.
- t com dispersão alta: 71 (4,67 / 4,06), 12 (4,79 / 4,15), 60 (10,56 / 10,06). L da 71 (13,17 / 15,21).

### Resultados já calculados (8 molas ensaiadas)

K forçado pela origem, com os pontos até a plastificação (incluindo o ponto de plastificação). No cálculo de h da 60, o 6,25 foi excluído (provável digitação).

| ID | K (kgf/mm) | R² | Falha (kgf) | Modo | K/m | Falha/m (kgf/g) |
|---|---|---|---|---|---|---|
| 73 | 0,1178 | 1,000 | 6–7 | plastificação | 0,0083 | 0,46 |
| 60 | 0,0864 | 0,9996 | 4,76 | plastificação | 0,0040 | 0,22 |
| 36 | 0,0664 | 0,9998 | 4,05 | plastificação | 0,0027 | 0,16 |
| 12 | 0,0480 | 0,998 | 2,275 | fratura | 0,0107 | **0,51** |
| 71 | 0,0465 | 1,000 | 2,108 | plastificação | 0,0053 | 0,24 |
| 80 | 0,0384 | 0,9997 | 2,0 | plastificação | 0,0026 | 0,13 |
| 65 | 0,0104 | 2 pontos | 0,30–0,33 | plastificação | 0,0009 | 0,026 |
| 56 | 0,0075 | 0,9995 | 1,1 | plastificação | 0,0004 | 0,058 |

**Rigidez — modelo validado.** c1 entre 92 e 125 (mediana ~105, CV 10%). Regressão com expoentes livres em log (R² 0,999):

| | t | h_ef | n | Ltot |
|---|---|---|---|---|
| Teoria (viga) | 1 | 3 | −1 | −3 |
| Ajustado | 1,02 | 2,84 | −0,85 | −2,65 |

**Resistência — modelo ainda indefinido.** Fmax = c2·t·h_min²/Ltot tem CV 30% (20% sem a 65). Expoentes livres: t 1,68, h_min 2,27, Ltot −2,81 (R² 0,93); teoria seria 1, 2, −1. O desvio em t e Ltot provavelmente vem da **correlação r = 0,78 entre ln t e ln Ltot** no conjunto ensaiado: as molas altas (t ~10) são as largas (Ltot ~48–52) e vice-versa. Com esses dados, a regressão não consegue separar os dois efeitos.

Notas:
- 65: só 2 pontos (um deles é o de plastificação); h disperso, h_min 1,54 — candidata a defeito proposital; é a que mais destoa no modelo de resistência.
- 73: plastificou entre 6 e 7 kgf com 55 mm (intervalo por leitura visual; o programa da máquina registra o valor exato). Ponto de 10 kgf medido já plastificada, fora do K.
- 56 (1,5 kgf) e 80 (2,5 kgf / 74,93 mm): pontos posteriores à plastificação, fora do K.
- 12: estourou (fratura) com 2,275 kgf e 48,95 mm.

### Próximas 4 molas (para fechar 12)
Prioridade é **quebrar a correlação t × Ltot**: molas de t alto com Ltot curto, e de t baixo com Ltot longo. Isso define os expoentes do modelo de resistência, que é o que decide "mais forte".

## 8. Otimização e resultado final

O problema **sem restrições é degenerado**: com K fixo, Fmax/m cresce quando n e Ltot diminuem, então o otimizador levaria a n = 1 e Ltot → 0 (uma mola minúscula, sem curso). As restrições usadas são:

1. Cada variável dentro da faixa [mín, máx] das molas ensaiadas (região de confiança do modelo). Com as 8 atuais: h_ef ~1,6–3,4; t ~4,4–10,3; n 3–5; Ltot ~26,5–51,7. Calcular sempre a partir dos dados, não fixar no código.
2. n inteiro ≥ 1; h ≥ menor h imprimível (referência: menor h do lote).
3. Geometria compatível com a fixação da máquina (blocos das pontas, comprimento total).

Algoritmo:
1. Grade sobre h, n, Ltot. Para cada ponto, resolver t exatamente a partir do K alvo: `t = K_alvo · n · Ltot³ / (c1 · h³)`. Descartar se t sair da faixa. Toda mola que sobra bate o alvo por construção.
2. Calcular massa e Fmax de cada candidata pelos modelos.
3. **Fronteira de Pareto**: ordenar por massa e manter só as molas com Fmax maior que o de todas as mais leves.
4. Aplicar o critério escolhido para destacar a mola final.
5. Reportar a previsão com incerteza (a partir da dispersão das repetições e dos resíduos) e avisar se a mola estiver perto da borda da faixa testada.
6. Enquanto o modelo de resistência não estiver confiável (seção 5.4), mostrar o resultado com as duas versões (teórica e livre). Se elas apontarem molas diferentes, isso deve aparecer na saída.

Critério de escolha (parâmetro; ainda sendo confirmado com o professor — a tendência é a força ter mais peso):
- `teto_massa`: a mola mais forte com m ≤ X g;
- `fmax_por_massa`: maximiza Fmax/m;
- `ponderado`: maximiza Fmax^α / m, com α > 1 configurável.

O gráfico de Pareto, com os pontos de cada critério marcados, é a peça central da apresentação.

## 9. Parâmetros configuráveis (um único lugar, ex.: `config.py`)

- `K_ALVO` (kgf/mm) — provisório até o professor informar
- `TOLERANCIA_K` (padrão 0,10)
- `CORTE_LIMITE_ELASTICO` (padrão 0,20; fallback quando não há linha em `falha`)
- `H_CRIT_METODO` (`"min"` ou `"mu_menos_2sigma"`)
- `Z_FLAG` (padrão 2,0)
- `CRITERIO` (`"teto_massa"`, `"fmax_por_massa"`, `"ponderado"`), `TETO_MASSA_G`, `ALFA`
- `DENSIDADE_PLA` (padrão 1,24 g/cm³)
- `MODELO_RESISTENCIA` (`"teorico"`, `"livre"`, `"melhor_loo"`)
- `EXCLUSOES`: lista de (id, grandeza, posicao, motivo) para valores descartados manualmente
- `CAMINHO_DADOS`, `PASTA_SAIDA`

## 10. Estrutura sugerida

```
molas/
  data/molas_dados_brutos.xlsx
  config.py
  src/
    leitura.py        # carrega as abas, tipagem, junta por id
    validacao.py      # checagens da seção 5.5 → tabela de avisos
    geometria.py      # μ, σ, h_ef, h_crit, z-flags, médias de t/L/Ltot/C
    ensaio.py         # rigidez por trecho, limite elástico, K, R², falha censurada
    modelos.py        # ajuste teórico e livre de K, Fmax e massa; LOO; resíduos
    relevancia.py     # elasticidades, contribuição, tornado, efeito com K fixo
    selecao.py        # correlação, D-ótimo, sugestão de cargas para as próximas molas
    otimizacao.py     # grade, resolução de t, Pareto, critério
    graficos.py       # carga×alongamento, distribuição de h, previsto×medido, Pareto
  main.py             # pipeline ponta a ponta, imprime resultado e salva gráficos
  saidas/             # gráficos PNG e tabelas CSV gerados
  requirements.txt
```

Stack: **pandas, numpy, scipy, matplotlib**. Nada além disso sem necessidade clara.

O pipeline deve rodar de ponta a ponta **com as 8 molas atuais** e melhorar sozinho conforme novas linhas entram na planilha, sem mudança de código.

## 11. Em aberto

- **K_alvo**: o professor ainda não informou.
- **Critério força × massa**: confirmar com o professor se há teto de massa ou se vale a relação força/massa.
- Conferir os valores marcados em laranja na aba `dimensoes` (seção 7).
- Ensaiar as 4 molas restantes para completar 12 (seção 6).
- Modelo de resistência: expoentes de t e Ltot indefinidos até as próximas molas.
- Massa da 12: resíduo de +47% no modelo de massa; conferir.
- Se a máquina exportar a curva completa força × deslocamento (CSV/TXT), usar no lugar dos 5 pontos.
