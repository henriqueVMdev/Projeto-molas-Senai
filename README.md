# Projeto Molas Serpentina (SENAI ADS)

Projeta uma mola serpentina em PLA com rigidez alvo, a mais forte e a mais leve possível, a partir dos ensaios de tração.
Contexto completo e decisões: `AGENTS.md`.

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Parâmetros (K alvo, critério, exclusões manuais etc.): `config.py`.
Dados de entrada: `data/molas_dados_brutos.xlsx` (só dados brutos, sem fórmulas).
Saída: tabelas no terminal + gráficos PNG e CSVs em `saidas/`.

Novas molas entram só pela planilha, sem mudar código. Mola medida e não ensaiada (`ensaiada = nao`) aparece na lista de próximas a ensaiar.

## Gráficos

| Arquivo | O que mostra |
|---|---|
| `01_curvas_ensaio.png` | carga × alongamento de cada mola, reta do K e ponto de falha |
| `02_espessura_ao_longo.png` | h por posição, μ ± 2σ, h efetivo, medidas marcadas/excluídas |
| `03_previsto_vs_medido.png` | modelos teórico × livre de K, Fmax e massa |
| `04_correlacao.png` | correlação entre ln t, ln h, ln n, ln Ltot no lote |
| `05_tornado.png` | efeito de ±10% em cada variável |
| `06_efeito_k_fixo.png` | o que acontece com Fmax e massa mantendo o K alvo |
| `07_pareto.png` | **fronteira de Pareto massa × força e a mola projetada** |

## Checagens

Cada módulo de `src/` tem um self-check: `python -m src.geometria` (idem `ensaio`, `modelos`, `relevancia`, `selecao`, `otimizacao`).
