"""Parâmetros do projeto. Único lugar para mudar valores."""
from pathlib import Path

RAIZ = Path(__file__).parent
CAMINHO_DADOS = RAIZ / "data" / "molas_dados_brutos.xlsx"
PASTA_SAIDA = RAIZ / "saidas"

# Provisório: K da mola 73. Trocar pelo valor do professor.
K_ALVO = 0.1178  # kgf/mm
TOLERANCIA_K = 0.10

CORTE_LIMITE_ELASTICO = 0.20  # fallback quando a mola não tem linha em `falha`

H_CRIT_METODO = "min"  # "min" ou "mu_menos_2sigma"
Z_FLAG = 2.0
DISPERSAO_MAX = 0.04  # (máx − mín)/média acima disso gera aviso (t, L, Ltot, C)

CRITERIO = "ponderado"  # "teto_massa", "fmax_por_massa", "ponderado"
TETO_MASSA_G = 10.0
ALFA = 2.0

DENSIDADE_PLA = 1.24  # g/cm³

MODELO_RESISTENCIA = "melhor_loo"  # "teorico", "livre", "melhor_loo"

# (id, grandeza, posicao, motivo): medidas descartadas à mão
EXCLUSOES = [
    (60, "h", 1, "6,25 é quase o dobro das outras medidas; provável digitação de 3,25"),
]
