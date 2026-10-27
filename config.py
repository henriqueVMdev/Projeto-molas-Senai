"""Parâmetros do projeto. Único lugar para mudar valores."""
from pathlib import Path

RAIZ = Path(__file__).parent
CAMINHO_DADOS = RAIZ / "data" / "molas_dados_brutos.xlsx"
PASTA_SAIDA = RAIZ / "saidas"
ABRIR_GRAFICOS = True  # abre saidas/graficos.html no navegador ao terminar

K_ALVO = 0.1178  # kgf/mm
# De onde veio o K_ALVO enquanto o professor não informa o oficial. Trocar por None ao receber o valor.
K_ALVO_PROVISORIO = "K medido da mola 73"
K_ALVO_TEXTO = f"{K_ALVO} kgf/mm" + (f" (PROVISÓRIO: {K_ALVO_PROVISORIO})" if K_ALVO_PROVISORIO else "")
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
