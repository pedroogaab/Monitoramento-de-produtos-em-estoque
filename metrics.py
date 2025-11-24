from ultralytics import YOLO
import os
import shutil

MODELO_PATH = "models/train_lote12/best.pt"
DATA_YAML_PATH = "data/data.yaml"

RESULTS_DIR = "data/metrics"
PROJECT_DIR = os.path.dirname(RESULTS_DIR) # 'data'
NAME_DIR = os.path.basename(RESULTS_DIR)    # 'metrics'


# --- Limpeza do diretório anterior ---
if os.path.exists(RESULTS_DIR):
    try:
        shutil.rmtree(RESULTS_DIR)
    except Exception as e:
        exit(1)

model = YOLO(MODELO_PATH)

# Os resultados do val são salvos no diretório 'runs/detect/val' (ou 'valN')
# As métricas e gráficos serão salvos neste diretório.

# Argumentos-chave:
#   - data: Caminho para o arquivo .yaml.
#   - split: Define o conjunto de dados a ser usado ('test' ou 'val').
#   - plots: True para gerar os gráficos (P-R curve, F1 curve, Confusion Matrix).
#   - save_hybrid: False para garantir que as curvas de Precision, Recall e mAP
#                  não sejam agrupadas ou 'híbridas' (mantendo a separação por classe se houvesse mais de uma).
#                  Para uma única classe, a curva P-R é a da classe e a global.
#                  (O padrão do YOLO é não agrupar por padrão para o conjunto de teste,
#                   mas este parâmetro pode ser útil para controle).
#   - project: Onde salvar os resultados principais (e.g., 'runs/detect').
#   - name: Subdiretório específico para este run (e.g., 'validation_results').

results = model.val(
    data=DATA_YAML_PATH,
    split='test',
    plots=True,
    save_hybrid=False,
    project=PROJECT_DIR, # Define a pasta 'data'
    name=NAME_DIR,       # Define a subpasta 'metrics'
)


print("\n## 📊 Métricas de Desempenho (Threshold IoU=0.5:0.95)")

# Acessando as métricas do objeto 'results'
metrics = results.box
print(f"**mAP@0.5:0.95 (Média):** {metrics.maps.mean():.4f}")
print(f"**mAP@0.5:** {metrics.map50:.4f}")
print(f"**mAP@0.75:** {metrics.map75:.4f}")
# A Precision (mp) e o Recall (mr) são listas por classe. Usamos a média delas.
print(f"**Precision:** {metrics.mp.mean():.4f}")
print(f"**Recall:** {metrics.mr.mean():.4f}")

# O caminho de salvamento agora é simplesmente o RESULTS_DIR que definimos
print(f"\nOs resultados detalhados (incluindo gráficos) foram salvos em: {RESULTS_DIR}")