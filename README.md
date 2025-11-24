# 📦 Monitoramento de Produtos em Estoque

Um sistema inteligente de detecção e monitoramento de produtos em estoque usando visão computacional. O projeto utiliza **YOLOv8n** para detecção em tempo real e suporta análise de imagens estáticas e vídeos.

---

## 📋 Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Treinamento do Modelo](#treinamento-do-modelo)
- [Uso dos Scripts](#uso-dos-scripts)
- [Estrutura de Arquivos](#estrutura-de-arquivos)
- [Métricas e Resultados](#métricas-e-resultados)

---

## 🎯 Visão Geral

Este projeto implementa um sistema de monitoramento de ocupação de regiões em estoque através de:

- **Detecção de Objetos**: Usando YOLOv8n para identificar produtos
- **Mapeamento Manual**: Definição de regiões de interesse via interface interativa
- **Análise de Imagens**: Validação do modelo em imagens estáticas
- **Análise em Vídeo**: Detecção em tempo real em feeds de vídeo
- **Métricas**: Avaliação de desempenho usando ferramentas nativas do YOLO

---

## 🏗️ Arquitetura do Projeto

```
Monitoramento-de-produtos-em-estoque/
├── main.py                              # Validação em imagens estáticas
├── video_detect.py                      # Detecção em tempo real (vídeo)
├── mapping_images.py                    # Mapeamento manual de regiões
├── transform_video_to_frameIMG.py      # Extração de frames de vídeos
├── metrics.py                           # Avaliação de métricas
├── train.ipynb                          # Notebook de treinamento
├── requirements.txt                     # Dependências do projeto
├── data/
│   ├── data.yaml                        # Configuração do dataset
│   ├── regions_mapping.json             # Regiões mapeadas
│   ├── imgs_test/                       # Imagens para teste
│   ├── imgs_output/                     # Saída das detecções em imagens
│   ├── imgs_mapping_example/            # Exemplos de mapeamento
│   └── metrics/                         # Resultados das métricas
└── models/
    ├── train_lote1/ ~ train_lote12/     # Modelos treinados
    │   ├── best.pt                      # Melhor modelo da época
    │   └── last.pt                      # Último modelo treinado
```

---

## 📦 Requisitos

- Python 3.8+
- CUDA 11.0+ (recomendado para GPU)
- pip ou conda

### Dependências Python

```
ultralytics>=8.0.0
torch
opencv-python
pathlib
```

---

## 🚀 Instalação

### 1. Clonar o Repositório

```bash
git clone https://github.com/pedroogaab/Monitoramento-de-produtos-em-estoque.git
cd Monitoramento-de-produtos-em-estoque
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

---

## 🎓 Treinamento do Modelo

O modelo foi treinado usando **YOLOv8n** com os seguintes parâmetros:

```bash
yolo task=detect mode=train model=yolov8n.pt data="{dataset_path}/data.yaml" epochs=80 imgsz=640
```

### Parâmetros de Treinamento

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| **Task** | detect | Tarefa de detecção de objetos |
| **Mode** | train | Modo de treinamento |
| **Model** | yolov8n.pt | Modelo nano do YOLOv8 |
| **Epochs** | 80 | Número de épocas |
| **Image Size** | 640 | Tamanho da imagem |

### Arquivos de Configuração

- **`data/data.yaml`**: Define os caminhos para train/val/test e classes
  ```yaml
  train: images/train
  val: images/val
  test: images/test
  
  nc: 1  # Número de classes
  names: ['Product']  # Nome das classes
  ```

---

## 📂 Uso dos Scripts

### 1. **main.py** - Validação em Imagens Estáticas

Processa imagens individuais e valida a detecção do modelo contra regiões mapeadas.

```bash
python main.py
```

**Funcionalidades:**
- Carrega o modelo YOLO treinado
- Lê imagens do diretório `data/imgs_test/`
- Verifica ocupação das regiões definidas em `data/regions_mapping.json`
- Salva imagens com anotações em `data/imgs_output/`
- Exibe estatísticas de ocupação

**Configuração (em main.py):**
```python
MODEL_PATH = "models/train_lote12/best.pt"
REGIONS_JSON = "data/regions_mapping.json"
IMAGES_PATH = "data/imgs_test"
OUTPUT_PATH = "data/imgs_output"
```

**Saída:**
```
Total de imagens processadas
Total de regiões analisadas
Regiões ocupadas
Regiões vazias
```

---

### 2. **video_detect.py** - Detecção em Tempo Real (Vídeo)

Processa vídeos em tempo real, detectando produtos e verificando ocupação das regiões.

```bash
python video_detect.py
```

**Funcionalidades:**
- Uso das mesmas regiões mapeadas do arquivo JSON
- Visualização com anotações
- Opção de salvar vídeo processado
- Estatísticas agregadas por frame

**Configuração (em video_detect.py):**
```python
MODEL_PATH = "models/train_lote12/best.pt" #melhor modelo
REGIONS_JSON = "data/regions_mapping.json"
VIDEO_PATH = "data/mercadinho.mp4"
OUTPUT_PATH = "data/mercadinho_output.avi"
VIDEO_KEY = "frame_mercadinho.jpg"  # Chave correspondente no JSON
```

**Características de Visualização:**
- Regiões em vermelho (ocupadas) ou verde (vazias)
- Painel informativo com estatísticas
---

### 3. **mapping_images.py** - Mapeamento Manual de Regiões

Interface interativa para mapear manualmente as regiões de interesse nas imagens.

```bash
python mapping_images.py
```

**Funcionalidades:**
- Interface gráfica interativa
- Definição de regiões por clique de 2 pontos
- Salvamento automático em JSON
- Suporte a múltiplas imagens
- Desfazer última região

**Controles:**
| Ação | Tecla |
|------|-------|
| Adicionar ponto | Clique esquerdo |
| Desfazer região | U |
| Próxima imagem | N |
| Imagem anterior | P |
| Sair e salvar | Q |

**Arquivo de Saída:**
```json
{
  "frame_mercadinho.jpg": [
    {
      "xy_min": [100, 150],
      "xy_max": [200, 250]
    },
    {
      "xy_min": [250, 150],
      "xy_max": [350, 250]
    }
  ]
}
```

---

### 4. **transform_video_to_frameIMG.py** - Extração de Frames

Extrai frames de vídeos para usá-los como imagens de teste.

```bash
python transform_video_to_frameIMG.py
```

**Funcionalidades:**
- Reprodução de vídeo frame a frame
- Captura de frames sob demanda

**Controles:**
| Ação | Tecla |
|------|-------|
| Próximo frame | ESPAÇO |
| Salvar frame | S |
| Sair | Q |

**Configuração (em transform_video_to_frameIMG.py):**
```python
video = cv2.VideoCapture("data/mercadinho.mp4")
# Frame salvo em: data/imgs_test/frame_mercadinho.jpg
```

---

### 5. **metrics.py** - Avaliação de Desempenho

Calcula métricas de desempenho do modelo usando o conjunto de teste.

```bash
python metrics.py
```

**Funcionalidades:**
- Validação no conjunto de teste
- Geração de gráficos (P-R curve, F1 curve, Confusion Matrix)
- Cálculo de mAP, Precision e Recall
- Salvamento de resultados em `data/metrics/`

**Métricas Calculadas:**

| Métrica | Descrição |
|---------|-----------|
| **mAP@0.5:0.95** | Mean Average Precision (IoU 0.5:0.95) |
| **mAP@0.5** | Mean Average Precision (IoU 0.5) |
| **mAP@0.75** | Mean Average Precision (IoU 0.75) |
| **Precision** | Taxa de verdadeiros positivos |
| **Recall** | Taxa de detecções corretas |

**Saída Esperada:**
```
📊 Métricas de Desempenho (Threshold IoU=0.5:0.95)
mAP@0.5:0.95 (Média): 0.8450
mAP@0.5: 0.9200
mAP@0.75: 0.8900
Precision: 0.9150
Recall: 0.8900

Resultados salvos em: data/metrics/
```

---

### 6. **train.ipynb** - Notebook de Treinamento

Notebook Jupyter para treinamento do modelo em Google Colab.

**Etapas:**
1. Verificar disponibilidade de GPU
2. Conectar ao Google Drive
3. Instalar ultralytics
4. Fazer verificações do ambiente
5. Executar treinamento com YOLOv8n
6. Salvar modelo treinado

**Uso:**
- Abra em [Google Colab](https://colab.research.google.com)
- Configure o caminho do dataset
- Execute as células sequencialmente

---

## 📊 Métricas e Resultados

Os modelos treinados estão armazenados em `models/train_lote1/` até `models/train_lote12/`.

Cada pasta contém:
- **best.pt**: Modelo com melhor desempenho (recomendado para uso)
- **last.pt**: Último modelo da época

### Como Usar Um Modelo Específico

Altere a variável `MODEL_PATH` em qualquer script:

```python
# Para usar train_lote12 (ultimo modelo treinado)
MODEL_PATH = "models/train_lote12/best.pt"
```

---

## 💡 Fluxo de Trabalho Típico

```
1. Extrair frames do vídeo
   └─ transform_video_to_frameIMG.py

2. Mapear regiões manualmente
   └─ mapping_images.py

3. Validar em imagens estáticas
   └─ main.py

4. Testar em tempo real com vídeo
   └─ video_detect.py

5. Avaliar métricas do modelo
   └─ metrics.py
```

---


## 📚 Referências

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [SKU110k Dataset](https://github.com/facebookresearch/SKU110K_CVPR19)
