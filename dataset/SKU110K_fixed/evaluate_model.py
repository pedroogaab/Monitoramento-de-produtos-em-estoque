import os
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2

def load_labels(label_path):
    """Carrega os labels do arquivo txt no formato YOLO: class x_center y_center width height"""
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls, x, y, w, h = map(float, parts)
                labels.append([cls, x, y, w, h])
    return np.array(labels)

def yolo_to_xyxy(labels, img_w, img_h):
    """Converte boxes do formato YOLO (cx, cy, w, h) para (x1, y1, x2, y2)"""
    if len(labels) == 0:
        return np.empty((0, 4))
    cx, cy, w, h = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.stack([x1, y1, x2, y2], axis=1)

def compute_iou(boxA, boxB):
    """Calcula IoU entre duas bounding boxes no formato [x1, y1, x2, y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def clear_memory():
    """Libera memória do sistema e GPU se disponível"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Caminhos das pastas - ajustados para a estrutura correta
script_dir = Path(__file__).parent.absolute()
labels_dir = script_dir / 'labels' / 'val'
images_dir = script_dir / 'images' / 'val'

print(f"Diretório de imagens: {images_dir}")
print(f"Diretório de labels: {labels_dir}")

# Verifica se os diretórios existem
if not images_dir.exists():
    print(f"Erro: Diretório de imagens não encontrado: {images_dir}")
    exit(1)
if not labels_dir.exists():
    print(f"Erro: Diretório de labels não encontrado: {labels_dir}")
    exit(1)

# Carrega o modelo YOLO treinado
model_path = script_dir.parent.parent / 'detect' / 'weights' / 'best.pt'
print(f"Caminho do modelo: {model_path}")
if not model_path.exists():
    print(f"Erro: Arquivo do modelo não encontrado: {model_path}")
    exit(1)

model = YOLO(str(model_path))

# Lista todas as imagens no diretório de validação
image_paths = sorted(list(images_dir.glob('*.jpg')))  # Pega todas as imagens .jpg
if not image_paths:
    print("Nenhuma imagem encontrada no diretório:", images_dir)
    exit(1)

print(f"Encontradas {len(image_paths)} imagens para processamento")

# Inicializa variáveis de métricas
all_precisions = []
all_recalls = []
iou_threshold = 0.5  # limiar padrão para considerar acerto
valid_images = 0

try:
    # Processa as imagens em lotes pequenos
    batch_size = 5  # Reduzido para 5 imagens por lote
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(image_paths), batch_size):
        # Limpa memória no início de cada lote
        clear_memory()
        
        # Pega o próximo lote de imagens
        batch_paths = image_paths[batch_idx:batch_idx + batch_size]
        current_batch = batch_idx // batch_size + 1
        print(f"\nProcessando lote {current_batch}/{total_batches}")
        
        for image_path in batch_paths:
            try:
                # Encontra o arquivo de label correspondente
                label_file = labels_dir / f"{image_path.stem}.txt"
                
                if not label_file.exists():
                    print(f'[AVISO] Sem label para {image_path.name}')
                    continue

                gt_labels = load_labels(label_file)
                if len(gt_labels) == 0:
                    print(f'[AVISO] Label vazio em {image_path.name}')
                    continue

                # Carrega e processa a imagem
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f'[ERRO] Falha ao ler imagem: {image_path}')
                    continue

                img_h, img_w = img.shape[:2]
                gt_boxes = yolo_to_xyxy(gt_labels, img_w, img_h)

                # Faz predição com o modelo YOLO
                results = model(image_path, verbose=False)
                pred_boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.array([])

                if len(pred_boxes) == 0:
                    print(f'[INFO] Nenhuma predição em {image_path.name}')
                    all_precisions.append(0)
                    all_recalls.append(0)
                    continue

                # Calcula matches entre ground truth e predições
                matches = []
                used_pred = set()
                for i, gt_box in enumerate(gt_boxes):
                    best_iou = 0
                    best_pred = -1
                    for j, pred_box in enumerate(pred_boxes):
                        if j in used_pred:
                            continue
                        iou = compute_iou(gt_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_pred = j
                    if best_iou > iou_threshold:
                        matches.append((i, best_pred))
                        used_pred.add(best_pred)

                # Calcula métricas
                tp = len(matches)
                fp = len(pred_boxes) - tp
                fn = len(gt_boxes) - tp

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                all_precisions.append(precision)
                all_recalls.append(recall)
                valid_images += 1

                print(f'{image_path.name} → TP={tp}, FP={fp}, FN={fn}, precision={precision:.2f}, recall={recall:.2f}')

                # Libera memória após processar cada imagem
                del img, results
                clear_memory()

            except Exception as e:
                print(f'[ERRO] Erro ao processar {image_path.name}: {str(e)}')
                continue

        # Força limpeza de memória após cada lote
        clear_memory()

except KeyboardInterrupt:
    print("\nInterrompido pelo usuário.")

finally:
    # Calcula e exibe métricas finais
    if valid_images > 0:
        mean_precision = sum(all_precisions) / len(all_precisions)
        mean_recall = sum(all_recalls) / len(all_recalls)
        print(f"\nPrecisão média: {mean_precision:.4f}")
        print(f"Recall médio: {mean_recall:.4f}")
    else:
        print("\n⚠️ Nenhuma imagem válida processada. Verifique caminhos e formatos.")

    print("Avaliação concluída.")