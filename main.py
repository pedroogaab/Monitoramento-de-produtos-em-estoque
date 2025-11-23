import torch
from ultralytics import YOLO
import cv2
import json
import os
import numpy as np
from pathlib import Path

class RegionOccupancyDetector:
    def __init__(self, model_path, regions_json, images_path, output_path="output_detections"):
        """
        Inicializa o detector de ocupação
        
        Args:
            model_path: Caminho para o modelo YOLO
            regions_json: Caminho para o JSON com regiões mapeadas
            images_path: Caminho para as imagens de teste
            output_path: Caminho para salvar as imagens com detecções
        """
        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo atual: {self.device}")
        
        # Carregar modelo
        print(f"Carregando modelo de {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Carregar regiões mapeadas
        print(f"Carregando regiões de {regions_json}...")
        with open(regions_json, 'r') as f:
            self.regions_map = json.load(f)
        
        self.images_path = images_path
        self.output_path = output_path
        
        # Criar diretório de saída
        os.makedirs(output_path, exist_ok=True)
        
        # Estatísticas
        self.stats = {
            "total_images": 0,
            "total_regions": 0,
            "occupied_regions": 0,
            "empty_regions": 0
        }
    
    def get_box_center(self, box):
        """
        Calcula o ponto central de uma box
        
        Args:
            box: [x1, y1, x2, y2] ou {"xy_min": [x, y], "xy_max": [x, y]}
        
        Returns:
            (center_x, center_y)
        """
        if isinstance(box, dict):
            x1, y1 = box["xy_min"]
            x2, y2 = box["xy_max"]
        else:
            x1, y1, x2, y2 = box
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return int(center_x), int(center_y)
    
    def is_point_in_box(self, point, box):
        """
        Verifica se um ponto está dentro de uma box
        
        Args:
            point: (x, y)
            box: {"xy_min": [x, y], "xy_max": [x, y]}
        
        Returns:
            bool
        """
        px, py = point
        x1, y1 = box["xy_min"]
        x2, y2 = box["xy_max"]
        
        return x1 <= px <= x2 and y1 <= py <= y2
    
    def check_region_occupancy(self, manual_region, model_detections):
        """
        Verifica se uma região manual está ocupada por alguma detecção do modelo
        
        Args:
            manual_region: {"xy_min": [x, y], "xy_max": [x, y]}
            model_detections: Lista de detecções do modelo
        
        Returns:
            bool: True se ocupada, False se vazia
        """
        for detection in model_detections:
            # Obter coordenadas da box do modelo
            x1, y1, x2, y2 = detection['box']
            model_box = [x1, y1, x2, y2]
            
            # Calcular ponto central da detecção do modelo
            center = self.get_box_center(model_box)
            
            # Verificar se o centro está dentro da região manual
            if self.is_point_in_box(center, manual_region):
                return True
        
        return False
    
    def draw_region(self, image, region, label, color):
        """
        Desenha uma região na imagem com rótulo
        
        Args:
            image: Imagem OpenCV
            region: {"xy_min": [x, y], "xy_max": [x, y]}
            label: Texto do rótulo
            color: Cor BGR da box
        """
        x1, y1 = region["xy_min"]
        x2, y2 = region["xy_max"]
        
        # Desenhar retângulo
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Preparar rótulo
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_w, label_h = label_size
        
        # Desenhar fundo do rótulo
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        
        # Desenhar texto
        cv2.putText(image, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def process_image(self, image_name):
        """
        Processa uma imagem: detecta objetos e verifica ocupação das regiões
        
        Args:
            image_name: Nome do arquivo da imagem
        """
        # Carregar imagem
        image_path = os.path.join(self.images_path, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Erro ao carregar imagem: {image_path}")
            return
        
        print(f"\nProcessando: {image_name}")
        
        # Fazer predições com o modelo
        results = self.model(image, verbose=False)
        
        # Extrair detecções
        model_detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())
                
                model_detections.append({
                    'box': box,
                    'confidence': conf,
                    'class': cls
                })
        
        print(f"  Detecções do modelo: {len(model_detections)}")
        
        # Criar cópia da imagem para desenhar
        output_image = image.copy()
        
        # Verificar se há regiões mapeadas para esta imagem
        if image_name not in self.regions_map:
            print(f"  Nenhuma região mapeada para esta imagem")
            cv2.imwrite(os.path.join(self.output_path, image_name), output_image)
            return
        
        manual_regions = self.regions_map[image_name]
        print(f"  Regiões manuais: {len(manual_regions)}")
        
        # Verificar ocupação de cada região manual
        occupied_count = 0
        empty_count = 0
        
        for region in manual_regions:
            is_occupied = self.check_region_occupancy(region, model_detections)
            
            if is_occupied:
                # Região ocupada - desenhar com rótulo "objeto"
                color = (0, 255, 0)  # Verde
                label = "objeto"
                occupied_count += 1
            else:
                # Região vazia - desenhar com rótulo "vazio"
                color = (0, 0, 255)  # Vermelho
                label = "vazio"
                empty_count += 1
            
            self.draw_region(output_image, region, label, color)
        
        print(f"  Ocupadas: {occupied_count} | Vazias: {empty_count}")
        
        # Atualizar estatísticas
        self.stats["total_regions"] += len(manual_regions)
        self.stats["occupied_regions"] += occupied_count
        self.stats["empty_regions"] += empty_count
        
        # Salvar imagem processada
        output_file = os.path.join(self.output_path, image_name)
        cv2.imwrite(output_file, output_image)
        print(f"  Salvo em: {output_file}")
    
    def process_all_images(self):
        """
        Processa todas as imagens no diretório
        """
        # Obter lista de imagens
        image_files = [f for f in os.listdir(self.images_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        self.stats["total_images"] = len(image_files)
        
        print(f"\n{'='*60}")
        print(f"Processando {len(image_files)} imagens...")
        print(f"{'='*60}")
        
        for i, image_name in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]", end=" ")
            self.process_image(image_name)
        
        self.print_summary()
    
    def print_summary(self):
        """
        Imprime resumo das estatísticas
        """
        print(f"\n{'='*60}")
        print("RESUMO DO PROCESSAMENTO")
        print(f"{'='*60}")
        print(f"Total de imagens processadas: {self.stats['total_images']}")
        print(f"Total de regiões analisadas: {self.stats['total_regions']}")
        print(f"Regiões ocupadas: {self.stats['occupied_regions']}")
        print(f"Regiões vazias: {self.stats['empty_regions']}")
        
        if self.stats['total_regions'] > 0:
            ocupacao_pct = (self.stats['occupied_regions'] / self.stats['total_regions']) * 100
            print(f"Taxa de ocupação: {ocupacao_pct:.2f}%")
        
        print(f"\nImagens salvas em: {self.output_path}")
        print(f"{'='*60}")

# Executar o detector
if __name__ == "__main__":
    # Configurações
    MODEL_PATH = "models/train_lote12/best.pt"
    REGIONS_JSON = "data/regions_mapping.json"
    IMAGES_PATH = "data/imgs_test"
    OUTPUT_PATH = "data/imgs_output"
    
    # Criar e executar detector
    detector = RegionOccupancyDetector(
        model_path=MODEL_PATH,
        regions_json=REGIONS_JSON,
        images_path=IMAGES_PATH,
        output_path=OUTPUT_PATH
    )
    
    detector.process_all_images()