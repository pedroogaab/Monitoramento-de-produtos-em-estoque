import torch
from ultralytics import YOLO
import cv2
import json
import os
import numpy as np
from pathlib import Path
import time

class VideoRegionOccupancyDetector:
    def __init__(self, model_path, regions_json, video_path, output_path=None, 
                 video_key="frame_mercadinho", show_fps=True):
        """
        Inicializa o detector de ocupação para vídeos
        
        Args:
            model_path: Caminho para o modelo YOLO
            regions_json: Caminho para o JSON com regiões mapeadas
            video_path: Caminho para o vídeo
            output_path: Caminho para salvar o vídeo processado (None = não salva)
            video_key: Chave no JSON para as coordenadas (ex: "mercadinho.jpg")
            show_fps: Mostrar FPS na tela
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
            regions_map = json.load(f)
        
        # Obter regiões para este vídeo
        if video_key not in regions_map:
            raise ValueError(f"Chave '{video_key}' não encontrada no JSON. "
                           f"Chaves disponíveis: {list(regions_map.keys())}")
        
        self.manual_regions = regions_map[video_key]
        print(f"Carregadas {len(self.manual_regions)} regiões mapeadas")
        
        self.video_path = video_path
        self.output_path = output_path
        self.show_fps = show_fps
        
        # Estatísticas em tempo real
        self.frame_stats = {
            "frame_count": 0,
            "total_occupied": 0,
            "total_empty": 0
        }
        
        # FPS tracking
        self.fps = 0
        self.frame_time = 0
    
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
    
    def draw_info_panel(self, image, occupied_count, empty_count):
        """
        Desenha painel de informações no frame
        
        Args:
            image: Imagem OpenCV
            occupied_count: Número de regiões ocupadas
            empty_count: Número de regiões vazias
        """
        # Configurações do painel
        panel_height = 120
        panel_color = (0, 0, 0)
        text_color = (255, 255, 255)
        
        # Criar painel semi-transparente
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, panel_height), panel_color, -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Textos
        y_offset = 30
        texts = [
            f"Ocupadas: {occupied_count} | Vazias: {empty_count}",
            f"Total Regioes: {len(self.manual_regions)}",
        ]
        
        if self.show_fps:
            texts.append(f"FPS: {self.fps:.1f}")
        
        for text in texts:
            cv2.putText(image, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            y_offset += 25
    
    def process_frame(self, frame):
        """
        Processa um frame do vídeo
        
        Args:
            frame: Frame do vídeo (imagem OpenCV)
        
        Returns:
            Frame processado com anotações
        """
        start_time = time.time()
        
        # Fazer predições com o modelo
        results = self.model(frame, verbose=False)
        
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
        
        # Criar cópia do frame para desenhar
        output_frame = frame.copy()
        
        # Verificar ocupação de cada região manual
        occupied_count = 0
        empty_count = 0
        
        for region in self.manual_regions:
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
            
            self.draw_region(output_frame, region, label, color)
        
        # Atualizar estatísticas
        self.frame_stats["frame_count"] += 1
        self.frame_stats["total_occupied"] += occupied_count
        self.frame_stats["total_empty"] += empty_count
        
        # Desenhar painel de informações
        self.draw_info_panel(output_frame, occupied_count, empty_count)
        
        # Calcular FPS
        self.frame_time = time.time() - start_time
        self.fps = 1.0 / self.frame_time if self.frame_time > 0 else 0
        
        return output_frame
    
    def run(self):
        """
        Executa o detector no vídeo
        """
        # Abrir vídeo
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Erro ao abrir vídeo: {self.video_path}")
            return
        
        # Obter propriedades do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVídeo: {self.video_path}")
        print(f"Resolução: {width}x{height}")
        print(f"Regiões monitoradas: {len(self.manual_regions)}")
        
        # Configurar gravação se necessário
        out = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'M','J','P','G')  # Codec
            out = cv2.VideoWriter(filename=self.output_path, fourcc=fourcc, fps=fps, frameSize=(width, height))
            print(f"Vídeo salvo em: {self.output_path}")
        
        print("\nProcessando vídeo...")
        print("Pressione 'q' para sair, 'p' para pausar/continuar")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("\nFim do vídeo")
                        break
                    
                    # Processar frame
                    processed_frame = self.process_frame(frame)
                    
                    # Mostrar frame
                    cv2.imshow('Video Region Occupancy Detection', processed_frame)
                    
                    # Salvar frame se necessário
                    if out:
                        out.write(processed_frame)
                    
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nInterrompido pelo usuário")
                    break
                elif key == ord('p'):
                    paused = not paused
                    status = "PAUSADO" if paused else "CONTINUANDO"
                    print(f"\n{status}")
        
        finally:
            # Limpar recursos
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Mostrar estatísticas finais
            self.print_summary()
    
    def print_summary(self):
        """
        Imprime resumo das estatísticas
        """
        print(f"\n{'='*60}")
        print("RESUMO DO PROCESSAMENTO")
        print(f"{'='*60}")
        print(f"Total de regiões monitoradas: {len(self.manual_regions)}")
        
        if self.output_path:
            print(f"\nVídeo salvo em: {self.output_path}")
        
        print(f"{'='*60}")

# Executar o detector de vídeo
if __name__ == "__main__":
    # Configurações
    MODEL_PATH = "models/train_lote12/best.pt"
    REGIONS_JSON = "data/regions_mapping.json"
    VIDEO_PATH = "data/mercadinho.mp4"
    # OUTPUT_PATH = "data/mercadinho_output.mp4"  # None para não salvar
    OUTPUT_PATH = "data/mercadinho_output.avi"  # None para não salvar
    VIDEO_KEY = "frame_mercadinho.jpg"  # Chave no JSON com as coordenadas
    
    # Criar e executar detector de vídeo
    detector = VideoRegionOccupancyDetector(
        model_path=MODEL_PATH,
        regions_json=REGIONS_JSON,
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,  # Altere para None se não quiser salvar
        video_key=VIDEO_KEY,
        show_fps=True
    )
    
    detector.run()