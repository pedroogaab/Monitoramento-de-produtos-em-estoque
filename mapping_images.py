import os
import json
import cv2
import numpy as np
from pathlib import Path

class ImageRegionMapper:
    def __init__(self, images_path="data/imgs_test", output_json="regions_mapping.json"):
        self.images_path = images_path
        self.output_json = output_json
        self.current_image = None
        self.current_image_name = None
        self.display_image = None
        self.points = []  # Pontos temporários da imagem atual
        self.all_regions = {}  # Dicionário com todas as regiões mapeadas
        self.window_name = "Image Region Mapper"
        
        # Carregar dados existentes se houver
        self.load_existing_data()
        
    def load_existing_data(self):
        """Carrega dados existentes do arquivo JSON"""
        if os.path.exists(self.output_json):
            with open(self.output_json, 'r') as f:
                self.all_regions = json.load(f)
            print(f"Dados existentes carregados de {self.output_json}")
        else:
            print("Nenhum dado existente encontrado. Iniciando novo mapeamento.")
    
    def save_data(self):
        """Salva os dados no arquivo JSON"""
        with open(self.output_json, 'w') as f:
            json.dump(self.all_regions, f, indent=2)
        print(f"Dados salvos em {self.output_json}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos do mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Adicionar ponto
            self.points.append((x, y))
            print(f"Ponto adicionado: ({x}, {y})")
            
            # Se temos 2 pontos, criar região
            if len(self.points) == 2:
                self.add_region()
            
            self.draw_points()
    
    def add_region(self):
        """Adiciona uma região ao dicionário"""
        if len(self.points) == 2:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            
            # Garantir que min e max estão corretos
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)
            
            region = {
                "xy_min": [x_min, y_min],
                "xy_max": [x_max, y_max]
            }
            
            # Adicionar ao dicionário
            if self.current_image_name not in self.all_regions:
                self.all_regions[self.current_image_name] = []
            
            self.all_regions[self.current_image_name].append(region)
            print(f"Região adicionada: XYmin=({x_min}, {y_min}), XYmax=({x_max}, {y_max})")
            
            # Limpar pontos temporários
            self.points = []
            
            # Salvar automaticamente
            self.save_data()
    
    def draw_points(self):
        """Desenha todos os pontos e regiões na imagem"""
        self.display_image = self.current_image.copy()
        
        # Desenhar regiões já salvas
        if self.current_image_name in self.all_regions:
            for region in self.all_regions[self.current_image_name]:
                xy_min = tuple(region["xy_min"])
                xy_max = tuple(region["xy_max"])
                
                # Desenhar retângulo da região
                cv2.rectangle(self.display_image, xy_min, xy_max, (0, 255, 0), 2)
                
                # Desenhar pontos min e max
                cv2.circle(self.display_image, xy_min, 5, (255, 0, 0), -1)
                cv2.circle(self.display_image, xy_max, 5, (0, 0, 255), -1)
        
        # Desenhar pontos temporários
        for i, point in enumerate(self.points):
            color = (255, 0, 0) if i == 0 else (0, 0, 255)
            cv2.circle(self.display_image, point, 5, color, -1)
        
        # Se temos 1 ponto, desenhar linha até o cursor (opcional)
        if len(self.points) == 1:
            cv2.putText(self.display_image, "Clique para o segundo ponto", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Adicionar instruções na imagem
        instructions = [
            f"Imagem: {self.current_image_name}",
            f"Regioes: {len(self.all_regions.get(self.current_image_name, []))}",
            "Clique 2 pontos para criar regiao",
            "U: Desfazer ultima regiao",
            "N: Proxima imagem | P: Anterior",
            "Q: Sair e salvar"
        ]
        
        y_offset = 60
        for instruction in instructions:
            cv2.putText(self.display_image, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        cv2.imshow(self.window_name, self.display_image)
    
    def undo_last_region(self):
        """Desfaz a última região adicionada"""
        if self.current_image_name in self.all_regions and self.all_regions[self.current_image_name]:
            removed = self.all_regions[self.current_image_name].pop()
            print(f"Região removida: {removed}")
            self.save_data()
            self.draw_points()
        else:
            print("Nenhuma região para desfazer nesta imagem.")
    
    def load_image(self, image_name):
        """Carrega uma imagem"""
        image_path = os.path.join(self.images_path, image_name)
        self.current_image = cv2.imread(image_path)
        
        if self.current_image is None:
            print(f"Erro ao carregar imagem: {image_path}")
            return False
        
        self.current_image_name = image_name
        self.points = []
        print(f"\nCarregando: {image_name}")
        self.draw_points()
        return True
    
    def run(self):
        """Executa o mapeador interativo"""
        # Obter lista de imagens
        if not os.path.exists(self.images_path):
            print(f"Erro: Caminho {self.images_path} não encontrado!")
            return
        
        image_files = [f for f in os.listdir(self.images_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"Nenhuma imagem encontrada em {self.images_path}")
            return
        
        print(f"Encontradas {len(image_files)} imagens")
        
        # Criar janela
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        current_index = 0
        self.load_image(image_files[current_index])
        
        print("\n=== CONTROLES ===")
        print("Clique com o botão esquerdo: Adicionar ponto")
        print("U: Desfazer última região")
        print("N: Próxima imagem")
        print("P: Imagem anterior")
        print("Q: Sair e salvar")
        print("================\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q ou ESC
                print("Saindo...")
                self.save_data()
                break
            
            elif key == ord('n'):  # Próxima imagem
                current_index = (current_index + 1) % len(image_files)
                self.load_image(image_files[current_index])
            
            elif key == ord('p'):  # Imagem anterior
                current_index = (current_index - 1) % len(image_files)
                self.load_image(image_files[current_index])
            
            elif key == ord('u'):  # Desfazer
                self.undo_last_region()
        
        cv2.destroyAllWindows()
        
        # Mostrar resumo
        print("\n=== RESUMO DO MAPEAMENTO ===")
        total_regions = sum(len(regions) for regions in self.all_regions.values())
        print(f"Total de imagens mapeadas: {len(self.all_regions)}")
        print(f"Total de regiões: {total_regions}")
        print(f"Dados salvos em: {self.output_json}")

# Executar o mapeador
if __name__ == "__main__":
    mapper = ImageRegionMapper(
        images_path="data/imgs_test",
        output_json="data/regions_mapping.json"
    )
    mapper.run()