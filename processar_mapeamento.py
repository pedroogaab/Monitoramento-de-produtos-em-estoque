import cv2
import json
import os
from pathlib import Path
import numpy as np


class ProcessadorMapeamento:
    """Processa e visualiza mapeamentos salvos"""
    
    def __init__(self, arquivo_json):
        """
        Inicializa o processador
        
        Args:
            arquivo_json: Caminho para o arquivo JSON do mapeamento
        """
        self.arquivo_json = Path(arquivo_json)
        
        # Carregar dados
        with open(arquivo_json, 'r', encoding='utf-8') as f:
            self.dados = json.load(f)
        
        print(f"✓ Mapeamento carregado: {self.arquivo_json.name}")
        print(f"  Total de IDs: {self.dados['total_ids']}")
        print(f"  Total de itens: {self.dados['total_items']}")
    
    def visualizar_mapeamento(self, salvar=True, arquivo_saida=None):
        """
        Visualiza o mapeamento desenhando todas as regiões
        
        Args:
            salvar: Se True, salva a imagem
            arquivo_saida: Nome do arquivo de saída (opcional)
        """
        # Carregar imagem
        caminho_img = self.dados['imagem']['caminho']
        
        # Tenta vários caminhos
        if not os.path.exists(caminho_img):
            # Tenta no mesmo diretório do JSON
            caminho_img = self.arquivo_json.parent.parent / self.dados['imagem']['nome']
        
        if not os.path.exists(caminho_img):
            print(f"⚠️  Imagem não encontrada: {caminho_img}")
            caminho_img = input("Digite o caminho da imagem: ").strip()
        
        img = cv2.imread(str(caminho_img))
        
        if img is None:
            print(f"❌ Não foi possível carregar a imagem")
            return None
        
        # Desenhar regiões
        for regiao in self.dados['regioes']:
            id_regiao = regiao['id']
            cor = self._gerar_cor_id(id_regiao)
            
            for item in regiao['items']:
                coords = item['coords']
                
                # Retângulo
                cv2.rectangle(img,
                            (coords['x_min'], coords['y_min']),
                            (coords['x_max'], coords['y_max']),
                            cor, 3)
                
                # Label
                label = f"ID:{id_regiao}-{item['item_numero']}"
                tamanho = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Fundo do texto
                cv2.rectangle(img,
                            (coords['x_min'], coords['y_min'] - tamanho[1] - 10),
                            (coords['x_min'] + tamanho[0] + 8, coords['y_min']),
                            cor, -1)
                
                # Texto
                cv2.putText(img, label,
                           (coords['x_min'] + 4, coords['y_min'] - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Salvar
        if salvar:
            if arquivo_saida is None:
                arquivo_saida = self.arquivo_json.stem + "_visualizado.jpg"
            
            cv2.imwrite(arquivo_saida, img)
            print(f"✓ Imagem salva: {arquivo_saida}")
        
        # Mostrar
        cv2.imshow('Mapeamento Visualizado', img)
        print("\nPressione qualquer tecla para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return img
    
    def _gerar_cor_id(self, id_regiao):
        """Gera cor consistente para um ID"""
        np.random.seed(id_regiao * 123)
        return tuple(np.random.randint(50, 255, 3).tolist())
    
    def estatisticas(self):
        """Mostra estatísticas do mapeamento"""
        print("\n" + "=" * 70)
        print("📊 ESTATÍSTICAS DO MAPEAMENTO")
        print("=" * 70)
        
        # Informações da imagem
        print(f"\n📷 IMAGEM:")
        print(f"   Nome: {self.dados['imagem']['nome']}")
        print(f"   Resolução: {self.dados['imagem']['resolucao']['largura']}x"
              f"{self.dados['imagem']['resolucao']['altura']}")
        
        # Informações gerais
        print(f"\n📦 GERAL:")
        print(f"   Total de IDs: {self.dados['total_ids']}")
        print(f"   Total de itens: {self.dados['total_items']}")
        print(f"   Média de itens por ID: {self.dados['total_items'] / self.dados['total_ids']:.1f}")
        
        # Estatísticas por ID
        print(f"\n🔢 POR ID:")
        
        areas_por_id = {}
        items_por_id = {}
        
        for regiao in self.dados['regioes']:
            id_regiao = regiao['id']
            items_por_id[id_regiao] = regiao['total_items']
            
            area_total = 0
            for item in regiao['items']:
                coords = item['coords']
                area_total += coords['largura'] * coords['altura']
            
            areas_por_id[id_regiao] = area_total
            
            print(f"   ID {id_regiao}:")
            print(f"      Itens: {regiao['total_items']}")
            print(f"      Área total: {area_total:,} pixels²")
            print(f"      Área média/item: {area_total / regiao['total_items']:,.0f} pixels²")
        
        # Estatísticas de áreas
        todas_areas = []
        for regiao in self.dados['regioes']:
            for item in regiao['items']:
                coords = item['coords']
                todas_areas.append(coords['largura'] * coords['altura'])
        
        print(f"\n📐 ÁREAS:")
        print(f"   Menor: {min(todas_areas):,} pixels²")
        print(f"   Maior: {max(todas_areas):,} pixels²")
        print(f"   Média: {sum(todas_areas) / len(todas_areas):,.0f} pixels²")
        print(f"   Total mapeado: {sum(todas_areas):,} pixels²")
        
        # Percentual da imagem
        area_imagem = (self.dados['imagem']['resolucao']['largura'] * 
                      self.dados['imagem']['resolucao']['altura'])
        percentual = (sum(todas_areas) / area_imagem) * 100
        print(f"   Cobertura da imagem: {percentual:.2f}%")
        
        print("\n" + "=" * 70)
    
    def exportar_csv(self, arquivo_saida=None):
        """Exporta mapeamento para CSV"""
        import csv
        
        if arquivo_saida is None:
            arquivo_saida = self.arquivo_json.stem + "_export.csv"
        
        with open(arquivo_saida, 'w', newline='', encoding='utf-8') as f:
            campos = ['id', 'item_numero', 'x_min', 'y_min', 'x_max', 'y_max',
                     'largura', 'altura', 'area', 'timestamp']
            
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            
            for regiao in self.dados['regioes']:
                for item in regiao['items']:
                    coords = item['coords']
                    writer.writerow({
                        'id': regiao['id'],
                        'item_numero': item['item_numero'],
                        'x_min': coords['x_min'],
                        'y_min': coords['y_min'],
                        'x_max': coords['x_max'],
                        'y_max': coords['y_max'],
                        'largura': coords['largura'],
                        'altura': coords['altura'],
                        'area': coords['largura'] * coords['altura'],
                        'timestamp': item.get('timestamp', '')
                    })
        
        print(f"✓ CSV exportado: {arquivo_saida}")
        return arquivo_saida
    
    def listar_detalhado(self):
        """Lista todos os itens com detalhes"""
        print("\n" + "=" * 70)
        print("📋 LISTAGEM DETALHADA")
        print("=" * 70)
        
        for regiao in self.dados['regioes']:
            print(f"\n🔹 ID: {regiao['id']} ({regiao['total_items']} itens)")
            print("-" * 70)
            
            for item in regiao['items']:
                coords = item['coords']
                area = coords['largura'] * coords['altura']
                
                print(f"   📦 Item {item['item_numero']}:")
                print(f"      Coordenadas: ({coords['x_min']}, {coords['y_min']}) → "
                      f"({coords['x_max']}, {coords['y_max']})")
                print(f"      Dimensões: {coords['largura']}x{coords['altura']} px")
                print(f"      Área: {area:,} pixels²")
                
                if 'timestamp' in item:
                    print(f"      Criado em: {item['timestamp']}")
                print()
        
        print("=" * 70)
    
    def filtrar_por_id(self, id_procurado):
        """Filtra e mostra apenas um ID específico"""
        for regiao in self.dados['regioes']:
            if regiao['id'] == id_procurado:
                print(f"\n🔍 ID {id_procurado} encontrado!")
                print(f"   Total de itens: {regiao['total_items']}")
                
                for item in regiao['items']:
                    coords = item['coords']
                    print(f"\n   Item {item['item_numero']}:")
                    print(f"      Coords: ({coords['x_min']}, {coords['y_min']}) → "
                          f"({coords['x_max']}, {coords['y_max']})")
                return regiao
        
        print(f"⚠️  ID {id_procurado} não encontrado")
        return None


def menu_interativo():
    """Menu interativo para processar mapeamentos"""
    print("\n" + "=" * 70)
    print("PROCESSADOR DE MAPEAMENTOS")
    print("=" * 70)
    
    # Solicitar arquivo
    arquivo = input("\nDigite o caminho do arquivo JSON: ").strip()
    
    if not os.path.exists(arquivo):
        print(f"❌ Arquivo não encontrado: {arquivo}")
        return
    
    # Criar processador
    proc = ProcessadorMapeamento(arquivo)
    
    while True:
        print("\n" + "-" * 70)
        print("OPÇÕES:")
        print("  1 - Visualizar mapeamento")
        print("  2 - Estatísticas")
        print("  3 - Listar detalhado")
        print("  4 - Exportar para CSV")
        print("  5 - Filtrar por ID")
        print("  0 - Sair")
        print("-" * 70)
        
        opcao = input("\nEscolha uma opção: ").strip()
        
        if opcao == '1':
            proc.visualizar_mapeamento()
        
        elif opcao == '2':
            proc.estatisticas()
        
        elif opcao == '3':
            proc.listar_detalhado()
        
        elif opcao == '4':
            proc.exportar_csv()
        
        elif opcao == '5':
            try:
                id_busca = int(input("Digite o ID: ").strip())
                proc.filtrar_por_id(id_busca)
            except ValueError:
                print("⚠️  Digite um número válido")
        
        elif opcao == '0':
            print("\n👋 Até mais!")
            break
        
        else:
            print("⚠️  Opção inválida")


if __name__ == "__main__":
    menu_interativo()