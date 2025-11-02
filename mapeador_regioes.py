import cv2
import json
import os
from pathlib import Path
from datetime import datetime


class MapeadorRegioes:
    """
    Sistema para mapear manualmente regiões em imagens
    Cada região tem um ID único e coordenadas (x_min, y_min, x_max, y_max)
    """
    
    def __init__(self, caminho_imagem, pasta_saida="mapeamentos"):
        """
        Inicializa o mapeador
        
        Args:
            caminho_imagem: Caminho para a imagem a ser mapeada
            pasta_saida: Pasta onde serão salvos os mapeamentos
        """
        self.caminho_imagem = Path(caminho_imagem)
        self.pasta_saida = Path(pasta_saida)
        self.pasta_saida.mkdir(exist_ok=True)
        
        # Carregar imagem
        self.imagem_original = cv2.imread(str(caminho_imagem))
        if self.imagem_original is None:
            raise ValueError(f"Não foi possível carregar a imagem: {caminho_imagem}")
        
        self.altura, self.largura = self.imagem_original.shape[:2]
        self.imagem_display = self.imagem_original.copy()
        
        # Variáveis de controle
        self.desenhando = False
        self.ponto_inicial = None
        self.ponto_final = None
        
        # Armazenamento de regiões
        self.regioes = {}  # {id: {'coords': (x_min, y_min, x_max, y_max), 'items': []}}
        self.id_atual = 1
        self.item_atual = 1
        
        # Região temporária sendo desenhada
        self.regiao_temp = None
        
        # Nome da janela
        self.nome_janela = "Mapeador de Regiões"
        
        print("=" * 70)
        print("MAPEADOR DE REGIÕES")
        print("=" * 70)
        print(f"Imagem: {self.caminho_imagem.name}")
        print(f"Resolução: {self.largura}x{self.altura}")
        print("=" * 70)
        self._mostrar_ajuda()
    
    def _mostrar_ajuda(self):
        """Mostra instruções de uso"""
        print("\n📋 INSTRUÇÕES:")
        print("-" * 70)
        print("DESENHAR REGIÃO:")
        print("  1. Clique e arraste com o mouse para desenhar uma região")
        print("  2. Solte o botão para finalizar")
        print()
        print("TECLAS:")
        print("  ENTER     - Confirmar região atual (salva com ID)")
        print("  N         - Nova região (incrementa ID)")
        print("  I         - Adicionar item na região atual (mesmo ID)")
        print("  D         - Deletar última região/item")
        print("  L         - Listar todas as regiões mapeadas")
        print("  S         - Salvar mapeamento em arquivo JSON")
        print("  R         - Resetar visualização")
        print("  H         - Mostrar esta ajuda")
        print("  ESC ou Q  - Sair")
        print("-" * 70)
        print(f"\n🎯 ID Atual: {self.id_atual} | Item: {self.item_atual}")
        print(f"📊 Total de IDs mapeados: {len(self.regioes)}")
        print("=" * 70)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos do mouse"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Início do desenho
            self.desenhando = True
            self.ponto_inicial = (x, y)
            self.ponto_final = (x, y)
            print(f"\n📍 Ponto inicial: ({x}, {y})")
        
        elif event == cv2.EVENT_MOUSEMOVE:
            # Atualizando durante o desenho
            if self.desenhando:
                self.ponto_final = (x, y)
                self.atualizar_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finaliza o desenho
            self.desenhando = False
            self.ponto_final = (x, y)
            
            if self.ponto_inicial:
                x_min = min(self.ponto_inicial[0], self.ponto_final[0])
                y_min = min(self.ponto_inicial[1], self.ponto_final[1])
                x_max = max(self.ponto_inicial[0], self.ponto_final[0])
                y_max = max(self.ponto_inicial[1], self.ponto_final[1])
                
                # Valida se a região tem tamanho mínimo
                if (x_max - x_min) > 5 and (y_max - y_min) > 5:
                    self.regiao_temp = {
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'largura': x_max - x_min,
                        'altura': y_max - y_min
                    }
                    
                    print(f"📍 Ponto final: ({x}, {y})")
                    print(f"📦 Região desenhada:")
                    print(f"   Coords: ({x_min}, {y_min}) → ({x_max}, {y_max})")
                    print(f"   Tamanho: {x_max - x_min}x{y_max - y_min} pixels")
                    print(f"\n💡 Pressione ENTER para confirmar ou redesenhe")
                else:
                    print("⚠️  Região muito pequena! Desenhe novamente.")
                    self.regiao_temp = None
            
            self.atualizar_display()
    
    def atualizar_display(self):
        """Atualiza a visualização da imagem"""
        self.imagem_display = self.imagem_original.copy()
        
        # Desenhar regiões confirmadas
        for id_regiao, dados in self.regioes.items():
            # Cada ID pode ter múltiplos itens
            for idx, item in enumerate(dados['items'], 1):
                coords = item['coords']
                
                # Cor baseada no ID (para diferenciar)
                cor = self._gerar_cor_id(id_regiao)
                
                # Desenhar retângulo
                cv2.rectangle(
                    self.imagem_display,
                    (coords['x_min'], coords['y_min']),
                    (coords['x_max'], coords['y_max']),
                    cor, 2
                )
                
                # Label com ID e número do item
                label = f"ID:{id_regiao}-{idx}"
                tamanho_texto = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Fundo para o texto
                cv2.rectangle(
                    self.imagem_display,
                    (coords['x_min'], coords['y_min'] - tamanho_texto[1] - 8),
                    (coords['x_min'] + tamanho_texto[0] + 4, coords['y_min']),
                    cor, -1
                )
                
                # Texto
                cv2.putText(
                    self.imagem_display, label,
                    (coords['x_min'] + 2, coords['y_min'] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
        
        # Desenhar região temporária (amarela)
        if self.regiao_temp:
            cv2.rectangle(
                self.imagem_display,
                (self.regiao_temp['x_min'], self.regiao_temp['y_min']),
                (self.regiao_temp['x_max'], self.regiao_temp['y_max']),
                (0, 255, 255), 2
            )
            label = f"TEMP (ID:{self.id_atual}-{self.item_atual})"
            cv2.putText(
                self.imagem_display, label,
                (self.regiao_temp['x_min'], self.regiao_temp['y_min'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
        
        # Desenhar região sendo desenhada (azul)
        if self.desenhando and self.ponto_inicial and self.ponto_final:
            cv2.rectangle(
                self.imagem_display,
                self.ponto_inicial,
                self.ponto_final,
                (255, 0, 0), 2
            )
        
        # Informações na tela
        info_texto = f"ID: {self.id_atual} | Item: {self.item_atual} | Total IDs: {len(self.regioes)}"
        cv2.putText(
            self.imagem_display, info_texto,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        
        # Instruções rápidas
        cv2.putText(
            self.imagem_display, "ENTER=Confirmar | N=Novo ID | I=Novo Item | S=Salvar | H=Ajuda",
            (10, self.altura - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        cv2.imshow(self.nome_janela, self.imagem_display)
    
    def _gerar_cor_id(self, id_regiao):
        """Gera uma cor única baseada no ID"""
        # Usa o ID para gerar cores consistentes
        np.random.seed(id_regiao * 123)
        cor = tuple(np.random.randint(50, 255, 3).tolist())
        return cor
    
    def confirmar_regiao(self):
        """Confirma a região temporária e adiciona ao mapeamento"""
        if not self.regiao_temp:
            print("⚠️  Nenhuma região para confirmar!")
            return False
        
        # Cria entrada para o ID se não existir
        if self.id_atual not in self.regioes:
            self.regioes[self.id_atual] = {
                'id': self.id_atual,
                'items': []
            }
        
        # Adiciona o item
        item_data = {
            'item_numero': self.item_atual,
            'coords': self.regiao_temp,
            'timestamp': datetime.now().isoformat()
        }
        
        self.regioes[self.id_atual]['items'].append(item_data)
        
        print(f"\n✓ Região confirmada!")
        print(f"   ID: {self.id_atual}")
        print(f"   Item: {self.item_atual}")
        print(f"   Total de itens no ID {self.id_atual}: {len(self.regioes[self.id_atual]['items'])}")
        
        # Incrementa item para o próximo
        self.item_atual += 1
        
        # Limpa região temporária
        self.regiao_temp = None
        self.atualizar_display()
        
        return True
    
    def novo_id(self):
        """Cria um novo ID para as próximas regiões"""
        self.id_atual += 1
        self.item_atual = 1
        print(f"\n🆕 Novo ID criado: {self.id_atual}")
        print(f"   Item resetado para: {self.item_atual}")
        self.atualizar_display()
    
    def adicionar_item_mesmo_id(self):
        """Adiciona um novo item no ID atual"""
        print(f"\n➕ Pronto para adicionar item {self.item_atual} no ID {self.id_atual}")
        print(f"   Desenhe a próxima região e pressione ENTER")
    
    def deletar_ultimo(self):
        """Deleta o último item mapeado"""
        if not self.regioes:
            print("⚠️  Nenhuma região para deletar!")
            return
        
        # Encontra o último ID com itens
        ultimo_id = max(self.regioes.keys())
        
        if self.regioes[ultimo_id]['items']:
            item_removido = self.regioes[ultimo_id]['items'].pop()
            print(f"\n🗑️  Item removido:")
            print(f"   ID: {ultimo_id}")
            print(f"   Item: {item_removido['item_numero']}")
            
            # Se não tem mais itens, remove o ID
            if not self.regioes[ultimo_id]['items']:
                del self.regioes[ultimo_id]
                print(f"   ID {ultimo_id} removido (sem itens)")
                
                # Ajusta ID atual
                if self.regioes:
                    self.id_atual = max(self.regioes.keys())
                    self.item_atual = len(self.regioes[self.id_atual]['items']) + 1
                else:
                    self.id_atual = 1
                    self.item_atual = 1
            else:
                self.item_atual = len(self.regioes[ultimo_id]['items']) + 1
        
        self.atualizar_display()
    
    def listar_regioes(self):
        """Lista todas as regiões mapeadas"""
        print("\n" + "=" * 70)
        print("📋 REGIÕES MAPEADAS")
        print("=" * 70)
        
        if not self.regioes:
            print("⚠️  Nenhuma região mapeada ainda!")
            print("=" * 70)
            return
        
        total_items = 0
        
        for id_regiao, dados in sorted(self.regioes.items()):
            print(f"\n🔹 ID: {id_regiao}")
            print(f"   Total de itens: {len(dados['items'])}")
            
            for item in dados['items']:
                coords = item['coords']
                print(f"   📦 Item {item['item_numero']}:")
                print(f"      Coordenadas: ({coords['x_min']}, {coords['y_min']}) → ({coords['x_max']}, {coords['y_max']})")
                print(f"      Dimensões: {coords['largura']}x{coords['altura']} pixels")
                print(f"      Área: {coords['largura'] * coords['altura']} pixels²")
                total_items += 1
        
        print("\n" + "-" * 70)
        print(f"📊 RESUMO:")
        print(f"   Total de IDs: {len(self.regioes)}")
        print(f"   Total de itens: {total_items}")
        print(f"   Média de itens por ID: {total_items / len(self.regioes):.1f}")
        print("=" * 70)
    
    def salvar_mapeamento(self):
        """Salva o mapeamento em arquivo JSON"""
        if not self.regioes:
            print("⚠️  Nenhuma região para salvar!")
            return False
        
        # Preparar dados
        dados_salvar = {
            'imagem': {
                'nome': self.caminho_imagem.name,
                'caminho': str(self.caminho_imagem),
                'resolucao': {
                    'largura': self.largura,
                    'altura': self.altura
                }
            },
            'data_mapeamento': datetime.now().isoformat(),
            'total_ids': len(self.regioes),
            'total_items': sum(len(dados['items']) for dados in self.regioes.values()),
            'regioes': []
        }
        
        # Converter regiões para formato serializável
        for id_regiao, dados in sorted(self.regioes.items()):
            regiao_data = {
                'id': id_regiao,
                'total_items': len(dados['items']),
                'items': dados['items']
            }
            dados_salvar['regioes'].append(regiao_data)
        
        # Nome do arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_base = self.caminho_imagem.stem
        arquivo_json = self.pasta_saida / f"{nome_base}_mapeamento_{timestamp}.json"
        
        # Salvar
        with open(arquivo_json, 'w', encoding='utf-8') as f:
            json.dump(dados_salvar, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Mapeamento salvo!")
        print(f"   Arquivo: {arquivo_json}")
        print(f"   Total de IDs: {dados_salvar['total_ids']}")
        print(f"   Total de itens: {dados_salvar['total_items']}")
        
        # Também salva versão simplificada (apenas coordenadas)
        arquivo_simples = self.pasta_saida / f"{nome_base}_coords_{timestamp}.txt"
        with open(arquivo_simples, 'w') as f:
            for id_regiao, dados in sorted(self.regioes.items()):
                for item in dados['items']:
                    coords = item['coords']
                    f.write(f"ID:{id_regiao} Item:{item['item_numero']} ")
                    f.write(f"{coords['x_min']},{coords['y_min']},{coords['x_max']},{coords['y_max']}\n")
        
        print(f"   Coords TXT: {arquivo_simples}")
        
        return True
    
    def executar(self):
        """Loop principal do mapeador"""
        cv2.namedWindow(self.nome_janela, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.nome_janela, self.mouse_callback)
        
        self.atualizar_display()
        
        while True:
            tecla = cv2.waitKey(1) & 0xFF
            
            # ESC ou Q: Sair
            if tecla == 27 or tecla == ord('q'):
                print("\n❓ Deseja salvar antes de sair? (S/N)")
                resposta = input("Digite S ou N: ").strip().upper()
                if resposta == 'S':
                    self.salvar_mapeamento()
                print("\n👋 Saindo...")
                break
            
            # ENTER: Confirmar região
            elif tecla == 13:
                self.confirmar_regiao()
            
            # N: Novo ID
            elif tecla == ord('n'):
                self.novo_id()
            
            # I: Adicionar item no mesmo ID
            elif tecla == ord('i'):
                self.adicionar_item_mesmo_id()
            
            # D: Deletar último
            elif tecla == ord('d'):
                self.deletar_ultimo()
            
            # L: Listar regiões
            elif tecla == ord('l'):
                self.listar_regioes()
            
            # S: Salvar
            elif tecla == ord('s'):
                self.salvar_mapeamento()
            
            # R: Resetar visualização
            elif tecla == ord('r'):
                self.atualizar_display()
            
            # H: Ajuda
            elif tecla == ord('h'):
                self._mostrar_ajuda()
        
        cv2.destroyAllWindows()


def main():
    """Função principal"""
    print("\n" + "=" * 70)
    print("SISTEMA DE MAPEAMENTO MANUAL DE REGIÕES")
    print("=" * 70)
    
    # Solicita caminho da imagem
    caminho_imagem = input("\nDigite o caminho da imagem: ").strip()
    
    if not os.path.exists(caminho_imagem):
        print(f"❌ Erro: Arquivo não encontrado: {caminho_imagem}")
        return
    
    # Solicita pasta de saída (opcional)
    pasta_saida = input("Pasta para salvar mapeamentos (ENTER para 'mapeamentos'): ").strip()
    if not pasta_saida:
        pasta_saida = "mapeamentos"
    
    try:
        # Inicia o mapeador
        mapeador = MapeadorRegioes(caminho_imagem, pasta_saida)
        mapeador.executar()
        
        print("\n" + "=" * 70)
        print("✓ Mapeamento finalizado!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()