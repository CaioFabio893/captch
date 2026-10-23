# OCR de números coloridos (OpenCV + Tesseract)

Extrai números amarelos, vermelhos e pretos de imagens com fundo claro/amarelado
(displays, contadores), informando também **a cor de cada número**.

## 1. Instalar as bibliotecas Python

```bash
pip install opencv-python pytesseract numpy
```

(ou `pip install -r requirements.txt`)

## 2. Instalar o motor do Tesseract

O `pytesseract` só chama o programa `tesseract`; ele precisa estar instalado na máquina.
Use a versão **4.1 ou mais nova** (a whitelist de dígitos não funciona bem nas antigas).

### Windows

1. Abra o PowerShell e rode:
   ```bash
   winget install --id UB-Mannheim.TesseractOCR
   ```
   Sem winget: baixe o instalador `tesseract-ocr-w64-setup-*.exe` em
   https://github.com/UB-Mannheim/tesseract/wiki e instale com as opções padrão.
2. O instalador **não** coloca o Tesseract no PATH. O script já procura em
   `C:\Program Files\Tesseract-OCR\tesseract.exe`; se instalou em outra pasta, use
   `--tesseract "D:\caminho\tesseract.exe"` ou a variável de ambiente `TESSERACT_CMD`.
3. Feche e reabra o terminal e confira:
   ```bash
   "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
   ```

### Linux

```bash
sudo apt update && sudo apt install -y tesseract-ocr
```

Fedora: `sudo dnf install tesseract` · Arch: `sudo pacman -S tesseract tesseract-data-eng`.
Confira com `tesseract --version`.

> Servidor sem interface gráfica? Use `--sem-janela --salvar-debug`, ou instale
> `opencv-python-headless` no lugar de `opencv-python`.

### macOS

```bash
brew install tesseract
```

(instale o Homebrew antes, se não tiver: https://brew.sh). Confira com `tesseract --version`.

## 3. Usar

```bash
python ocr_numeros.py minha_imagem.jpg
```

Saída real com a imagem de exemplo (`exemplo_display.jpg`):

```
Tesseract 5.4.0.20240606 | escala 0.62x | 9 combinações testadas

NÚMEROS ENCONTRADOS: 4827 315 96

  número       cor        confiança   posição (x, y, w, h)
  4827         amarelo          96%   (29, 73, 244, 70)
  315          vermelho         96%   (332, 73, 184, 70)
  96           preto            96%   (561, 73, 124, 70)

Melhor combinação: distancia_cor + psm 11 (pontuação 8.29)
```

Uma janela abre com o **mosaico de debug**: a imagem com as detecções e a máscara
de cada estratégia (números pretos em fundo branco). Feche com qualquer tecla.

| Opção | Para quê |
|---|---|
| `--psm 11` | números espalhados pela imagem (pode repetir: `--psm 6 --psm 7`) |
| `--segmentos` | display de 7 segmentos: fecha os vãos e usa um decodificador de segmentos (o Tesseract lê 7 segmentos mal) |
| `--estrategia canal_minimo` | força uma estratégia (`distancia_cor`, `canal_minimo`, `adaptativo`) |
| `-v` | mostra o placar de todas as combinações testadas |
| `--sem-janela` | não abre janela |
| `--salvar-debug [arquivo]` | salva o mosaico (padrão: `<imagem>_debug.png`) |
| `--txt ARQUIVO` | onde salvar o resultado em texto (padrão: `<imagem>.txt`, ao lado da imagem) |
| `--json` | resultado em JSON, para usar em outro programa |
| `--tesseract CAMINHO` | caminho do `tesseract.exe` |

Códigos de saída: `0` encontrou números · `1` não encontrou · `2` erro (Tesseract ausente, imagem inválida).

Para testar sem uma imagem real: `python gerar_imagem_teste.py` cria `exemplo_display.jpg`
e `exemplo_7seg.png`.

## O que foi melhorado em relação ao prompt original

Teste com 9 variações da imagem de exemplo (normal, amarelo pastel, amarelo ouro, borrada,
ruído forte, 250 px, 2400 px, fundo branco, meia sombra):

| | acertos |
|---|---|
| Prompt original (cinza + Otsu + whitelist) | **0 de 9**: perde o amarelo e junta tudo (`31596`) |
| Este script | **9 de 9**, com a cor certa de cada número |


- **Amarelo não some mais.** Em escala de cinza, amarelo (≈226) e fundo creme (≈235) são
  quase iguais e o threshold apaga o número amarelo. O script usa o **canal mínimo**
  `min(B,G,R)`, onde amarelo, vermelho e preto ficam todos escuros, e a **distância de cor
  em Lab** até o fundo. Nos testes, cinza + Otsu perdeu o número amarelo inteiro; as novas
  estratégias pegaram as três cores.
- **Fundo estimado localmente**: corrige gradiente de iluminação e reflexos de foto.
- **Escala automática**: redimensiona para os dígitos ficarem com ~40 px, o tamanho em que o
  Tesseract acerta mais, e os filtros morfológicos passam a ter tamanho previsível.
- **Limpeza por componentes conexos** (remove pontinhos e molduras) em vez de só abertura
  morfológica, que afina e quebra traços finos.
- **A whitelist do prompt original atrapalha.** No Tesseract 5, `tessedit_char_whitelist=0123456789`
  também proíbe o espaço: `4827 315 96` vira `482731596` com confiança 0%. E colocar o espaço
  entre aspas na whitelist não funciona no Windows (o pytesseract não trata as aspas). Por isso o
  OCR é feito em **duas etapas**: o Tesseract sem whitelist separa os números; depois cada número
  tem confusões corrigidas (`O`→0, `l`/`!`→1, `S`→5, `B`→8...) e é relido **com** whitelist.
  Fica a versão de maior confiança.
- **Testa 3 estratégias × 3 modos (`--psm 6, 7, 11`)** e escolhe a leitura de maior
  confiança, em vez de depender de um único chute.
- **Decodificador de 7 segmentos** (`--segmentos`): o Tesseract leu `27 50 381` como `150 381`;
  o decodificador, que verifica quais segmentos estão acesos, acertou. Limitação: assume dígitos
  retos (sem inclinação) e o "1" alinhado à direita, como nos displays comuns.
- **Cor de cada número** e posição na imagem original.
- **Modo 7 segmentos**, **saída JSON**, suporte a PNG transparente e a caminhos com acento
  (o `cv2.imread` falha com eles no Windows).
- Tesseract encontrado automaticamente no Windows; mensagem clara se não estiver instalado.
- A janela de debug não trava quando é fechada no X.

## Se a leitura não sair 100%

1. Rode com `-v --salvar-debug` e olhe o mosaico: os números estão inteiros e pretos?
2. Número amarelo fraco/pastel falhando: force `--estrategia distancia_cor`.
3. Display de LED/LCD de 7 segmentos: `--segmentos`.
4. Números em posições soltas: `--psm 11`.
