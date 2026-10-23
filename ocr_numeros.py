#!/usr/bin/env python3
"""
ocr_numeros.py - Extrai números coloridos (amarelo, vermelho, preto) de uma imagem
de display/contador usando OpenCV + Tesseract.

Por que não usar só escala de cinza + threshold?
    Em cinza, amarelo puro (R=255, G=255, B=0) vira ~226, quase o mesmo tom de um
    fundo claro amarelado (~235). Nenhum threshold separa bem os dois. Este script
    usa representações em que AS TRÊS cores ficam escuras e o fundo fica claro:

      * canal mínimo  min(B, G, R): amarelo -> B~0, vermelho -> G,B~0, preto -> ~0
      * distância de cor (Lab) de cada pixel até o fundo estimado localmente

    Depois testa várias estratégias de limpeza x vários modos do Tesseract (--psm)
    e fica com a leitura de maior confiança. Também informa a cor de cada número.

Uso:
    python ocr_numeros.py imagem.png
    python ocr_numeros.py imagem.jpg --psm 11          # números espalhados
    python ocr_numeros.py display.png --segmentos      # display de 7 segmentos
    python ocr_numeros.py imagem.png --sem-janela --salvar-debug
    python ocr_numeros.py imagem.png --json            # saída para outros programas
"""

import argparse
import json
import math
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# Altura (px) em que o Tesseract lê dígitos melhor. A imagem é redimensionada
# para que os dígitos fiquem perto disso, o que também deixa os kernels
# morfológicos abaixo com tamanho fixo e previsível.
ALTURA_ALVO = 40
MARGEM = 20  # borda branca em volta: o Tesseract erra com texto colado na borda
PSMS_PADRAO = (6, 7, 11)  # bloco uniforme, linha única, texto esparso
DIGITOS = "0123456789"
# Trocas comuns quando o Tesseract lê um dígito como letra/símbolo.
CONFUSOES = str.maketrans("OoDQlIi|![]ZzSsBGbgq", "00001111111225586699")
_releituras = {}  # cache: (forma, hash do recorte) -> (texto, confiança)

# Display de 7 segmentos: região de cada segmento (x0, x1, y0, y1), relativa à caixa do dígito.
#    aaa
#   f   b
#    ggg
#   e   c
#    ddd
REGIOES_7SEG = {
    "a": (0.25, 0.75, 0.00, 0.15), "b": (0.70, 1.00, 0.10, 0.45),
    "c": (0.70, 1.00, 0.55, 0.90), "d": (0.25, 0.75, 0.85, 1.00),
    "e": (0.00, 0.30, 0.55, 0.90), "f": (0.00, 0.30, 0.10, 0.45),
    "g": (0.25, 0.75, 0.43, 0.57),
}
PADROES_7SEG = {  # segmentos acesos, na ordem abcdefg
    "1111110": "0", "0110000": "1", "1101101": "2", "1111001": "3", "0110011": "4",
    "1011011": "5", "1011111": "6", "0011111": "6", "1110000": "7", "1110010": "7",
    "1111111": "8", "1111011": "9", "1110011": "9",
}

CAMINHOS_TESSERACT = {
    "Windows": [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ],
    "Darwin": ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"],
    "Linux": ["/usr/bin/tesseract", "/usr/local/bin/tesseract"],
}

INSTRUCOES_INSTALACAO = """
ERRO: o executável do Tesseract não foi encontrado.
O pacote 'pytesseract' é só uma ponte; o motor do Tesseract precisa estar instalado.

  Windows : winget install --id UB-Mannheim.TesseractOCR
            (ou o instalador em https://github.com/UB-Mannheim/tesseract/wiki)
  Linux   : sudo apt install tesseract-ocr        (Debian/Ubuntu)
            sudo dnf install tesseract            (Fedora)
  macOS   : brew install tesseract

Se instalou em outro lugar, informe o caminho:
  python ocr_numeros.py imagem.png --tesseract "C:\\caminho\\tesseract.exe"
  (ou defina a variável de ambiente TESSERACT_CMD)

Detalhes passo a passo no README.md.
"""


@dataclass
class Leitura:
    texto: str
    confianca: float
    caixa: tuple  # (x, y, w, h) na imagem redimensionada
    cor: str = "?"


@dataclass
class Candidato:
    estrategia: str
    psm: object  # número do --psm, "7seg" (decodificador), "10" (dígito a dígito) ou "voto"
    leituras: list
    mascara: np.ndarray

    @property
    def modo(self):
        return {"7seg": "7 segmentos", "10": "1 a 1", "voto": "votação"}.get(self.psm, f"psm {self.psm}")

    @property
    def total_digitos(self):
        return sum(len(l.texto) for l in self.leituras)

    @property
    def pontuacao(self):
        # Confiança ao quadrado penaliza "dígitos" de ruído com confiança baixa,
        # mas ainda premia quem lê mais dígitos com confiança alta.
        return sum(len(l.texto) * (l.confianca / 100) ** 2 for l in self.leituras)


# --------------------------------------------------------------------------- #
# Configuração e carga
# --------------------------------------------------------------------------- #
def configurar_tesseract(caminho=None):
    """Localiza o executável do Tesseract e devolve sua versão (ou encerra com instruções)."""
    caminho = caminho or os.environ.get("TESSERACT_CMD")
    if caminho:
        pytesseract.pytesseract.tesseract_cmd = caminho
    elif not shutil.which("tesseract"):
        # No Windows o instalador geralmente NÃO coloca o Tesseract no PATH.
        for candidato in CAMINHOS_TESSERACT.get(platform.system(), []):
            if Path(candidato).is_file():
                pytesseract.pytesseract.tesseract_cmd = candidato
                break

    try:
        versao = pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print(INSTRUCOES_INSTALACAO, file=sys.stderr)
        sys.exit(2)

    try:
        if (versao.major, versao.minor) < (4, 1):
            print(f"AVISO: Tesseract {versao} é antigo; a whitelist de dígitos só funciona "
                  "bem a partir da 4.1. Atualize se aparecerem letras.", file=sys.stderr)
    except AttributeError:
        pass
    return str(versao)


def carregar_imagem(caminho):
    """Lê a imagem como BGR. Usa imdecode porque cv2.imread falha com acentos no caminho (Windows)."""
    p = Path(caminho)
    if not p.is_file():
        raise FileNotFoundError(f"arquivo não encontrado: {p}")
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"não foi possível decodificar a imagem (formato inválido?): {p}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # PNG com transparência: compõe sobre branco (senão o transparente vira preto).
        alfa = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3] * alfa + 255 * (1 - alfa)).astype(np.uint8)
    if img.dtype != np.uint8:  # PNG de 16 bits
        img = cv2.convertScaleAbs(img, alpha=255.0 / img.max())
    return img


# --------------------------------------------------------------------------- #
# Pré-processamento
# --------------------------------------------------------------------------- #
def canal_minimo(img):
    """min(B, G, R) por pixel: amarelo, vermelho e preto ficam escuros; fundo claro fica claro."""
    b, g, r = cv2.split(img)
    return cv2.min(cv2.min(b, g), r)


def estimar_altura_digitos(img, segmentos=False):
    """Estimativa grosseira da altura dos dígitos (px), usada para escolher a escala."""
    _, mascara = cv2.threshold(canal_minimo(img), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if segmentos:
        # Junta os segmentos de cada dígito, senão mede só um segmento.
        vao = max(3, int(0.06 * img.shape[0]))
        mascara = cv2.dilate(mascara, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, vao // 2), vao)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    alturas = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    validos = (alturas >= 5) & (alturas < img.shape[0] * 0.95)
    if not validos.any():
        return None
    alturas, areas = alturas[validos], areas[validos]
    # Ruído costuma ser numeroso, mas pequeno: mede só os componentes com área de dígito.
    referencia = np.median(np.sort(areas)[-5:])
    altura = float(np.median(alturas[areas >= 0.25 * referencia]))
    return altura - (vao - 1) if segmentos else altura  # desconta o que a dilatação somou


def calcular_escala(img, altura, alvo=ALTURA_ALVO):
    h, w = img.shape[:2]
    if altura is None:
        return 2.0 if max(h, w) < 400 else 1.0
    escala = float(np.clip(alvo / altura, 0.25, 6.0))
    return min(escala, 5000 / max(h, w))  # evita imagens gigantes


def estimar_fundo_local(img):
    """Imagem "sem os números": mediana numa janela bem maior que um dígito.

    Corrige gradientes de iluminação e reflexos de fotos de displays, coisa que
    um único threshold global não consegue.
    """
    h, w = img.shape[:2]
    pequeno = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    lado = min(pequeno.shape[:2])
    if lado < 15:
        return np.full_like(img, np.median(img.reshape(-1, 3), axis=0).astype(np.uint8))
    k = min(31, lado if lado % 2 else lado - 1)
    fundo = cv2.medianBlur(pequeno, k)
    return cv2.resize(fundo, (w, h), interpolation=cv2.INTER_LINEAR)


def limiar_robusto(valores, k=6.0, minimo=20.0):
    """Threshold "tudo que foge do ruído do fundo". Diferente do Otsu, não ignora o
    amarelo quando há preto na imagem (Otsu tende a cortar entre amarelo e preto)."""
    mediana = float(np.median(valores))
    mad = float(np.median(np.abs(valores - mediana))) * 1.4826
    return max(mediana + k * max(mad, 1.0), minimo)


def mascara_distancia_cor(img, fundo):
    """Estratégia 1: distância de cor (Lab) de cada pixel até o fundo local."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_fundo = cv2.cvtColor(fundo, cv2.COLOR_BGR2LAB).astype(np.float32)
    distancia = cv2.GaussianBlur(np.linalg.norm(lab - lab_fundo, axis=2), (3, 3), 0)
    return np.where(distancia > limiar_robusto(distancia), 255, 0).astype(np.uint8)


def mascara_canal_minimo(img, fundo):
    """Estratégia 2: canal mínimo normalizado pelo fundo + Otsu invertido."""
    minimo = canal_minimo(img).astype(np.float32)
    minimo_fundo = canal_minimo(fundo).astype(np.float32)
    normalizado = np.clip(minimo / np.maximum(minimo_fundo, 1) * 255, 0, 255).astype(np.uint8)
    normalizado = cv2.GaussianBlur(normalizado, (3, 3), 0)
    _, mascara = cv2.threshold(normalizado, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mascara


def mascara_adaptativa(img, fundo):
    """Estratégia 3: threshold adaptativo (a abordagem "clássica"), mas no canal mínimo."""
    return cv2.adaptiveThreshold(canal_minimo(img), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 2 * ALTURA_ALVO + 1, 15)


ESTRATEGIAS = {
    "distancia_cor": mascara_distancia_cor,
    "canal_minimo": mascara_canal_minimo,
    "adaptativo": mascara_adaptativa,
}


def limpar_mascara(mascara, segmentos=False):
    """Remove ruído. Máscara: 255 = número, 0 = fundo."""
    # Abertura com kernel pequeno tira pontinhos e serrilhado sem afinar os traços.
    elipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elipse)
    if segmentos:
        # Display de 7 segmentos: fecha os vãos entre os segmentos para formar dígitos inteiros.
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)))

    # Remove componentes conexos pequenos demais (ruído) e molduras do display.
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if n <= 1:
        return mascara
    alt, larg = mascara.shape
    areas = stats[1:, cv2.CC_STAT_AREA]
    referencia = float(np.median(np.sort(areas)[-10:]))  # tamanho típico de um dígito
    area_minima = max(15.0, 0.03 * referencia)

    manter = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        encosta_borda = x == 0 or y == 0 or x + w >= larg or y + h >= alt
        if area < area_minima:
            continue
        if encosta_borda and w > 0.8 * larg:  # linha/moldura horizontal do display
            continue
        if area > 0.5 * alt * larg:  # mancha enorme não é dígito
            continue
        manter[i] = True
    return np.where(manter[rotulos], 255, 0).astype(np.uint8)


def preparar_para_tesseract(mascara):
    """Números pretos em fundo branco, com margem."""
    return cv2.copyMakeBorder(255 - mascara, MARGEM, MARGEM, MARGEM, MARGEM,
                              cv2.BORDER_CONSTANT, value=255)


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def reler_com_whitelist(imagem, caixa):
    """Relê UM número com o Tesseract restrito a dígitos (psm 7 = uma linha só)."""
    x, y, w, h = caixa
    recorte = cv2.copyMakeBorder(imagem[y:y + h, x:x + w], MARGEM, MARGEM, MARGEM, MARGEM,
                                 cv2.BORDER_CONSTANT, value=255)
    chave = (recorte.shape, hash(recorte.tobytes()))  # psm diferentes costumam achar a mesma caixa
    if chave not in _releituras:
        dados = pytesseract.image_to_data(recorte, config=f"--psm 7 -c tessedit_char_whitelist={DIGITOS}",
                                          output_type=pytesseract.Output.DICT)
        palavras = [(t.strip(), float(c)) for t, c in zip(dados["text"], dados["conf"])
                    if t.strip() and float(c) >= 0]
        _releituras[chave] = ("".join(t for t, _ in palavras), min((c for _, c in palavras), default=0.0))
    return _releituras[chave]


def ler_digitos(imagem, psm):
    """OCR em duas etapas.

    1. Tesseract SEM whitelist localiza e separa os números. (No Tesseract 5 a
       whitelist de dígitos também proíbe o espaço: "4827 315" vira "4827315" com
       confiança 0. E aspas para incluir o espaço na whitelist não funcionam no Windows.)
    2. Cada número tem as confusões típicas corrigidas (O->0, l->1, S->5...) e é
       relido COM whitelist; fica a versão de maior confiança.
    """
    dados = pytesseract.image_to_data(imagem, config=f"--oem 3 --psm {psm}",
                                      output_type=pytesseract.Output.DICT)
    leituras = []
    for i, bruto in enumerate(dados["text"]):
        bruto, confianca = bruto.strip(), float(dados["conf"][i])
        if not bruto or confianca < 0:
            continue
        caixa = (dados["left"][i], dados["top"][i], dados["width"][i], dados["height"][i])

        opcoes = []
        corrigido = "".join(c for c in bruto.translate(CONFUSOES) if c in DIGITOS)
        tem_digito = any(c in DIGITOS for c in bruto)
        if tem_digito and len(corrigido) >= 0.75 * len(bruto):
            # penaliza caracteres que precisaram ser descartados
            opcoes.append((corrigido, confianca * len(corrigido) / len(bruto)))
        texto_wl, conf_wl = reler_com_whitelist(imagem, caixa)
        # Palavra sem nenhum dígito (ex.: "Total") só vira número se a releitura for muito confiante.
        if texto_wl and (tem_digito or conf_wl >= 80):
            opcoes.append((texto_wl, conf_wl))
        if not opcoes:
            continue

        texto, confianca = max(opcoes, key=lambda o: o[1])
        x, y, w, h = caixa
        leituras.append(Leitura(texto, confianca, (x - MARGEM, y - MARGEM, w, h)))
    return leituras


def decodificar_7seg(img, mascara):
    """Alternativa ao Tesseract para displays de 7 segmentos (que ele lê mal):
    verifica quais segmentos de cada dígito estão acesos."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if n <= 1:
        return []
    altura_max = stats[1:, cv2.CC_STAT_HEIGHT].max()
    altos = [s for s in stats[1:] if s[3] >= 0.6 * altura_max]  # descarta ponto decimal, ":", ruído
    if not altos:
        return []
    largura_tipica = float(np.percentile([s[2] for s in altos], 90))
    if largura_tipica < 0.3 * altura_max:  # só tem "1"s: não dá para medir a célula
        largura_tipica = 0.55 * altura_max

    digitos = []  # (x, y, w, h, dígito, clareza)
    for x, y, w, h, _ in altos:
        # "1", "3" e "7" não acendem os segmentos da esquerda: alarga a caixa para a
        # largura de um dígito inteiro, ancorada à direita, antes de medir.
        if w < 0.85 * largura_tipica:
            x, w = max(0, int(x + w - largura_tipica)), int(largura_tipica)
        roi = mascara[y:y + h, x:x + w] > 0
        estados, clarezas = "", []
        for seg in "abcdefg":
            x0, x1, y0, y1 = REGIOES_7SEG[seg]
            regiao = roi[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1),
                         int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)]
            preenchido = float(regiao.mean())
            estados += "1" if preenchido > 0.4 else "0"
            clarezas.append((preenchido - 0.4) / 0.6 if preenchido > 0.4 else (0.4 - preenchido) / 0.4)
        if estados in PADROES_7SEG:
            digitos.append((x, y, w, h, PADROES_7SEG[estados], float(np.mean(clarezas))))

    # Agrupa dígitos vizinhos da mesma cor em números.
    leituras, grupo, fim_anterior = [], [], None
    for d in sorted(digitos, key=lambda d: (round(d[1] / altura_max), d[0])):
        x, y, w, h = d[:4]
        cor = classificar_cor(img, mascara, (x, y, w, h))
        if grupo and (cor != grupo[-1][1] or x - fim_anterior > 0.35 * altura_max
                      or abs(y - grupo[-1][0][1]) > 0.5 * altura_max):
            leituras.append(_juntar_digitos(grupo))
            grupo = []
        grupo.append((d, cor))
        fim_anterior = x + w
    if grupo:
        leituras.append(_juntar_digitos(grupo))
    return leituras


def _juntar_digitos(grupo):
    x0 = min(d[0] for d, _ in grupo)
    y0 = min(d[1] for d, _ in grupo)
    x1 = max(d[0] + d[2] for d, _ in grupo)
    y1 = max(d[1] + d[3] for d, _ in grupo)
    return Leitura("".join(d[4] for d, _ in grupo), 100 * float(np.mean([d[5] for d, _ in grupo])),
                   (x0, y0, x1 - x0, y1 - y0), grupo[0][1])


def separar_caracteres(mascara, n):
    """Caixas dos n componentes com mais cara de dígito, da esquerda para a direita."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    caixas = sorted((list(s[:4]) for s in stats[1:]), key=lambda c: c[0])

    # Junta pedaços do mesmo dígito (um "5" quebrado, por exemplo): muita sobreposição horizontal.
    unidas = []
    for x, y, w, h in caixas:
        if unidas:
            ux, uy, uw, uh = unidas[-1]
            if min(ux + uw, x + w) - max(ux, x) > 0.5 * min(uw, w):
                nx, ny = min(ux, x), min(uy, y)
                unidas[-1] = [nx, ny, max(ux + uw, x + w) - nx, max(uy + uh, y + h) - ny]
                continue
        unidas.append([x, y, w, h])

    def parece_digito(c):
        x, y, w, h = c
        preenchimento = (mascara[y:y + h, x:x + w] > 0).mean()
        # dígitos são mais altos que largos e não são blocos maciços nem riscos finos
        return 0.9 <= h / max(w, 1) <= 6 and 0.12 <= preenchimento <= 0.8

    candidatas = [c for c in unidas if parece_digito(c)] or unidas
    altura_ref = np.median(sorted((c[3] for c in candidatas), reverse=True)[:n])
    candidatas = [c for c in candidatas if c[3] >= 0.5 * altura_ref]
    escolhidas = sorted(candidatas, key=lambda c: c[2] * c[3], reverse=True)[:n]
    return sorted(escolhidas, key=lambda c: c[0])


def votar_digitos(candidatos, n):
    """Combina as leituras que acharam exatamente n dígitos: cada posição fica com o
    dígito mais votado (peso = confiança²). A confiança final é a % de concordância."""
    if not candidatos:
        return None
    votos = [{} for _ in range(n)]
    for c in candidatos:
        sequencia = [(d, l.confianca) for l in sorted(c.leituras, key=lambda l: l.caixa[0]) for d in l.texto]
        for i, (d, conf) in enumerate(sequencia):
            votos[i][d] = votos[i].get(d, 0.0) + (conf / 100) ** 2
    vencedores = [max(v, key=v.get) if v else "" for v in votos]
    concordancia = [100 * v[d] / sum(v.values()) if v and sum(v.values()) else 0.0
                    for v, d in zip(votos, vencedores)]

    # Caixas: de preferência de uma leitura dígito a dígito (uma caixa por dígito).
    base = max(candidatos, key=lambda c: (len(c.leituras) == n, c.pontuacao))
    if len(base.leituras) == n:
        caixas = [l.caixa for l in sorted(base.leituras, key=lambda l: l.caixa[0])]
        leituras = [Leitura(d, conf, caixa) for d, conf, caixa in zip(vencedores, concordancia, caixas)]
    else:
        x0 = min(l.caixa[0] for l in base.leituras)
        y0 = min(l.caixa[1] for l in base.leituras)
        x1 = max(l.caixa[0] + l.caixa[2] for l in base.leituras)
        y1 = max(l.caixa[1] + l.caixa[3] for l in base.leituras)
        leituras = [Leitura("".join(vencedores), float(np.mean(concordancia)), (x0, y0, x1 - x0, y1 - y0))]
    return Candidato(base.estrategia, "voto", leituras, base.mascara)


def ler_um_digito(recorte):
    """Lê um único dígito (preto no branco) com psm 10. Devolve (dígito, confiança) ou ("", 0)."""
    recorte = cv2.copyMakeBorder(recorte, MARGEM, MARGEM, MARGEM, MARGEM, cv2.BORDER_CONSTANT, value=255)
    opcoes = []
    for config, corrigir in ((f"--psm 10 -c tessedit_char_whitelist={DIGITOS}", False),
                             ("--psm 10", True)):
        dados = pytesseract.image_to_data(recorte, config=config, output_type=pytesseract.Output.DICT)
        for t, c in zip(dados["text"], dados["conf"]):
            t = t.strip().translate(CONFUSOES) if corrigir else t.strip()
            t = "".join(ch for ch in t if ch in DIGITOS)
            if t and float(c) >= 0:
                opcoes.append((t[0], float(c)))
    return max(opcoes, key=lambda o: o[1]) if opcoes else ("", 0.0)


def ler_por_caractere(mascara, n):
    """Quando se sabe quantos dígitos há: lê cada um isolado (psm 10 = um caractere)."""
    imagem = 255 - mascara
    leituras = []
    for x, y, w, h in separar_caracteres(mascara, n):
        texto, conf = ler_um_digito(imagem[y:y + h, x:x + w])
        if texto:
            leituras.append(Leitura(texto, conf, (x, y, w, h)))
    return leituras


def ler_por_cor(img, n, ks=(8, 12)):
    """Fundo com formas coloridas: separa a imagem em k cores (k-means) e procura dígitos
    em cada cor. Como cada dígito tem uma cor só, ele se solta das formas de outras
    cores que encostam nele. Fica com os n dígitos que o Tesseract lê com mais confiança.

    Testa mais de um k: com poucas cores, um dígito pode se fundir a uma forma de cor
    parecida atrás dele; com muitas, um dígito sombreado pode se partir em dois."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # fechamento: tapa os furos da borda serrilhada e de linhas finas que cruzam o dígito
    elipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    achados, vistos = [], set()  # achados: (confiança, dígito, caixa, máscara do componente)
    for k in ks:
        _, rotulos, _ = cv2.kmeans(lab, k, None, criterio, 2, cv2.KMEANS_PP_CENTERS)
        rotulos = rotulos.reshape(h, w)
        cor_fundo = np.bincount(rotulos.ravel()).argmax()
        for cor in range(k):
            if cor == cor_fundo:
                continue
            m = cv2.morphologyEx(np.where(rotulos == cor, 255, 0).astype(np.uint8), cv2.MORPH_CLOSE, elipse)
            num, comp, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, num):
                x, y, cw, ch, area = stats[i]
                # mesmo filtro de "cara de dígito" de separar_caracteres
                if not (0.2 * h <= ch <= 0.95 * h and 0.9 <= ch / max(cw, 1) <= 6
                        and 0.12 <= area / (cw * ch) <= 0.8):
                    continue
                pedaco = np.where(comp[y:y + ch, x:x + cw] == i, 255, 0).astype(np.uint8)
                chave = (x, y, cw, ch, hash(pedaco.tobytes()))
                if chave in vistos:  # o mesmo componente aparece com k diferentes
                    continue
                vistos.add(chave)
                digito, conf = ler_um_digito(255 - pedaco)
                if digito:
                    achados.append((conf, digito, (x, y, cw, ch), pedaco))

    # Agrupa as leituras que caem na mesma posição (mesmo dígito visto com k diferentes);
    # cada grupo vota no dígito (peso = confiança²) e ficam os n grupos com mais peso.
    grupos = []
    for achado in sorted(achados, key=lambda a: -a[0]):
        x, _, cw, _ = achado[2]
        for g in grupos:
            gx, _, gw, _ = g[0][2]
            if min(x + cw, gx + gw) - max(x, gx) >= 0.3 * min(cw, gw):
                g.append(achado)
                break
        else:
            grupos.append([achado])

    def votos(g):
        v = {}
        for conf, digito, _, _ in g:
            v[digito] = v.get(digito, 0.0) + (conf / 100) ** 2
        return v

    escolhidos = []
    for g in sorted(grupos, key=lambda g: -sum(votos(g).values()))[:n]:
        v = votos(g)
        vencedor = max(v, key=v.get)
        # caixa/máscara da leitura mais confiante desse dígito; confiança = % dos votos
        _, _, caixa, pedaco = next(a for a in g if a[1] == vencedor)
        escolhidos.append((100 * v[vencedor] / sum(v.values()), vencedor, caixa, pedaco))

    mascara = np.zeros((h, w), np.uint8)
    for _, _, (x, y, cw, ch), pedaco in escolhidos:
        mascara[y:y + ch, x:x + cw] |= pedaco
    leituras = [Leitura(d, conf, caixa) for conf, d, caixa, _ in sorted(escolhidos, key=lambda e: e[2][0])]
    return Candidato("cores", "10", leituras, mascara)


def ordenar_em_linhas(leituras):
    """Agrupa leituras em linhas (por sobreposição vertical) e ordena da esquerda p/ direita."""
    linhas = []
    for l in sorted(leituras, key=lambda l: l.caixa[1]):
        centro = l.caixa[1] + l.caixa[3] / 2
        for linha in linhas:
            topo = min(o.caixa[1] for o in linha)
            base = max(o.caixa[1] + o.caixa[3] for o in linha)
            if topo <= centro <= base:
                linha.append(l)
                break
        else:
            linhas.append([l])
    return [sorted(linha, key=lambda l: l.caixa[0]) for linha in linhas]


def classificar_cor(img, mascara, caixa):
    """Nome da cor predominante dos pixels do número dentro da caixa."""
    x, y, w, h = caixa
    x, y = max(x, 0), max(y, 0)
    roi = img[y:y + h, x:x + w]
    sel = mascara[y:y + h, x:x + w] > 0
    if roi.size == 0 or not sel.any():
        return "?"
    # Erode para ignorar a borda serrilhada (mistura do número com o fundo).
    miolo = cv2.erode(sel.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if miolo.sum() > 20:
        sel = miolo

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[sel].astype(np.float32)
    matiz, sat, val = hsv[:, 0] * 2, hsv[:, 1], hsv[:, 2]  # matiz em graus (0-360)
    if np.median(val) < 80 or np.median(sat) < 60:
        return "preto" if np.median(val) < 140 else "cinza"

    # Média circular da matiz (vermelho fica em 0° e 360° ao mesmo tempo).
    rad = np.radians(matiz[sat >= 60])
    graus = math.degrees(math.atan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360
    for limite, nome in ((20, "vermelho"), (40, "laranja"), (70, "amarelo"),
                         (170, "verde"), (260, "azul"), (330, "roxo")):
        if graus < limite:
            return nome
    return "vermelho"


# --------------------------------------------------------------------------- #
# Debug visual
# --------------------------------------------------------------------------- #
def desenhar_deteccoes(img, leituras):
    saida = img.copy()
    for l in leituras:
        x, y, w, h = l.caixa
        cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 160, 0), 2)
        rotulo = f"{l.texto} {l.cor} {l.confianca:.0f}%"
        cv2.putText(saida, rotulo, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 120, 0), 1, cv2.LINE_AA)
    return saida


def montar_mosaico(paineis, largura=560, colunas=2):
    """Junta (título, imagem) em uma grade. Títulos sem acento: putText só tem ASCII."""
    blocos = []
    for titulo, im in paineis:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        im = cv2.resize(im, (largura, max(1, round(h * largura / w))), interpolation=cv2.INTER_AREA)
        barra = np.full((28, largura, 3), 50, np.uint8)
        cv2.putText(barra, titulo, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        blocos.append(np.vstack([barra, im]))

    altura = max(b.shape[0] for b in blocos)
    blocos = [cv2.copyMakeBorder(b, 4, altura - b.shape[0] + 4, 4, 4,
                                 cv2.BORDER_CONSTANT, value=(30, 30, 30)) for b in blocos]
    while len(blocos) % colunas:
        blocos.append(np.full_like(blocos[0], 30))
    return np.vstack([np.hstack(blocos[i:i + colunas]) for i in range(0, len(blocos), colunas)])


def mostrar_janela(imagem, titulo="Debug OCR"):
    """Mostra a imagem; fecha com qualquer tecla OU no X. Retorna False se não há interface gráfica."""
    try:
        cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
        h, w = imagem.shape[:2]
        cv2.resizeWindow(titulo, min(w, 1400), min(h, 900))
        cv2.imshow(titulo, imagem)
        print("\n(pressione qualquer tecla na janela, ou feche-a, para terminar)")
        # waitKey(0) puro trava para sempre se o usuário fechar a janela no X.
        while cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) != -1:
                break
        cv2.destroyAllWindows()
        return True
    except cv2.error:
        return False


def salvar_imagem(caminho, imagem):
    cv2.imencode(Path(caminho).suffix or ".png", imagem)[1].tofile(str(caminho))


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def processar(img, psms, estrategias, segmentos, digitos=None):
    """Roda todas as combinações estratégia x modo de leitura. Devolve (melhor, candidatos, img_redimensionada, escala)."""
    if digitos:
        # Código curto que ocupa boa parte da altura: formas grandes no fundo enganam a
        # medição dos dígitos, então a escala vem da altura da imagem (~110 px).
        escala = float(np.clip(110 / img.shape[0], 0.25, 6.0))
    else:
        escala = calcular_escala(img, estimar_altura_digitos(img, segmentos), ALTURA_ALVO)
    interpolacao = cv2.INTER_CUBIC if escala > 1 else cv2.INTER_AREA
    img_r = cv2.resize(img, None, fx=escala, fy=escala, interpolation=interpolacao)
    fundo = estimar_fundo_local(img_r)

    candidatos = []
    for nome in estrategias:
        mascara = limpar_mascara(ESTRATEGIAS[nome](img_r, fundo), segmentos)
        entrada = preparar_para_tesseract(mascara)
        for psm in psms:
            candidatos.append(Candidato(nome, psm, ler_digitos(entrada, psm), mascara))
        if segmentos:
            candidatos.append(Candidato(nome, "7seg", decodificar_7seg(img_r, mascara), mascara))
        if digitos:
            candidatos.append(Candidato(nome, "10", ler_por_caractere(mascara, digitos), mascara))

    if digitos:
        candidatos.append(ler_por_cor(img_r, digitos))

    # Com --segmentos, o decodificador tem prioridade; o Tesseract fica de reserva.
    decodificados = [c for c in candidatos if c.psm == "7seg" and c.leituras]
    preferidos = decodificados or candidatos
    melhor = max(preferidos, key=lambda c: c.pontuacao)
    if digitos:
        # Só valem leituras com a quantidade certa de dígitos; elas votam posição a posição.
        # (Se nenhuma acertar a quantidade, fica a melhor leitura e o usuário é avisado.)
        melhor = votar_digitos([c for c in preferidos if c.total_digitos == digitos], digitos) or melhor
    for l in melhor.leituras:
        l.cor = classificar_cor(img_r, melhor.mascara, l.caixa)
    return melhor, candidatos, img_r, escala


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description="Extrai números coloridos de uma imagem (OpenCV + Tesseract).")
    ap.add_argument("imagem", help="caminho da imagem, ou de uma pasta para ler todas as imagens dela")
    ap.add_argument("--psm", type=int, action="append",
                    help="modo de segmentação do Tesseract (pode repetir). Padrão: testa 6, 7 e 11")
    ap.add_argument("--estrategia", choices=list(ESTRATEGIAS), action="append",
                    help="força uma estratégia de pré-processamento (pode repetir). Padrão: todas")
    ap.add_argument("--segmentos", action="store_true", help="imagem de display de 7 segmentos")
    ap.add_argument("--digitos", type=int, metavar="N",
                    help="a imagem tem sempre N dígitos: lê cada um e junta da esquerda para a direita")
    ap.add_argument("--tesseract", help="caminho do executável tesseract (se não estiver no PATH)")
    ap.add_argument("--sem-janela", action="store_true", help="não abre a janela de debug")
    ap.add_argument("--salvar-debug", nargs="?", const="", metavar="ARQUIVO",
                    help="salva o mosaico de debug (padrão: <imagem>_debug.png)")
    ap.add_argument("--json", action="store_true", help="imprime o resultado em JSON")
    ap.add_argument("--txt", metavar="ARQUIVO",
                    help="arquivo de texto com os números (padrão: <imagem>.txt, ao lado da imagem)")
    ap.add_argument("-v", "--verbose", action="store_true", help="mostra o placar de todas as combinações")
    args = ap.parse_args()

    versao = configurar_tesseract(args.tesseract)
    psms = args.psm or list(PSMS_PADRAO)
    estrategias = args.estrategia or list(ESTRATEGIAS)

    # Pasta: lê todas as imagens e grava "nome da imagem: números" em resultados.txt.
    pasta = Path(args.imagem)
    if pasta.is_dir():
        extensoes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        resultados = []
        for arquivo in sorted(p for p in pasta.iterdir() if p.suffix.lower() in extensoes):
            try:
                melhor, *_ = processar(carregar_imagem(arquivo), psms, estrategias, args.segmentos, args.digitos)
                ordenadas = sorted(melhor.leituras, key=lambda l: l.caixa[0])
                texto = ("" if args.digitos else " ").join(l.texto for l in ordenadas) or "(nada encontrado)"
            except (FileNotFoundError, ValueError) as e:
                texto = f"(erro: {e})"
            resultados.append(f"{arquivo.name}: {texto}")
            print(resultados[-1])
        caminho_txt = Path(args.txt) if args.txt else pasta / "resultados.txt"
        caminho_txt.write_text("\n".join(resultados) + "\n", encoding="utf-8")
        print(f"\n{len(resultados)} imagens lidas. Resultado salvo em: {caminho_txt}")
        sys.exit(0)

    try:
        img = carregar_imagem(args.imagem)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)

    melhor, candidatos, img_r, escala = processar(img, psms, estrategias, args.segmentos, args.digitos)
    if args.digitos:
        # Código de N dígitos: tudo numa linha só, da esquerda para a direita, sem espaços.
        linhas = [sorted(melhor.leituras, key=lambda l: l.caixa[0])]
        texto_final = "".join(l.texto for l in linhas[0])
        if len(texto_final) != args.digitos:
            print(f"AVISO: esperava {args.digitos} dígitos, mas foram lidos {len(texto_final)}.",
                  file=sys.stderr)
    else:
        linhas = ordenar_em_linhas(melhor.leituras)
        texto_final = "\n".join(" ".join(l.texto for l in linha) for linha in linhas)

    # Resultado em texto: uma linha da imagem por linha do arquivo, números separados por espaço.
    caminho_txt = Path(args.txt) if args.txt else Path(args.imagem).with_suffix(".txt")
    caminho_txt.write_text(texto_final + "\n" if texto_final else "", encoding="utf-8")

    if args.json:
        print(json.dumps({
            "texto": texto_final,
            "numeros": [{
                "valor": l.texto,
                "cor": l.cor,
                "confianca": round(l.confianca, 1),
                # caixa convertida de volta para as coordenadas da imagem original
                "caixa": [round(v / escala) for v in l.caixa],
            } for linha in linhas for l in linha],
            "estrategia": melhor.estrategia,
            "psm": melhor.psm,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"Tesseract {versao} | escala {escala:.2f}x | {len(candidatos)} combinações testadas")
        if args.verbose:
            print("\n  estratégia      modo         pontuação  leitura")
            for c in sorted(candidatos, key=lambda c: -c.pontuacao):
                lido = " ".join(l.texto for l in c.leituras) or "-"
                print(f"  {c.estrategia:<15} {c.modo:<11} {c.pontuacao:9.2f}  {lido}")

        if melhor.leituras:
            print(f"\nNÚMEROS ENCONTRADOS: {texto_final.replace(chr(10), ' | ')}\n")
            print(f"  {'número':<12} {'cor':<10} {'confiança':>9}   posição (x, y, w, h)")
            for linha in linhas:
                for l in linha:
                    caixa = tuple(round(v / escala) for v in l.caixa)
                    print(f"  {l.texto:<12} {l.cor:<10} {l.confianca:>8.0f}%   {caixa}")
            print(f"\nMelhor combinação: {melhor.estrategia} + {melhor.modo} "
                  f"(pontuação {melhor.pontuacao:.2f})")
        else:
            print("\nNenhum número encontrado. Dicas: veja a imagem de debug; tente --segmentos "
                  "(display de 7 segmentos) ou --psm 11 (números espalhados).")
        print(f"\nResultado salvo em: {caminho_txt}")

    # Mosaico: original com detecções, máscara de cada estratégia e a entrada do Tesseract.
    melhor_por_estrategia = {}
    for c in candidatos:
        if c.pontuacao >= melhor_por_estrategia.get(c.estrategia, c).pontuacao:
            melhor_por_estrategia[c.estrategia] = c
    paineis = [("Original + deteccoes", desenhar_deteccoes(img_r, melhor.leituras))]
    for nome, c in melhor_por_estrategia.items():
        marca = " <- VENCEDORA" if c is melhor else ""
        lido = " ".join(l.texto for l in c.leituras) or "-"
        paineis.append((f"{nome} ({c.modo}): {lido}{marca}", 255 - c.mascara))
    mosaico = montar_mosaico(paineis)

    caminho_debug = None
    if args.salvar_debug is not None:
        caminho_debug = args.salvar_debug or str(Path(args.imagem).with_name(Path(args.imagem).stem + "_debug.png"))
    if not args.sem_janela and not mostrar_janela(mosaico):
        print("(sem interface gráfica disponível; salvando o debug em arquivo)", file=sys.stderr)
        caminho_debug = caminho_debug or str(Path(args.imagem).with_name(Path(args.imagem).stem + "_debug.png"))
    if caminho_debug:
        salvar_imagem(caminho_debug, mosaico)
        print(f"Imagem de debug salva em: {caminho_debug}", file=sys.stderr)

    sys.exit(0 if melhor.leituras else 1)


if __name__ == "__main__":
    main()
