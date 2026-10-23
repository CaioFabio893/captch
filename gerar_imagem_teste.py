"""
Gera imagens de teste parecidas com o caso real: fundo claro amarelado com
gradiente de iluminação, números amarelos, vermelhos e pretos, ruído e JPEG.

    python gerar_imagem_teste.py          -> cria exemplo_display.jpg e exemplo_7seg.png
"""

import cv2
import numpy as np

rng = np.random.default_rng(42)


def fundo_amarelado(h, w):
    img = np.zeros((h, w, 3), np.float32)
    img[:] = (205, 238, 248)  # BGR: creme/amarelado claro
    gradiente = np.linspace(-18, 10, w, dtype=np.float32)[None, :, None]  # iluminação desigual
    return np.clip(img + gradiente, 0, 255).astype(np.uint8)


def finalizar(img):
    img = img.astype(np.float32) + rng.normal(0, 6, img.shape)  # ruído do sensor
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(150):  # "sujeira": pontinhos escuros soltos
        y, x = rng.integers(0, img.shape[0]), rng.integers(0, img.shape[1])
        cv2.circle(img, (int(x), int(y)), 1, (90, 90, 90), -1)
    return img


def exemplo_display():
    img = fundo_amarelado(220, 760)
    fonte = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, "4827", (30, 140), fonte, 3.2, (20, 210, 250), 9, cv2.LINE_AA)   # amarelo
    cv2.putText(img, "315", (330, 140), fonte, 3.2, (30, 30, 215), 9, cv2.LINE_AA)    # vermelho
    cv2.putText(img, "96", (560, 140), fonte, 3.2, (25, 25, 25), 9, cv2.LINE_AA)      # preto
    img = finalizar(img)
    cv2.imwrite("exemplo_display.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])


SEGMENTOS = {  # a b c d e f g
    "0": "1111110", "1": "0110000", "2": "1101101", "3": "1111001", "4": "0110011",
    "5": "1011011", "6": "1011111", "7": "1110000", "8": "1111111", "9": "1111011",
}


def digito_7seg(img, d, x, y, cor, L=50, e=10, vao=3):
    """Desenha um dígito de 7 segmentos com vãos entre os segmentos."""
    h0, h1 = x + e + vao, x + L - e - vao  # segmentos horizontais (recuados p/ deixar vão)
    pos = {
        "a": ((h0, y), (h1, y + e)),
        "g": ((h0, y + L - e // 2), (h1, y + L + e // 2)),
        "d": ((h0, y + 2 * L - e), (h1, y + 2 * L)),
        "f": ((x, y + vao), (x + e, y + L - vao)),
        "b": ((x + L - e, y + vao), (x + L, y + L - vao)),
        "e": ((x, y + L + vao), (x + e, y + 2 * L - vao)),
        "c": ((x + L - e, y + L + vao), (x + L, y + 2 * L - vao)),
    }
    for seg, ligado in zip("abcdefg", SEGMENTOS[d]):
        if ligado == "1":
            cv2.rectangle(img, *pos[seg], cor, -1)


def exemplo_7seg():
    img = fundo_amarelado(200, 680)
    x = 30
    for numero, cor in (("27", (20, 210, 250)), ("50", (30, 30, 215)), ("381", (25, 25, 25))):
        for d in numero:
            digito_7seg(img, d, x, 45, cor)
            x += 68
        x += 30
    cv2.imwrite("exemplo_7seg.png", finalizar(img))


if __name__ == "__main__":
    exemplo_display()
    exemplo_7seg()
    print("Criadas: exemplo_display.jpg (esperado: 4827 315 96) e exemplo_7seg.png (esperado: 27 50 381)")
