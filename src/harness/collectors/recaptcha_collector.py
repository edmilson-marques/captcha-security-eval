"""Coletor de amostras do Google reCAPTCHA v2 (Atividade 1).

Demo oficial do Google: https://www.google.com/recaptcha/api2/demo

Mapeamento do DOM confirmado por inspeção ao vivo em 2026-11-01: a página
carrega dois iframes do widget — o "anchor" (`iframe[src*='api2/anchor']`,
o checkbox "Não sou um robô", com o elemento clicável em `#recaptcha-anchor`)
e o "bframe" (`iframe[src*='api2/bframe']`, o desafio de imagem em si,
pré-carregado no DOM mas oculto até o checkbox ser clicado).

Um navegador automatizado/headless dispara consistentemente o desafio de
imagem (grade 3x3, ex. "selecione todas as imagens com um ônibus") em vez de
passar direto no checkbox — por isso este coletor precisa clicar o checkbox
e esperar o bframe aparecer antes de capturar a amostra; só screenshotar o
anchor sem clicar captura apenas o checkbox vazio, não o desafio em si (que
é o dado de interesse para a Atividade 2 — o classificador de grade de
imagem em `src/attacks/image_grid_cnn/`).

Observação para quando o recorte em células individuais for implementado: a
grade 3x3 do desafio não serve 9 arquivos de imagem separados — cada
`<td class="rc-imageselect-tile">` contém um `<img class="rc-image-tile-33">`
com o MESMO `src` (uma única imagem composta) e um `style="top:X%; left:Y%"`
(valores 0%/-100%/-200%) que recorta visualmente a célula via CSS. Para obter
as 9 sub-imagens de verdade é preciso baixar essa imagem composta e recortá-la
pelos mesmos offsets, não tratar cada `<img>` como um arquivo distinto.
"""

from __future__ import annotations

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from harness.collectors.base import BaseCollector

ANCHOR_IFRAME_SELECTOR = "iframe[src*='api2/anchor']"
CHALLENGE_IFRAME_SELECTOR = "iframe[src*='api2/bframe']"


class RecaptchaCollector(BaseCollector):
    target = "recaptcha"
    demo_url = "https://www.google.com/recaptcha/api2/demo"

    def locate_challenge(self):
        """Clica o checkbox do reCAPTCHA e retorna o iframe do desafio exibido.

        Assume que `self.driver` já navegou para `self.demo_url` (feito por
        `BaseCollector.collect` antes de chamar este método).
        """
        wait = WebDriverWait(self.driver, 15)

        anchor_iframe = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ANCHOR_IFRAME_SELECTOR))
        )
        self.driver.switch_to.frame(anchor_iframe)
        checkbox = wait.until(EC.element_to_be_clickable((By.ID, "recaptcha-anchor")))
        checkbox.click()
        self.driver.switch_to.default_content()

        # O iframe do desafio ("bframe") já existe no DOM antes do clique
        # (pré-carregado, oculto) — por isso `presence_of_element_located`
        # sozinho não basta: é preciso esperar ele ficar visível (o Google
        # anima a abertura da caixa de desafio) e depois esperar a grade de
        # imagens em si renderizar dentro dele antes de capturar a amostra.
        challenge_iframe = wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, CHALLENGE_IFRAME_SELECTOR))
        )
        self.driver.switch_to.frame(challenge_iframe)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))
        self.driver.switch_to.default_content()

        return challenge_iframe
