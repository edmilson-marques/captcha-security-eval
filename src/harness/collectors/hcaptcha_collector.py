"""Coletor de amostras do hCaptcha (Atividade 1).

Demo oficial da Intuition Machines: https://accounts.hcaptcha.com/demo

Mapeamento do DOM confirmado por inspeção ao vivo em 2026-11-01: a página
carrega dois iframes do widget, cujo `src` contém um fragmento estável e
independente de locale — `#frame=checkbox` (o "Sou humano") e
`#frame=challenge` (o desafio em si, pré-carregado oculto, igual ao padrão
observado no reCAPTCHA — ver `recaptcha_collector.py`). Evite selecionar por
`title`/ordem do DOM: o título do iframe é localizado (varia com `hl`) e os
dois iframes compartilham o mesmo domínio em `src`.

Dentro do iframe do checkbox, o elemento clicável tem `id="checkbox"`.

**Achado relevante para a Atividade 2 (desenho dos módulos de ataque)**: ao
contrário do pressuposto inicial de "grade de imagens" (que vale para o
reCAPTCHA v2 — ver `recaptcha_collector.py`), o desafio servido por este
sitekey de demo do hCaptcha nas duas coletas de teste (2026-11-01) foi
consistentemente um **quebra-cabeça de arrastar-e-encaixar** ("Arraste o
elemento para o local onde se encaixa"), renderizado em `<canvas>`, não uma
grade de imagens `<img>` clicáveis. Isso significa que o classificador de
`src/attacks/image_grid_cnn/` (desenhado para seleção em grade) não se aplica
diretamente a este desafio — ele pede alinhamento geométrico/espacial (ex.
correlação cruzada de template entre a peça e o recorte no fundo), uma
técnica de ataque distinta a ser desenvolvida separadamente. Documentar este
contraste é, em si, um resultado relevante do estudo comparativo.
"""

from __future__ import annotations

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from harness.collectors.base import BaseCollector

CHECKBOX_IFRAME_SELECTOR = "iframe[src*='frame=checkbox']"
CHALLENGE_IFRAME_SELECTOR = "iframe[src*='frame=challenge']"


class HcaptchaCollector(BaseCollector):
    target = "hcaptcha"
    demo_url = "https://accounts.hcaptcha.com/demo"

    def locate_challenge(self):
        """Clica o checkbox "Sou humano" e retorna o iframe do desafio exibido.

        Assume que `self.driver` já navegou para `self.demo_url` (feito por
        `BaseCollector.collect` antes de chamar este método).
        """
        wait = WebDriverWait(self.driver, 15)

        checkbox_iframe = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, CHECKBOX_IFRAME_SELECTOR))
        )
        self.driver.switch_to.frame(checkbox_iframe)
        checkbox = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#checkbox")))
        checkbox.click()
        self.driver.switch_to.default_content()

        # Mesmo padrão do reCAPTCHA: o iframe do desafio já existe no DOM
        # antes do clique, só fica visível/populado depois — esperar
        # visibilidade e o container do desafio renderizar antes de capturar.
        challenge_iframe = wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, CHALLENGE_IFRAME_SELECTOR))
        )
        self.driver.switch_to.frame(challenge_iframe)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".challenge-container")))
        self.driver.switch_to.default_content()

        return challenge_iframe
