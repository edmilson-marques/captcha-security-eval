"""Integração real entre harness/collectors e o runner de experimentos (Atividades 4-6).

Diferente do resto do pacote `experiments` (que é testável por injeção de
dependência, sem rede), este módulo abre um navegador de verdade e navega
contra as páginas de demo reais. Por isso não tem suíte pytest — foi
validado manualmente, ao vivo, contra as demos oficiais do Google e do
hCaptcha em 2026-11-01 (ver commits a65a162 e c5763d8, que corrigiram os
coletores com base nessa mesma inspeção).

Implementa, por enquanto, só a técnica "baseline_no_select": clica o
checkbox e confirma sem selecionar nada no desafio. Não usa nenhuma técnica
de visão computacional — é o piso de comparação do estudo (o que acontece
sem tentar resolver o desafio), útil como primeiro ponto de dado real para
validar o pipeline inteiro (harness -> runner -> metrics -> analysis) antes
de conectar técnicas mais sofisticadas (`src/attacks/`), que dependem de um
dataset rotulado ainda não coletado em volume.
"""

from __future__ import annotations

import time

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from harness.browser import BrowserConfig, build_driver
from harness.collectors.hcaptcha_collector import (
    CHALLENGE_IFRAME_SELECTOR as HCAPTCHA_CHALLENGE_IFRAME_SELECTOR,
)
from harness.collectors.hcaptcha_collector import (
    CHECKBOX_IFRAME_SELECTOR as HCAPTCHA_CHECKBOX_IFRAME_SELECTOR,
)
from harness.collectors.recaptcha_collector import (
    ANCHOR_IFRAME_SELECTOR as RECAPTCHA_ANCHOR_IFRAME_SELECTOR,
)
from harness.collectors.recaptcha_collector import (
    CHALLENGE_IFRAME_SELECTOR as RECAPTCHA_CHALLENGE_IFRAME_SELECTOR,
)

RECAPTCHA_DEMO_URL = "https://www.google.com/recaptcha/api2/demo"
HCAPTCHA_DEMO_URL = "https://accounts.hcaptcha.com/demo"


def recaptcha_baseline_attempt(demo_url: str = RECAPTCHA_DEMO_URL, headless: bool = True) -> dict:
    """Clica o checkbox do reCAPTCHA e confirma sem selecionar nenhum tile.

    Sinal de sucesso/falha confirmado ao vivo: o checkbox `#recaptcha-anchor`
    expõe `aria-checked="true"` quando o desafio é aceito, e permanece
    `"false"` quando a resposta é rejeitada (a demo então recarrega um novo
    desafio na mesma caixa). Se nenhum desafio de imagem aparecer dentro do
    timeout curto, o checkbox passou direto ("challenge_not_presented").
    """
    driver = build_driver(BrowserConfig(headless=headless))
    try:
        wait = WebDriverWait(driver, 15)
        driver.get(demo_url)

        anchor_iframe = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, RECAPTCHA_ANCHOR_IFRAME_SELECTOR))
        )
        driver.switch_to.frame(anchor_iframe)
        checkbox = wait.until(EC.element_to_be_clickable((By.ID, "recaptcha-anchor")))
        checkbox.click()
        driver.switch_to.default_content()

        try:
            challenge_iframe = WebDriverWait(driver, 6).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, RECAPTCHA_CHALLENGE_IFRAME_SELECTOR))
            )
        except TimeoutException:
            driver.switch_to.frame(anchor_iframe)
            passed = driver.find_element(By.ID, "recaptcha-anchor").get_attribute("aria-checked") == "true"
            driver.switch_to.default_content()
            return {"outcome": "success" if passed else "challenge_not_presented"}

        driver.switch_to.frame(challenge_iframe)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))
        instruction = driver.find_element(
            By.CSS_SELECTOR, ".rc-imageselect-desc-wrapper, .rc-imageselect-instructions"
        ).text
        driver.find_element(By.ID, "recaptcha-verify-button").click()
        time.sleep(2)
        driver.switch_to.default_content()

        driver.switch_to.frame(anchor_iframe)
        succeeded = driver.find_element(By.ID, "recaptcha-anchor").get_attribute("aria-checked") == "true"
        driver.switch_to.default_content()

        return {
            "outcome": "success" if succeeded else "failure",
            "sub_challenges": 1,
            "extra": {"instruction": instruction},
        }
    finally:
        driver.quit()


def hcaptcha_baseline_attempt(demo_url: str = HCAPTCHA_DEMO_URL, headless: bool = True) -> dict:
    """Clica o checkbox do hCaptcha e confirma ("Pular"/"Confirmar") sem interagir com o desafio.

    Mesmo sinal de sucesso/falha do reCAPTCHA, confirmado ao vivo: o checkbox
    `#checkbox` expõe `aria-checked="true"` só quando aceito. O botão de
    confirmação tem a classe estável `.button-submit` (o texto varia entre
    "Pular"/"Confirmar" conforme o tipo de desafio e idioma).
    """
    driver = build_driver(BrowserConfig(headless=headless))
    try:
        wait = WebDriverWait(driver, 15)
        driver.get(demo_url)

        checkbox_iframe = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, HCAPTCHA_CHECKBOX_IFRAME_SELECTOR))
        )
        driver.switch_to.frame(checkbox_iframe)
        checkbox = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#checkbox")))
        checkbox.click()
        driver.switch_to.default_content()

        try:
            challenge_iframe = WebDriverWait(driver, 6).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, HCAPTCHA_CHALLENGE_IFRAME_SELECTOR))
            )
        except TimeoutException:
            driver.switch_to.frame(checkbox_iframe)
            passed = driver.find_element(By.CSS_SELECTOR, "#checkbox").get_attribute("aria-checked") == "true"
            driver.switch_to.default_content()
            return {"outcome": "success" if passed else "challenge_not_presented"}

        driver.switch_to.frame(challenge_iframe)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".challenge-container")))
        prompt = driver.find_element(By.CSS_SELECTOR, ".prompt-text").text
        driver.find_element(By.CSS_SELECTOR, ".button-submit").click()
        time.sleep(2)
        driver.switch_to.default_content()

        driver.switch_to.frame(checkbox_iframe)
        succeeded = driver.find_element(By.CSS_SELECTOR, "#checkbox").get_attribute("aria-checked") == "true"
        driver.switch_to.default_content()

        return {
            "outcome": "success" if succeeded else "failure",
            "sub_challenges": 1,
            "extra": {"prompt": prompt},
        }
    finally:
        driver.quit()
