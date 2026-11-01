"""Integração real entre harness/collectors e o runner de experimentos (Atividades 4-6).

Diferente do resto do pacote `experiments` (que é testável por injeção de
dependência, sem rede), este módulo abre um navegador de verdade e navega
contra as páginas de demo reais. Por isso não tem suíte pytest — foi
validado manualmente, ao vivo, contra as demos oficiais do Google e do
hCaptcha em 2026-11-01 (ver commits a65a162 e c5763d8, que corrigiram os
coletores com base nessa mesma inspeção).

Implementa duas técnicas para o reCAPTCHA:

- "baseline_no_select": clica o checkbox e confirma sem selecionar nada no
  desafio. Não usa nenhuma técnica de visão computacional — é o piso de
  comparação do estudo (o que acontece sem tentar resolver o desafio), útil
  como primeiro ponto de dado real para validar o pipeline inteiro
  (harness -> runner -> metrics -> analysis).
- "llm_vision": resolve de verdade via `src/attacks/llm_vision_solver.py`
  (modelo multimodal, API paga da OpenAI — requer `OPENAI_API_KEY`).
  Validado ao vivo em 2026-11-01 (ver `recaptcha_llm_vision_attempt`).

O hCaptcha ainda só tem a técnica "baseline_no_select" — os dois tipos de
desafio observados ao vivo (arrastar-e-encaixar, seleção de "laços
fechados") não são grade de tiles, então `image_grid_cnn`/`llm_vision_solver`
ainda não se aplicam diretamente a ele (ver nota em `hcaptcha_collector.py`).
"""

from __future__ import annotations

import time

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from attacks.llm_vision_solver import solve_recaptcha_grid
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
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, RECAPTCHA_CHALLENGE_IFRAME_SELECTOR)
                )
            )
        except TimeoutException:
            driver.switch_to.frame(anchor_iframe)
            passed = (
                driver.find_element(By.ID, "recaptcha-anchor").get_attribute("aria-checked")
                == "true"
            )
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
        succeeded = (
            driver.find_element(By.ID, "recaptcha-anchor").get_attribute("aria-checked") == "true"
        )
        driver.switch_to.default_content()

        return {
            "outcome": "success" if succeeded else "failure",
            "sub_challenges": 1,
            "extra": {"instruction": instruction},
        }
    finally:
        driver.quit()


def recaptcha_llm_vision_attempt(demo_url: str = RECAPTCHA_DEMO_URL, headless: bool = True) -> dict:
    """Resolve o desafio reCAPTCHA de verdade via o solver de visão (LLM).

    Mesmo fluxo de `recaptcha_baseline_attempt` até o desafio aparecer; a
    partir daí: tira um screenshot só da grade (`<table>`), manda pro solver
    (`attacks.llm_vision_solver.solve_recaptcha_grid` — API paga da OpenAI,
    requer `OPENAI_API_KEY`), clica os tiles que ele apontar e confirma.

    O número de tiles (`td.rc-imageselect-tile`) é usado para inferir o
    tamanho da grade (3x3 ou 4x4, sempre quadrada nas amostras observadas ao
    vivo) — o `id` de cada `<td>` já é o índice 0-based na mesma ordem
    (esquerda->direita, cima->baixo) usada no prompt do solver, então os
    índices retornados mapeiam direto para `td[id='<índice>']`, sem tradução.

    **Resultado real medido em 2026-11-01 (4 tentativas via
    `run_live_cli.py --target recaptcha --technique llm_vision`): 0/4
    sucessos.** Não é um bug do código — é uma limitação de design real e
    documentável:
    - 3 de 4 tentativas caíram numa grade 3x3 "dinâmica": ao clicar um tile
      correto, o reCAPTCHA **substitui aquele tile por uma nova foto**, que
      também pode corresponder à instrução — resolver de verdade exige
      repetir "olhar os tiles visíveis -> clicar os corretos -> esperar
      substituição" até não sobrar mais nenhum antes de confirmar. Esta
      função faz só UMA passada (screenshot -> clica -> confirma), então é
      estruturalmente incapaz de vencer esse tipo de grade — e é precisamente
      o mecanismo de defesa que a literatura (ver TCC1, seção 3.1.1) descreve
      como o ponto forte do reCAPTCHA contra ataques de passada única.
    - 1 de 4 foi uma grade 4x4 "estática" (sem substituição) onde o próprio
      modelo super-segmentou: respondeu `[8, 9, 12, 13]` quando a moto só
      ocupava visivelmente as células 9 e 13 — um erro de precisão visual,
      não de arquitetura. (Um teste manual anterior, fora deste loop, já
      tinha acertado `[5, 6, 9, 10]` numa grade estática equivalente — ver
      `attacks/llm_vision_solver.py` — então a precisão varia por amostra.)

    Melhoria natural de próximo passo (não implementada): loop de
    refinamento iterativo para grades dinâmicas — detectar troca de tile
    (ex. comparar o atributo `src`/hash da imagem do tile antes/depois do
    clique) e re-consultar o solver só nos tiles que mudaram, até estabilizar.
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
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, RECAPTCHA_CHALLENGE_IFRAME_SELECTOR)
                )
            )
        except TimeoutException:
            driver.switch_to.frame(anchor_iframe)
            passed = (
                driver.find_element(By.ID, "recaptcha-anchor").get_attribute("aria-checked")
                == "true"
            )
            driver.switch_to.default_content()
            return {"outcome": "success" if passed else "challenge_not_presented"}

        driver.switch_to.frame(challenge_iframe)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))

        instruction = driver.find_element(
            By.CSS_SELECTOR, ".rc-imageselect-desc-wrapper, .rc-imageselect-instructions"
        ).text
        tiles = driver.find_elements(By.CSS_SELECTOR, "td.rc-imageselect-tile")
        grid_size = round(len(tiles) ** 0.5)
        grid_screenshot = driver.find_element(By.CSS_SELECTOR, "table").screenshot_as_png

        answer = solve_recaptcha_grid(
            screenshot_png=grid_screenshot,
            instruction=instruction,
            grid_rows=grid_size,
            grid_cols=grid_size,
        )

        for tile_index in answer.tile_indices:
            driver.find_element(
                By.CSS_SELECTOR, f"td.rc-imageselect-tile[id='{tile_index}']"
            ).click()
        if answer.tile_indices:
            time.sleep(0.5)  # dá tempo da animação de seleção/troca de tile assentar

        driver.find_element(By.ID, "recaptcha-verify-button").click()
        time.sleep(2)
        driver.switch_to.default_content()

        driver.switch_to.frame(anchor_iframe)
        succeeded = (
            driver.find_element(By.ID, "recaptcha-anchor").get_attribute("aria-checked") == "true"
        )
        driver.switch_to.default_content()

        return {
            "outcome": "success" if succeeded else "failure",
            "sub_challenges": 1,
            "extra": {
                "instruction": instruction,
                "grid_size": grid_size,
                "tile_indices": answer.tile_indices,
                "reasoning": answer.reasoning,
            },
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
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, HCAPTCHA_CHALLENGE_IFRAME_SELECTOR)
                )
            )
        except TimeoutException:
            driver.switch_to.frame(checkbox_iframe)
            passed = (
                driver.find_element(By.CSS_SELECTOR, "#checkbox").get_attribute("aria-checked")
                == "true"
            )
            driver.switch_to.default_content()
            return {"outcome": "success" if passed else "challenge_not_presented"}

        driver.switch_to.frame(challenge_iframe)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".challenge-container")))
        prompt = driver.find_element(By.CSS_SELECTOR, ".prompt-text").text
        driver.find_element(By.CSS_SELECTOR, ".button-submit").click()
        time.sleep(2)
        driver.switch_to.default_content()

        driver.switch_to.frame(checkbox_iframe)
        succeeded = (
            driver.find_element(By.CSS_SELECTOR, "#checkbox").get_attribute("aria-checked")
            == "true"
        )
        driver.switch_to.default_content()

        return {
            "outcome": "success" if succeeded else "failure",
            "sub_challenges": 1,
            "extra": {"prompt": prompt},
        }
    finally:
        driver.quit()
