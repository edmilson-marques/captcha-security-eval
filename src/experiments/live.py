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


def recaptcha_llm_vision_attempt(
    demo_url: str = RECAPTCHA_DEMO_URL,
    headless: bool = True,
    max_rounds: int = 5,
) -> dict:
    """Resolve o desafio reCAPTCHA de verdade via o solver de visão (LLM).

    Mesmo fluxo de `recaptcha_baseline_attempt` até o desafio aparecer; a
    partir daí roda um **loop de refinamento por rodadas** (até `max_rounds`):
    tira um screenshot da grade (`<table>`), manda pro solver
    (`attacks.llm_vision_solver.solve_recaptcha_grid` — API paga da OpenAI,
    requer `OPENAI_API_KEY`), clica só os tiles que ainda **não** tinham sido
    clicados em rodadas anteriores, espera a animação de troca/seleção
    assentar, e repete — re-fotografando e repergunting ao modelo "quais
    tiles correspondem agora" a cada rodada. O loop para quando uma rodada
    não traz nenhum índice novo (convergiu) ou ao atingir `max_rounds`, e só
    então clica confirmar.

    Esse desenho resolve tanto grades "estáticas" (3x3/4x4 sem troca de
    imagem — a 2ª rodada não traz nada novo, custo de só +1 chamada de
    verificação) quanto "dinâmicas" (o reCAPTCHA substitui a imagem de um
    tile certo por uma foto nova — a rodada seguinte enxerga a imagem nova e
    pode identificar mais correspondências, até esgotar). Pesquisei como um
    framework de automação interno não-relacionado lida com CAPTCHA antes de
    desenhar isso: eles também não tinham essa refinamento iterativo pós-clique
    (só uma checagem de staleness pontual antes de clicar) — então este loop
    é um desenho próprio, não copiado de lugar nenhum.

    O número de tiles (`td.rc-imageselect-tile`) é usado para inferir o
    tamanho da grade (3x3 ou 4x4, sempre quadrada nas amostras observadas ao
    vivo) — o `id` de cada `<td>` já é o índice 0-based na mesma ordem
    (esquerda->direita, cima->baixo) usada no prompt do solver, então os
    índices retornados mapeiam direto para `td[id='<índice>']`, sem tradução.
    Clicar o mesmo índice duas vezes **desmarcaria** o tile no reCAPTCHA, por
    isso o conjunto `selected` é rastreado e nunca reclicado.

    **Antes desta versão** (passada única, sem o loop), 4 tentativas reais em
    2026-11-01 deram 0/4 sucessos — 3 por cair em grade dinâmica (resolvido
    agora pelo loop) e 1 por super-segmentação visual do modelo.

    **Com o loop** (3 tentativas reais em 2026-11-01, ainda 0/3, mas por um
    motivo NOVO e mais relevante): o mecanismo do loop em si funciona
    corretamente — 2 das 3 tentativas identificaram tiles na 1ª rodada e
    convergiram (0 índices novos) já na 2ª, confirmando que o "repete até não
    sobrar nada" funciona como desenhado. A causa da falha mudou: em 1 das 3
    tentativas, o modelo **recusou responder por segurança**
    (`reasoning: "Não posso ajudar a resolver ou contornar um CAPTCHA."`,
    `tile_indices: []`) — a API comercial da OpenAI aplica moderação de
    conteúdo que às vezes reconhece "resolver CAPTCHA" como tarefa vedada,
    mesmo com a saída forçada em JSON estruturado (a recusa só populou o
    schema com lista vazia em vez de quebrar o pipeline). Isso é uma barreira
    de defesa adicional que um classificador treinado do zero (`image_grid_cnn/`)
    não teria — achado relevante para a comparação do estudo entre técnicas.
    A recusa pareceu probabilística (não ocorreu em todas as tentativas),
    condizente com o comportamento típico de classificadores de segurança de
    LLM. Não investiguei mitigação (ex. reformular o prompt) — fica como
    próximo passo caso o orientador considere relevante.
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

        selected: set[int] = set()
        rounds_used = 0
        last_reasoning = ""
        for round_number in range(1, max_rounds + 1):
            rounds_used = round_number
            grid_screenshot = driver.find_element(By.CSS_SELECTOR, "table").screenshot_as_png
            answer = solve_recaptcha_grid(
                screenshot_png=grid_screenshot,
                instruction=instruction,
                grid_rows=grid_size,
                grid_cols=grid_size,
            )
            last_reasoning = answer.reasoning
            new_indices = [idx for idx in answer.tile_indices if idx not in selected]
            if not new_indices:
                break

            for tile_index in new_indices:
                driver.find_element(
                    By.CSS_SELECTOR, f"td.rc-imageselect-tile[id='{tile_index}']"
                ).click()
                selected.add(tile_index)
            time.sleep(1.0)  # dá tempo da animação de seleção/troca de tile assentar

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
            "sub_challenges": rounds_used,
            "extra": {
                "instruction": instruction,
                "grid_size": grid_size,
                "tile_indices": sorted(selected),
                "reasoning": last_reasoning,
                "rounds_used": rounds_used,
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
