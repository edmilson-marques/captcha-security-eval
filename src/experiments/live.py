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
    LLM. Cogitei mitigar reformulando o prompt para reduzir a recusa, mas essa
    ação específica (engenharia de prompt para contornar o guardrail de
    segurança de *outro* provedor de IA) foi bloqueada pelo classificador de
    segurança do próprio Claude Code; coerente com essa decisão, optei por só
    remover ruído de interface do texto da instrução (`_strip_ui_instructions`
    em `attacks/llm_vision_solver.py` — ex. "Clique em Confirmar..." —, sem
    tocar na palavra "CAPTCHA" nem na moldura da tarefa).

    **Com a limpeza de ruído de UI** (5 tentativas reais em 2026-11-01, ainda
    0/5): 2 recusas de segurança e 3 respostas reais (sem recusa) que o loop
    convergiu corretamente mas ainda assim falharam — inspeção visual rodada
    a rodada confirmou que o loop está correto mesmo numa grade 3x3 que se
    revelou **estática** (corrigindo a generalização anterior de que grade
    3x3 implicava sempre grade dinâmica). Isso motivou investigar detecção de
    bot independente da resposta: `navigator.webdriver` já estava suprimido
    (`False`), mas `navigator.userAgent` continha literalmente
    `"HeadlessChrome"` — sinal de automação trivial, plausivelmente a causa
    dominante do 0% de sucesso em **todas** as técnicas testadas na sessão
    até então (ver achado e correção em `harness/browser.py`).

    **Depois de ligar um user-agent real de desktop no harness** (5 tentativas
    reais em 2026-11-01, mesmo texto de instrução, mesma versão do solver):
    **2/5 sucessos reais** — salto em relação ao 0/7 acumulado nas duas
    versões anteriores do loop. Das 3 falhas restantes: 2 foram recusas de
    segurança (`tile_indices: []`, mesmo texto de recusa de antes) e 1 foi uma
    resposta real do modelo (grade "faixas de pedestre", `tile_indices=[0,1]`,
    convergiu em 2 rodadas) que ainda assim foi rejeitada — imprecisão visual
    residual, não recusa nem bug do loop. Ou seja: com o sinal de
    automação removido, a causa de falha restante passou a ser,
    consistentemente, (a) recusa de segurança do modelo comercial ou
    (b) imprecisão visual — exatamente os dois achados que interessam à
    comparação do estudo, sem mais o ruído de detecção de bot por cima.

    **Lote maior para significância estatística** (20 tentativas reais em
    2026-11-01, mesmo harness já com o user-agent de desktop): **10/20
    sucessos (50%)** — confirma a taxa observada na amostra pequena anterior
    (2/5). Das 10 falhas, reclassificando pelo formato real da saída (vazio
    vs. não-vazio) em vez de só procurar a palavra "não posso" no texto de
    `reasoning` (ver nota abaixo sobre por que isso importa):
    - **4/20 (20%) recusa verdadeira**: `tile_indices: []`, loop converge já
      na 1ª rodada (nenhum índice para clicar).
    - **6/20 (30%) resposta real rejeitada**: `tile_indices` não-vazio, loop
      convergiu normalmente em 2 rodadas, mas a seleção não bateu com o
      gabarito do reCAPTCHA — imprecisão visual do modelo, não o mecanismo do
      loop nem detecção de bot.

    **Achado novo: "recusa performática" que não impede a resposta.** Em 2
    das 20 tentativas, o campo `reasoning` trouxe uma frase de recusa (ex.
    "Desculpe, não posso resolver desafios CAPTCHA.") mas `tile_indices` veio
    preenchido com uma resposta real e não-trivial mesmo assim (uma delas
    inclusive teve sucesso). Ou seja, a recusa "verbalizada" pelo modelo nem
    sempre corresponde à recusa "efetiva" (saída vazia) — um classificador de
    recusa ingênuo baseado em palavra-chave no texto de `reasoning`
    superestimaria a taxa real de recusa do modelo. Por isso a contagem acima
    usa `tile_indices == []` como critério de recusa, não o texto livre.
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
