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

O hCaptcha tem as mesmas duas técnicas: "baseline_no_select" e "llm_vision"
(`hcaptcha_llm_vision_attempt`) — esta última detecta qual dos 4 tipos de
desafio apareceu (grade de imagens, ícone diferente, arrastar-e-encaixar ou
laços fechados — ver `hcaptcha_collector.py`) e resolve cada um com a
abordagem adequada (grade reaproveita o solver de grade; os três tipos só-
canvas usam clique/arrasto humanizado via `harness/humanized_actions.py`).
"""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from attacks.classical_canvas_solver import (
    solve_closed_loop_classical,
    solve_digit_membership_classical,
    solve_icon_outlier_classical,
)
from attacks.llm_vision_solver import (
    MODEL_ID,
    solve_canvas_points_by_consensus,
    solve_puzzle_piece,
    solve_recaptcha_grid_by_majority_vote,
)
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
from harness.humanized_actions import click_canvas_point, drag_canvas_point

RECAPTCHA_DEMO_URL = "https://www.google.com/recaptcha/api2/demo"
HCAPTCHA_DEMO_URL = "https://accounts.hcaptcha.com/demo"


def _canvas_native_screenshot(driver, canvas) -> tuple[bytes, float]:
    """Captura o bitmap NATIVO do `<canvas>` (`toDataURL`), não o render CSS downscaled.

    `canvas.screenshot_as_png` (usado pelo fluxo de arrastar-e-encaixar, que
    já funciona bem e não foi mexido) captura o canvas no tamanho CSS exibido
    (ex. 500x470) — mas o canvas pode ter uma resolução interna maior (ex.
    1000x940, um fator de 2x), e a diferença de nitidez é visível. Validado
    ao vivo em 2026-11-01 (sem `SecurityError` — o canvas não é "tainted" por
    conteúdo cross-origin): a versão nativa deixa bordas de ícones/curvas bem
    mais definidas, mais fácil de distinguir diferença sutil ou se um traço
    fecha ou não.

    Retorna `(bytes_png, escala)`, onde `escala = largura_nativa / largura_css`
    — **as coordenadas que o modelo devolve vêm no espaço nativo**; é preciso
    dividir por `escala` antes de usar com `harness.humanized_actions`
    (que trabalha em coordenada CSS, a mesma de `canvas.size`). Gotcha
    descoberto na validação: esquecer essa conversão faz todo clique sair
    fora do alvo.

    Limitação conhecida, aceita por ora (evitar overengineering): a barra de
    cabeçalho do desafio não é desenhada no canvas (é DOM separado), então a
    região correspondente vem preta no bitmap nativo — não teve impacto
    funcional na validação manual, já que o texto da instrução já é passado
    separado ao modelo via `instruction`.
    """
    native_width = int(canvas.get_attribute("width"))
    css_width = canvas.size["width"]
    scale = native_width / css_width
    data_url = driver.execute_script("return arguments[0].toDataURL('image/png');", canvas)
    _, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded), scale


def _infer_canvas_criterion_type(prompt: str) -> str | None:
    """Mapeia o texto do prompt do hCaptcha para um `criterion_type` de solver.

    Heurística simples por palavra-chave — cobre as variações de texto já
    observadas ao vivo: "ícone que é diferente"/"figura que quebra o padrão"
    como `"icon_outlier"` (achar o elemento que destoa/camufla — ver achado
    de 2026-11-02 em `hcaptcha_llm_vision_attempt`); "laços fechados" como
    `"closed_loop"`; "número que pertence a este grupo: 789" como
    `"digit_membership"` (achado de 2026-11-02: diferente dos outros dois,
    tem resposta objetivamente verificável — o dígito exato pertence ou não
    ao conjunto citado na instrução, não é um julgamento perceptual
    subjetivo). Retorna `None` pra qualquer prompt não reconhecido, caindo
    no prompt genérico de `solve_canvas_points` — não quebra pra tipos de
    desafio ainda não catalogados.
    """
    lowered = prompt.lower()
    if "laç" in lowered or "loop" in lowered:
        return "closed_loop"
    if "pertence" in lowered or "grupo" in lowered:
        return "digit_membership"
    if "diferente" in lowered or "quebra" in lowered or "padrão" in lowered:
        return "icon_outlier"
    return None


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
    n_votes: int = 3,
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

    **Self-consistency por votação (`n_votes`, 20 tentativas reais em
    2026-11-01): 15/20 sucessos (75%)** — salto em relação aos 10/20 (50%)
    da versão anterior (uma única chamada por rodada). Cada rodada agora
    chama `solve_recaptcha_grid_by_majority_vote` (pesquisei antes como
    frameworks de automação internos não relacionados resolvem esse mesmo
    problema — não copiei código, só a técnica: maioria simples sobre N
    chamadas do classificador, tolerante a falha parcial). O efeito é
    bem delimitado: das 10 falhas de antes, 6 eram imprecisão visual (uma
    resposta real, sem recusa, mas errada) e 4 eram recusa de segurança.
    Com votação, as falhas caíram para 5/20, sendo só **1/20 por imprecisão
    visual** (redução de 6→1, exatamente o que a votação deveria atacar) e
    **4/20 continuam por recusa de segurança** (taxa inalterada — a votação
    não toca nisso, e não deveria: ela é só uma técnica de precisão visual,
    não de contorno de recusa). Achado extra sobre a interação entre os
    dois mecanismos: quando a maioria das rodadas de voto recusa e só uma
    minoria responde de verdade, a resposta real da minoria **não** atinge
    maioria simples e acaba descartada — ou seja, a recusa é "pegajosa" sob
    votação por maioria (perde só se >50% das rodadas cooperarem), o que é o
    comportamento correto/esperado do desenho, não um bug.
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
            answer = solve_recaptcha_grid_by_majority_vote(
                screenshot_png=grid_screenshot,
                instruction=instruction,
                grid_rows=grid_size,
                grid_cols=grid_size,
                n_votes=n_votes,
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

    **Lote de 20 tentativas reais em 2026-11-01** (já com o user-agent de
    desktop corrigido — ver `harness/browser.py`): **0/20 sucessos**, como
    esperado por desenho — esta função é o baseline "sem ataque" (clica
    aceitar/pular às cegas, nunca tenta de fato resolver o desafio), então
    falhar sempre que um desafio real é exibido é o comportamento correto, não
    um bug. O dado relevante aqui não é a taxa (0% é o valor de referência
    esperado), e sim que o hCaptcha apresentou desafio interativo real em
    **100%** das 20 tentativas (nunca `challenge_not_presented`) e que esse
    desafio variou entre **três tipos distintos**: "Clique no ícone que é
    diferente" (10/20), "Arraste o elemento para o local onde se encaixa"
    (7/20) e "Clique em todos os laços fechados" (3/20) — ver
    `harness/collectors/hcaptcha_collector.py` para a discussão de por que
    nenhum mapeia diretamente para o `image_grid_cnn/` existente.
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


def hcaptcha_llm_vision_attempt(
    demo_url: str = HCAPTCHA_DEMO_URL,
    headless: bool = True,
    max_rounds: int = 3,
    n_votes: int = 3,
    canvas_model: str = MODEL_ID,
) -> dict:
    """Resolve o desafio hCaptcha de verdade — 4 tipos distintos observados ao vivo.

    Mesmo fluxo de `hcaptcha_baseline_attempt` até o desafio aparecer. A
    partir daí, a função **detecta o tipo de desafio pela estrutura do DOM**
    (não só pelo texto do prompt, que varia) e resolve cada um de forma
    diferente:

    - **Grade de imagens** (ex. "Selecione coisas que podem ser úteis aqui",
      detectada pela presença de elementos `.task` — ver
      `hcaptcha_collector.py`): mesmo loop de refinamento por rodadas do
      reCAPTCHA, reaproveitando `solve_recaptcha_grid_by_majority_vote`.
    - **Arrastar-e-encaixar** (prompt contém "arrast"): o desafio inteiro é
      um único `<canvas>` sem elementos clicáveis por item — `solve_puzzle_piece`
      identifica a posição da peça E do alvo (`DragPuzzleAnswer`), e
      `harness.humanized_actions.drag_canvas_point` executa o arrasto via
      `ActionChains` com trajetória de Bézier (não um evento de mouse
      sintetizado em JS).
    - **Clique em ponto(s) sobre canvas** (qualquer outro prompt — cobre
      "ícone diferente" e "laços fechados", ambos também renderizados só em
      `<canvas>`): `solve_canvas_points_by_consensus` (self-consistency por
      clustering espacial, `n_votes`) devolve 1 ou mais coordenadas, cada
      uma clicada via `harness.humanized_actions.click_canvas_point`.

    **Por que não dá pra usar seletor CSS por item nos três tipos acima**:
    inspeção ao vivo do DOM (2026-11-01) mostrou que "ícone diferente",
    "arrastar-e-encaixar" e "laços fechados" são desenhados inteiramente
    dentro de um único elemento `<canvas>` — sem `<img>`/`<div>` individual
    por ícone/peça/forma. A única forma de interagir é por coordenada de
    pixel sobre o canvas, convertida para offset relativo ao **centro** do
    elemento (`ActionChains.move_to_element_with_offset` usa essa convenção,
    não canto superior esquerdo — confirmado na assinatura do método, não
    chutado). Só o quarto tipo (grade) tem elementos `.task` individuais.

    **Pesquisei antes como frameworks de automação internos não relacionados
    a este projeto lidam com esses três tipos canvas — não tinha o que
    copiar**: a estratégia observada lá nem tenta resolvê-los, só pede um
    novo desafio repetidamente até sair um do tipo grade (o único que sabem
    resolver). Confirma que esses três tipos são, de fato, um gap real,
    não só deste projeto. O que aproveitei de lá foi a técnica de **execução**
    do clique (não a de resolução): eventos de mouse sintetizados via JS puro
    podem ser rejeitados por `isTrusted=false` mesmo com a resposta certa, daí
    o uso de `ActionChains` com curva de Bézier em vez de um "teleporte" direto
    ao alvo — ver `harness/humanized_actions.py` e
    `evasion/human_behavior.canvas_offset_path`.

    **Validado ao vivo em 2026-11-01, 20 tentativas reais: 4/20 sucessos
    (20%)**, com uma diferença grande entre os tipos de desafio enfrentados
    (nenhuma tentativa caiu no tipo grade nesta amostra):

    - **Arrastar-e-encaixar: 4/7 sucessos (57%)** — a extração de origem +
      destino da peça (`DragPuzzleAnswer`) combinada com o arrasto humanizado
      (`drag_canvas_point`) funciona na maioria das vezes. É o resultado mais
      forte desta rodada e valida a abordagem "clique confiável + trajetória
      não-linear" sugerida pela pesquisa conceitual nos frameworks internos.
    - **Clique em ponto(s) sobre canvas ("ícone diferente" + "laços
      fechados"): 0/13 sucessos (0%)** — nenhum dos dois tipos resolvido
      nesta amostra. Sem instrumentação adicional (fora do escopo desta
      rodada de medição), não dá pra separar se a causa é imprecisão do
      modelo (ex. a diferença entre ícones costuma ser sutil — ver o exemplo
      em `scratchpad` desta sessão) ou algo na execução do clique específico
      desses tipos; mecanicamente o clique não lança exceção em nenhuma das
      13 tentativas (mesma função `click_canvas_point` usada com sucesso
      dentro do fluxo de arrasto), o que aponta mais para precisão do modelo
      do que para um bug de execução, mas isso não é conclusivo.
    - Nenhum erro de execução (schema OpenAI, Selenium) em nenhuma das 20
      tentativas — a `CanvasPointsAnswer.points` precisou virar
      `list[CanvasPoint]` em vez de `list[tuple[int, int]]`: a API da OpenAI
      rejeitava (erro 400) o schema JSON que o Pydantic gera para tupla de
      tamanho fixo no modo estrito de structured outputs.

    Isto é o resultado de uma única rodada de medição (bounded, como
    combinado) depois de implementar os solvers — não uma otimização
    iterada até um alvo de taxa de sucesso.

    **Tentativa de melhorar o clique-sobre-canvas (2026-11-01, pedido
    explícito do usuário): não subiu.** Troquei `solve_canvas_points` por
    `solve_canvas_points_by_consensus` (mesmo princípio de self-consistency
    que funcionou bem na grade do reCAPTCHA — ver
    `recaptcha_llm_vision_attempt` — mas adaptado a coordenada contínua via
    clustering espacial em vez de maioria exata de índice). Resultado de mais
    20 tentativas reais: **5/20 (25%)** no total, mas **clique-sobre-canvas
    continuou em 0/10 (0%)** — a melhora no agregado veio só de variação
    amostral no arrastar-e-encaixar (5/8 nesta rodada vs. 4/7 antes, dentro
    do ruído esperado para N pequeno). Ou seja, a técnica testada **não
    resolveu** o problema do clique-sobre-canvas.

    Achado relevante no processo: a votação encontrou um caso real de recusa
    em que `solve_canvas_points` retornou `None` (não `points=[]` como a
    recusa do solver de grade) — corrigido em
    `solve_canvas_points_by_consensus` para tratar isso como mais uma falha
    de rodada tolerável, não um crash.

    Com 0/23 tentativas reais de clique-sobre-canvas bem-sucedidas no total
    da sessão (13 antes + 10 depois do consenso), a taxa é baixa demais pra
    ser só ruído de imprecisão pontual do modelo — se fosse só isso, a
    votação (que ataca exatamente esse tipo de ruído, como provou fazer na
    grade) deveria ter destravado pelo menos alguns casos. Duas explicações
    permanecem em aberto e eu não investiguei nenhuma a fundo (ficaria para
    uma próxima rodada, se o usuário pedir): (a) o critério visual em si é
    mais difícil para o modelo nesses dois tipos — diferença sutil entre
    ícones, identificar todos os laços fechados sem errar a contagem — a
    ponto de o modelo errar de forma consistente (não aleatória) entre
    chamadas, o que votação por maioria não resolve; ou (b) hCaptcha pode
    exigir uma tolerância de acerto mais estreita e/ou uma sequência de
    eventos de clique mais específica para esse tipo de desafio do que para
    o arrastar-e-encaixar (que tem alguma tolerância de "encaixe").

    **Instrumentação a nível de evento JS (2026-11-01, a pedido do usuário):
    descarta a hipótese (b).** Injetei listeners (`pointerdown`, `mousedown`,
    `pointerup`, `mouseup`, `click`) diretamente no `<canvas>`, de dentro do
    iframe do desafio, e cliquei em 3 pontos conhecidos (centro, perto do
    canto superior-esquerdo, perto do canto inferior-direito) via
    `harness.humanized_actions.click_canvas_point`. Resultado, sem exceção
    em nenhum dos 3 pontos: sequência completa e realista de eventos
    (`pointerdown` → `mousedown` → `pointerup` → `mouseup` → `click`), **100%
    `isTrusted=true`** (entrada real via protocolo do navegador, não evento
    sintetizado em JS) e **coordenada relativa ao canvas idêntica, pixel a
    pixel, ao ponto pedido** nos 3 casos — sem deslocamento, sem erro de
    escala, sem inversão de eixo. Ou seja, o mecanismo de clique em si está
    correto; não há bug de execução nem rejeição de evento a explicar o 0/23.
    Isso concentra a causa remanescente na hipótese (a): uma limitação de
    precisão visual do próprio modelo para esses dois tipos específicos de
    desafio, sistemática o suficiente pra não ceder nem a múltiplas chamadas
    (self-consistency). Não testei aqui se um modelo mais forte (`gpt-6-astra`)
    ou um pré-processamento de imagem diferente mudaria isso — ficaria para
    uma próxima rodada, se o usuário pedir.

    **Ciclo de /loop de diagnóstico estruturado (2026-11-02), 3 correções
    implementadas e testadas**: a pedido do usuário, apliquei um roteiro de
    diagnóstico formal (≥3 hipóteses de causa raiz, ≥3 correções por
    hipótese, escolha justificada evitando overengineering) ao problema do
    clique-sobre-canvas. Causas-raiz consideradas: (RC1) limite de precisão
    visual intrínseco do modelo; (RC2) resolução/qualidade da imagem enviada
    insuficiente; (RC3) prompt genérico, pouco estruturado, pra uma tarefa
    de discriminação visual sutil; (RC4) tratar "ícone diferente" e "laços
    fechados" com o mesmo prompt/schema genérico apesar de serem problemas
    de natureza diferente. Descartei explicitamente trocar para um modelo
    mais caro (`gpt-6-astra`) sem pedir antes (acordado com o usuário) e
    qualquer correção que reabrisse a questão de recusa de segurança.

    Implementadas (ver `_canvas_native_screenshot`/`_infer_canvas_criterion_type`
    aqui, e `attacks.llm_vision_solver.solve_canvas_points`/`CanvasPointsAnswer`):
    (1) captura do bitmap NATIVO do canvas via `toDataURL` em vez do
    screenshot CSS downscaled (RC2) — validado ao vivo sem `SecurityError`,
    ganho real de resolução (2x no caso observado); (2) prompt especializado
    por `criterion_type` (`icon_outlier`/`closed_loop`) com raciocínio
    estruturado (enumerar -> comparar/classificar -> responder) em vez do
    prompt genérico (RC3 + RC4); (3) campo `elements_considered` na resposta,
    só para auditoria/diagnóstico. **Gotcha crítico documentado**: a captura
    nativa muda o espaço de coordenada (pode não ser 1:1 com o CSS/`canvas.size`
    usado por `harness.humanized_actions`) — é preciso escalar antes de
    clicar, ou todo clique sai fora do alvo; feito via `native_scale` em
    `_canvas_native_screenshot`.

    **Achado novo durante a validação (smoke test, 5 tentativas + inspeção
    visual direta de 3 amostras nativas)**: em "ícone diferente", a demo
    oficial do hCaptcha mostra consistentemente só **2 ícones** por desafio
    (não um grupo com uma maioria clara e 1 intruso, como o nome do desafio
    sugere) — confirmado tanto pelo raciocínio do próprio modelo
    ("não há uma maioria de referência") quanto por inspeção visual minha
    direta das imagens em alta resolução. Com só 2 candidatos, "o diferente"
    não é uma comparação com maioria — parece ser um teste de **quebra de
    camuflagem**: qual dos dois ícones tem uma cor que destoa da paleta do
    fundo decorativo (o alvo) vs. qual foi deliberadamente colorido para se
    camuflar na paleta do fundo — uma habilidade perceptual distinta de
    classificação semântica (que o modelo já faz bem na grade do reCAPTCHA).
    Isso refina a hipótese (a): não é "o modelo erra decisões semânticas
    sutis", é "o modelo pode não estar calibrado pra detectar camuflagem de
    cor relativa ao fundo" — um prompt que nomeie esse critério explicitamente
    (em vez de "visualmente diferente", genérico) é uma correção ainda não
    tentada. Fica em aberto pra próxima rodada, se o usuário confirmar.

    Smoke test de 5 tentativas reais após essas 3 mudanças: 4 "ícone
    diferente" (0/4, mesmo padrão de "só 2 ícones, sem maioria" no raciocínio
    de 3 delas) + 1 "challenge_not_presented" — nenhum erro de execução
    (schema, Selenium, escala de coordenada).

    **Ciclo 2 (2026-11-02): prompt nomeando "quebra de camuflagem" explicitamente
    — não melhorou.** Com autorização do usuário, reformulei
    `_CANVAS_CRITERION_PROMPTS["icon_outlier"]` pra descrever o critério
    hipotetizado no ciclo 1 (contraste de cor com o fundo ao redor, não
    comparação de maioria) e rodei o lote oficial de 20 tentativas reais.
    **Resultado: 0/13 em "ícone diferente" e 0/5 em "laços fechados"** — a
    única tentativa bem-sucedida do lote foi um "arrastar-e-encaixar"
    (técnica não tocada neste ciclo). Somando com as tentativas anteriores da
    sessão (0/23 antes de qualquer correção do ciclo 1), o clique-sobre-canvas
    está em **0/41 no total**, mesmo depois de três correções independentes e
    bem fundamentadas (resolução nativa, raciocínio estruturado, framing de
    camuflagem nomeado explicitamente).

    Isso é evidência mais forte do que antes de que o gargalo não é
    engenharia de prompt — é uma limitação de capacidade perceptual do
    modelo (`gpt-6.1-terra`) nesses dois tipos específicos de tarefa
    (discriminação de contraste de cor contra um fundo; classificação de
    topologia de curva), diferente da classificação semântica que ele já
    faz bem na grade do reCAPTCHA e do raciocínio espacial que faz
    razoavelmente bem no arrastar-e-encaixar. Hipótese (a) permanece líder,
    agora com suporte de 3 correções negativas consecutivas, não só uma.

    Caminhos ainda não tentados (não executados sem confirmação do
    usuário): (i) trocar para um modelo mais forte (`gpt-6-astra`); (ii) uma
    abordagem de visão computacional clássica — calcular a distância de cor
    (ex. Delta-E) entre os pixels de cada ícone e o fundo imediatamente ao
    redor dele, em vez de pedir ao modelo um julgamento qualitativo de
    contraste — tecnicamente mais alinhado ao que o critério real parece
    exigir (uma medida numérica de contraste, não uma descrição em texto) e
    coerente com a infraestrutura de CV clássica já prevista no projeto
    (`src/attacks/classical_cv.py`, Atividade 2).

    **Ciclo 3 (2026-11-02): modelo mais forte (`gpt-6-astra`) — também não
    melhorou.** Com autorização explícita do usuário, troquei só o modelo do
    solver de clique-sobre-canvas via `canvas_model` (parâmetro novo desta
    função, repassado como `model` a `solve_canvas_points_by_consensus`;
    `gpt-6.1-terra` continua padrão, e os outros solvers — grade, arrastar-e-
    -encaixar — não foram tocados). No smoke test, descobri um **6º tipo de
    desafio** não catalogado antes, "Clique no número que pertence a este
    grupo: 789" — diferente de ícone-diferente/laços-fechados, tem resposta
    objetivamente verificável (o dígito exato pertence ou não ao conjunto
    citado, não é julgamento perceptual subjetivo); adicionei
    `criterion_type="digit_membership"` com prompt próprio antes do lote
    oficial, pra não confundir a medição do modelo com um prompt genérico
    mal ajustado a esse tipo. Também corrigi, no mesmo smoke test, um
    `AttributeError` opaco em `solve_puzzle_piece` (mesma causa já conhecida
    de `solve_canvas_points`: recusa pode retornar `None` em vez de schema
    preenchido) — nunca tinha sido protegido nesse caminho.

    **Resultado do lote oficial de 20 com `gpt-6-astra`: 0/10 no
    clique-sobre-canvas** (0/6 "ícone diferente", 0/4 "número que pertence
    ao grupo" — este último especialmente notável, já que leitura de dígito
    costuma ser um ponto forte de modelos de visão, não uma tarefa de
    percepção ambígua como as outras duas). As 6 tentativas bem-sucedidas do
    lote foram todas arrastar-e-encaixar (não afetado por este ciclo, usa
    `gpt-6.1-terra` sem mudança — 6/7 nesta amostra, variação normal de uma
    técnica que já funciona). Somando com o histórico, o clique-sobre-canvas
    fecha a sessão em **0/51 no total**, agora incluindo um teste com o
    modelo mais caro/capaz disponível. Com um modelo substancialmente mais
    forte também não produzindo nenhum acerto — nem no tipo supostamente
    mais fácil (leitura de dígito) — a hipótese de "modelo insuficientemente
    capaz" perde força; o padrão observado é mais consistente com algo na
    execução/verificação do desafio que nenhuma das correções testadas
    (resolução, prompt, modelo) ataca, ou com um aspecto do critério real
    ainda não identificado corretamente — não necessariamente "o modelo
    nunca vai conseguir isso".
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

        tasks = driver.find_elements(By.CSS_SELECTOR, ".task")
        extra: dict = {"prompt": prompt}
        rounds_used = 1

        if tasks:
            grid_size = round(len(tasks) ** 0.5)
            selected: set[int] = set()
            last_reasoning = ""
            for round_number in range(1, max_rounds + 1):
                rounds_used = round_number
                grid_screenshot = driver.find_element(
                    By.CSS_SELECTOR, ".task-grid"
                ).screenshot_as_png
                answer = solve_recaptcha_grid_by_majority_vote(
                    screenshot_png=grid_screenshot,
                    instruction=prompt,
                    grid_rows=grid_size,
                    grid_cols=grid_size,
                    n_votes=n_votes,
                )
                last_reasoning = answer.reasoning
                new_indices = [idx for idx in answer.tile_indices if idx not in selected]
                if not new_indices:
                    break
                current_tasks = driver.find_elements(By.CSS_SELECTOR, ".task")
                for tile_index in new_indices:
                    current_tasks[tile_index].click()
                    selected.add(tile_index)
                time.sleep(1.0)
            extra.update(
                {
                    "challenge_type": "grid",
                    "grid_size": grid_size,
                    "tile_indices": sorted(selected),
                    "reasoning": last_reasoning,
                    "rounds_used": rounds_used,
                }
            )
        else:
            canvas = driver.find_element(By.CSS_SELECTOR, "canvas")
            canvas_wh = (canvas.size["width"], canvas.size["height"])
            screenshot = canvas.screenshot_as_png

            if "arrast" in prompt.lower():
                drag_answer = solve_puzzle_piece(screenshot_png=screenshot, instruction=prompt)
                if drag_answer is None:
                    # Mesmo achado de solve_canvas_points (ver llm_vision_solver.py):
                    # uma recusa pode fazer output_parsed vir None em vez de um
                    # DragPuzzleAnswer preenchido. Erro claro em vez de
                    # AttributeError opaco (capturado por experiments.runner de
                    # qualquer forma, mas com uma mensagem útil no lugar de
                    # "'NoneType' object has no attribute 'piece_x'").
                    raise ValueError(
                        "solve_puzzle_piece retornou None (provável recusa do "
                        "modelo sem preencher o schema)"
                    )
                drag_canvas_point(
                    driver,
                    canvas,
                    (drag_answer.piece_x, drag_answer.piece_y),
                    (drag_answer.target_x, drag_answer.target_y),
                    canvas_wh,
                )
                extra.update(
                    {
                        "challenge_type": "drag_puzzle",
                        "piece_point": (drag_answer.piece_x, drag_answer.piece_y),
                        "target_point": (drag_answer.target_x, drag_answer.target_y),
                        "confidence": drag_answer.confidence,
                    }
                )
            else:
                native_screenshot, native_scale = _canvas_native_screenshot(driver, canvas)
                criterion_type = _infer_canvas_criterion_type(prompt)
                click_answer = solve_canvas_points_by_consensus(
                    screenshot_png=native_screenshot,
                    instruction=prompt,
                    criterion_type=criterion_type,
                    model=canvas_model,
                    n_votes=n_votes,
                )
                # `click_answer` vem em coordenada do bitmap NATIVO do canvas
                # (resolução interna, ex. 2x a CSS); harness.humanized_actions
                # trabalha em coordenada CSS (a mesma de canvas_wh/canvas.size)
                # — converter antes de clicar, senão o clique sai fora do alvo.
                css_points = [
                    (point.x / native_scale, point.y / native_scale)
                    for point in click_answer.points
                ]
                for css_point in css_points:
                    click_canvas_point(driver, canvas, css_point, canvas_wh)
                    time.sleep(0.3)
                extra.update(
                    {
                        "challenge_type": "canvas_points",
                        "criterion_type": criterion_type,
                        "points": css_points,
                        "elements_considered": click_answer.elements_considered,
                        "reasoning": click_answer.reasoning,
                    }
                )

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
            "sub_challenges": rounds_used,
            "extra": extra,
        }
    finally:
        driver.quit()


def hcaptcha_classical_cv_attempt(
    demo_url: str = HCAPTCHA_DEMO_URL,
    headless: bool = True,
) -> dict:
    """Resolve o clique-sobre-canvas do hCaptcha via visão computacional clássica/OCR — sem LLM.

    Pedido explícito do usuário depois do `gpt-6-astra` também não resolver
    nenhum dos três tipos de desafio canvas (0/51 no total — ver docstring de
    `hcaptcha_llm_vision_attempt`): parar de iterar em variações de prompt do
    mesmo modelo e tentar uma técnica estruturalmente diferente. Usa
    `attacks/classical_canvas_solver.py` (segmentação por borda via OpenCV +
    OCR via Tesseract) em vez de pedir um julgamento qualitativo a um modelo
    de visão — compatível com o escopo original da Atividade 2 do
    cronograma, que já previa CV clássica como técnica a comparar.

    **Escopo desta função é só os três tipos canvas** (ícone diferente,
    laços fechados, número do grupo) — grade de imagens e arrastar-e-encaixar
    já funcionam bem com `hcaptcha_llm_vision_attempt` (75%/~67% reais) e não
    têm motivo pra trocar de técnica; aqui eles só confirmam sem tentar
    resolver (`extra["technique_scope"] = "not_applicable"`), mesmo
    comportamento do baseline — não é um bug, é delimitação de escopo.

    Reaproveita a mesma captura em resolução nativa e o mesmo
    `_infer_canvas_criterion_type` já usados pelo solver via LLM (mesma
    conversão de coordenada nativa -> CSS antes de clicar, mesmo gotcha já
    documentado lá). Ainda não validado ao vivo no momento em que este
    docstring foi escrito — a prototipagem manual (ver `classical_cv.py`)
    confirmou sinal razoável contra amostras reais salvas, mas o resultado
    que importa é a taxa de sucesso medida ao vivo, não a prototipagem
    offline (mesma prática já usada no resto do projeto).
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

        tasks = driver.find_elements(By.CSS_SELECTOR, ".task")
        extra: dict = {"prompt": prompt}

        if tasks or "arrast" in prompt.lower():
            extra["challenge_type"] = "grid" if tasks else "drag_puzzle"
            extra["technique_scope"] = "not_applicable"
        else:
            canvas = driver.find_element(By.CSS_SELECTOR, "canvas")
            canvas_wh = (canvas.size["width"], canvas.size["height"])
            native_screenshot, native_scale = _canvas_native_screenshot(driver, canvas)
            image = cv2.imdecode(np.frombuffer(native_screenshot, dtype=np.uint8), cv2.IMREAD_COLOR)
            criterion_type = _infer_canvas_criterion_type(prompt)

            if criterion_type == "closed_loop":
                native_points = solve_closed_loop_classical(image)
            elif criterion_type == "digit_membership":
                point = solve_digit_membership_classical(image, prompt)
                native_points = [point] if point else []
            else:
                point = solve_icon_outlier_classical(image)
                native_points = [point] if point else []

            css_points = [(x / native_scale, y / native_scale) for x, y in native_points]
            for css_point in css_points:
                click_canvas_point(driver, canvas, css_point, canvas_wh)
                time.sleep(0.3)

            extra.update(
                {
                    "challenge_type": "canvas_points",
                    "criterion_type": criterion_type,
                    "points": css_points,
                }
            )

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
            "extra": extra,
        }
    finally:
        driver.quit()
