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
sitekey de demo do hCaptcha não é consistentemente um único tipo. Um lote de
20 tentativas reais via `experiments.live.hcaptcha_baseline_attempt`
(2026-11-01) mostrou **três tipos distintos**; exploração ao vivo adicional
revelou um **quarto tipo** (mais raro, não apareceu nesse lote de 20):

- **"Clique no ícone que é diferente"** (10/20 no lote — o mais comum): um
  pequeno conjunto de ícones onde um destoa visualmente dos demais;
  renderizado inteiramente em `<canvas>` (sem `<img>`/`<div>` por ícone) —
  interação só é possível por coordenada de pixel.
- **"Arraste o elemento para o local onde se encaixa"** (7/20 no lote):
  quebra-cabeça de arrastar-e-encaixar, também só em `<canvas>` — pede
  identificar a posição de origem (peça solta) e destino (recorte no fundo).
- **"Clique em todos os laços fechados"** (3/20 no lote): seleção de formas
  por critério geométrico (laço fechado vs. aberto), também só em `<canvas>`.
- **"Selecione coisas que podem ser úteis aqui"** (não observado nesse lote
  de 20, encontrado em exploração à parte): esse é o único dos quatro que
  usa DOM real por item — uma `div.task-grid` com 9 `div.task[aria-label="Imagem
  do desafio N"]`, mesma mecânica de grade do reCAPTCHA v2 (`aria-pressed`
  marca seleção em vez de classe CSS). Único tipo que reaproveita
  `attacks/image_grid_cnn/`/`llm_vision_solver.solve_recaptcha_grid*`
  diretamente.

Os três tipos canvas (ícone diferente, arrastar, laços fechados) não mapeiam
para `src/attacks/image_grid_cnn/` (desenhado para seleção em grade de fotos
por categoria semântica, com elemento DOM por tile) — cada um pediria uma
técnica de ataque distinta (classificação de anomalia visual entre poucos
ícones; correlação de template geométrico; detecção de topologia de forma).
Ver `experiments.live.hcaptcha_llm_vision_attempt` para os solvers
implementados (baseados em LLM de visão, não CV clássica) e os resultados
reais de cada tipo. Documentar essa diversidade de desafio é, em si, um
resultado relevante do estudo comparativo: o hCaptcha não tem um único
"formato" de desafio visual como o reCAPTCHA v2 tem (grade de fotos).

Também relevante: nas 20 tentativas, o hCaptcha apresentou desafio
interativo real em **100%** dos casos (nunca `challenge_not_presented`) para
o fingerprint testado (Chrome headless com user-agent de desktop — ver
`harness/browser.py`) — diferente do que se poderia esperar se o motor de
risco julgasse esse tráfego como baixo-risco o suficiente para pular o
desafio.
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
