"""Fábrica de WebDriver Selenium usada pelos coletores (Atividade 1)."""

from __future__ import annotations

from dataclasses import dataclass

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from evasion.user_agent_pool import DESKTOP_USER_AGENTS, random_user_agent


@dataclass
class BrowserConfig:
    headless: bool = True
    window_size: tuple[int, int] = (1366, 900)
    user_agent: str | None = None
    implicit_wait_seconds: int = 5


def build_driver(config: BrowserConfig) -> webdriver.Chrome:
    """Cria um WebDriver Chrome configurado conforme `config`.

    **Achado ao vivo em 2026-11-01**: o Chrome headless anuncia
    `"HeadlessChrome"` literalmente em `navigator.userAgent` por padrão —
    um sinal de automação trivial para qualquer motor de risco (confirmado
    via `driver.execute_script("return navigator.userAgent")` contra a demo
    real do Google, depois de uma sequência de tentativas com 0% de sucesso
    cuja causa provável é essa, não a precisão da técnica de ataque em si).
    Por isso, quando `headless=True` e nenhum `user_agent` explícito é
    passado, usa um user-agent real do pool de evasão da Atividade 3
    (`src/evasion/user_agent_pool.py`) como higiene mínima — esse módulo
    existia desde aquele commit mas nunca tinha sido realmente conectado ao
    harness.

    **Correção no mesmo dia**: a primeira versão sorteava de
    `DEFAULT_USER_AGENTS` (inclui 2 entradas mobile, Android/iOS). Como
    `window_size` aqui é sempre de desktop, sortear um user-agent mobile
    criava uma inconsistência de fingerprint pior que não randomizar nada
    (ex.: UA de "Pixel 8" Android com viewport 1366x900) — qualquer motor de
    risco minimamente sofisticado cruza esses dois sinais. Por isso usa-se
    `DESKTOP_USER_AGENTS` (subconjunto só desktop), não o pool completo.
    Outros sinais de automação (fingerprint de canvas/WebGL, comportamento
    de mouse) continuam fora do escopo desta função.
    """
    options = Options()
    if config.headless:
        options.add_argument("--headless=new")
    options.add_argument(f"--window-size={config.window_size[0]},{config.window_size[1]}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    # Necessário neste ambiente em container (sem isolamento de usuário do
    # Chrome disponível) — ver docs do Chromium sobre zygote sandbox em
    # containers/VMs.
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    user_agent = config.user_agent
    if user_agent is None and config.headless:
        user_agent = random_user_agent(DESKTOP_USER_AGENTS)
    if user_agent:
        options.add_argument(f"--user-agent={user_agent}")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.implicitly_wait(config.implicit_wait_seconds)
    return driver
