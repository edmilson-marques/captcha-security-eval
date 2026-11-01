"""Fábrica de WebDriver Selenium usada pelos coletores (Atividade 1)."""

from __future__ import annotations

from dataclasses import dataclass

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


@dataclass
class BrowserConfig:
    headless: bool = True
    window_size: tuple[int, int] = (1366, 900)
    user_agent: str | None = None
    implicit_wait_seconds: int = 5


def build_driver(config: BrowserConfig) -> webdriver.Chrome:
    """Cria um WebDriver Chrome configurado conforme `config`.

    Mantido deliberadamente simples (sem stealth/fingerprint) nesta etapa —
    os perfis de evasão (user-agent, fingerprint, comportamento) entram na
    Atividade 3, em `src/evasion/`.
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
    if config.user_agent:
        options.add_argument(f"--user-agent={config.user_agent}")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.implicitly_wait(config.implicit_wait_seconds)
    return driver
