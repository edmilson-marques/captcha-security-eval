"""Coletor de amostras do Cloudflare Turnstile (Atividade 1).

Diferente do reCAPTCHA/hCaptcha, a Cloudflare não mantém uma página de demo
oficial full-page (ver nota em `config/targets.yaml`). Por isso este coletor
recebe a URL alvo explicitamente — deve apontar para uma página própria do
aluno com uma sitekey Cloudflare real (modo Managed), conforme validado com
o orientador antes da coleta.
"""
from __future__ import annotations

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from harness.collectors.base import BaseCollector


class TurnstileCollector(BaseCollector):
    target = "turnstile"

    def __init__(self, driver, demo_url: str):
        super().__init__(driver)
        self.demo_url = demo_url

    def locate_challenge(self):
        wait = WebDriverWait(self.driver, 15)
        return wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "iframe[src*='challenges.cloudflare.com']")
            )
        )
