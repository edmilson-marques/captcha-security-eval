"""Coletor de amostras do Google reCAPTCHA v2 (Atividade 1).

Demo oficial do Google: https://www.google.com/recaptcha/api2/demo
"""
from __future__ import annotations

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from harness.collectors.base import BaseCollector


class RecaptchaCollector(BaseCollector):
    target = "recaptcha"
    demo_url = "https://www.google.com/recaptcha/api2/demo"

    def locate_challenge(self):
        wait = WebDriverWait(self.driver, 15)
        return wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[src*='recaptcha']"))
        )
