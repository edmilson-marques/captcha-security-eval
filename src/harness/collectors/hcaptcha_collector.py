"""Coletor de amostras do hCaptcha (Atividade 1).

Demo oficial da Intuition Machines: https://accounts.hcaptcha.com/demo
"""
from __future__ import annotations

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from harness.collectors.base import BaseCollector


class HcaptchaCollector(BaseCollector):
    target = "hcaptcha"
    demo_url = "https://accounts.hcaptcha.com/demo"

    def locate_challenge(self):
        wait = WebDriverWait(self.driver, 15)
        return wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[src*='hcaptcha']"))
        )
