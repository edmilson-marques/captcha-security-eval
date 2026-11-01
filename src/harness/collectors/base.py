"""Interface comum dos coletores de amostra (Atividade 1)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseCollector(ABC):
    target: str
    demo_url: str

    def __init__(self, driver):
        self.driver = driver

    @abstractmethod
    def locate_challenge(self):
        """Localiza o elemento (iframe/widget) do desafio na página carregada."""

    def collect(self, label: str | None = None) -> Path:
        """Navega até `demo_url`, localiza o desafio, tira screenshot e salva.

        Retorna o diretório da amostra salva (ver `harness.storage.save_sample`).
        """
        from harness.storage import save_sample  # import local evita ciclo

        self.driver.get(self.demo_url)
        challenge_element = self.locate_challenge()
        screenshot_png = challenge_element.screenshot_as_png
        page_html = self.driver.page_source

        return save_sample(
            target=self.target,
            source_url=self.demo_url,
            screenshot_png=screenshot_png,
            page_html=page_html,
            label=label,
        )
