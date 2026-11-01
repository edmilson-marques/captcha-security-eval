"""CLI para rodar a coleta de uma amostra (Atividade 1).

Exemplos:
    python -m harness.collect_cli --target recaptcha
    python -m harness.collect_cli --target hcaptcha --label "bus,bus,crosswalk"
    python -m harness.collect_cli --target turnstile --url https://meusite.exemplo/turnstile-demo
"""
from __future__ import annotations

import argparse

from harness.browser import BrowserConfig, build_driver
from harness.collectors.hcaptcha_collector import HcaptchaCollector
from harness.collectors.recaptcha_collector import RecaptchaCollector
from harness.collectors.turnstile_collector import TurnstileCollector

COLLECTORS = {
    "recaptcha": RecaptchaCollector,
    "hcaptcha": HcaptchaCollector,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=["recaptcha", "hcaptcha", "turnstile"])
    parser.add_argument("--url", help="URL alvo (obrigatório apenas para --target turnstile)")
    parser.add_argument("--label", help="Rótulo humano da amostra (resposta correta, se souber)")
    parser.add_argument("--headed", action="store_true", help="Roda com navegador visível (debug)")
    args = parser.parse_args()

    driver = build_driver(BrowserConfig(headless=not args.headed))
    try:
        if args.target == "turnstile":
            if not args.url:
                parser.error("--url é obrigatório para --target turnstile (não há demo oficial fixa)")
            collector = TurnstileCollector(driver, demo_url=args.url)
        else:
            collector = COLLECTORS[args.target](driver)

        sample_dir = collector.collect(label=args.label)
        print(f"Amostra salva em: {sample_dir}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
