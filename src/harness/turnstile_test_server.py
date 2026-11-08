"""Servidor HTTP local para testar o Cloudflare Turnstile (Atividade 1/6).

A Cloudflare não mantém uma página de demo oficial full-page (ver nota em
`config/targets.yaml`), diferente do Google/hCaptcha. A alternativa
documentada pela própria Cloudflare para testar uma integração sem precisar
de conta/domínio próprio são as *sitekeys de teste* publicadas em
developers.cloudflare.com/turnstile/troubleshooting/testing/ — funcionam em
qualquer origem, sem precisar registrar nada.

Este módulo gera uma página HTML mínima embutindo o script oficial do
widget (`challenges.cloudflare.com/turnstile/v0/api.js`) com uma sitekey de
teste, e serve essa página localmente via `http.server`, para que o
Selenium possa navegar até ela como faria com qualquer alvo real.

**Achado ao vivo em 2026-11-02**: o widget renderiza a UI visível (texto
"Confirme que é humano", checkbox, logo) sem nenhum `<iframe>` nem elemento
acessível via `document.querySelectorAll`/`innerHTML` no DOM principal —
confirmado inspecionando `document.documentElement.outerHTML` e testando
`element.shadowRoot` em todos os elementos da página (sempre `null`).
Consistente com o uso de **Shadow DOM fechado** (técnica legítima
anti-adulteração: um shadow root em modo "closed" é inacessível via JS
externo por desenho do próprio navegador, não é um bug daqui). Por isso a
detecção de resultado usa os *callbacks* JS oficiais do Turnstile
(`data-callback`/`data-error-callback`), não inspeção de DOM — verificado
ao vivo que eles disparam corretamente para as 3 sitekeys de teste antes de
escrever `experiments.live.turnstile_behavioral_attempt`.

**Limitação importante, documentada desde o início**: sitekeys de teste
têm comportamento FIXO (sempre passa / sempre bloqueia / sempre força
desafio) — servem para validar que o pipeline de coleta/medição funciona de
verdade, não para medir a robustez real do Turnstile (que reage a sinais
comportamentais genuínos do navegador, não a uma sitekey de teste). Medir
robustez real exigiria uma sitekey Cloudflare própria do aluno em modo
Managed, como já documentado em `config/targets.yaml` — passo que só o
aluno pode fazer (precisa de conta/domínio Cloudflare próprios).

**Achado em 2026-11-04**: a página oficial de teste da Cloudflare documenta
**5** sitekeys de teste, não só as 3 originalmente usadas neste projeto —
há também um par em modo *widget invisível*
(`1x00000000000000000000BB` sempre passa, `2x00000000000000000000BB`
sempre bloqueia), mais relevante pro escopo deste estudo do que as 3
visíveis, já que o Turnstile real opera majoritariamente invisível. Essa
página (`PAGE_TEMPLATE`) já funciona com elas sem nenhuma mudança — o
widget invisível não renderiza UI, mas ainda dispara os mesmos callbacks
JS. Validado ao vivo (`turnstile_behavioral_attempt`, sem alterar nenhum
código): `1x...BB` -> `success` (mesmo token `"XXXX.DUMMY.TOKEN.XXXX"`);
`2x...BB` -> `blocked` (mesmo `error_code` `"600010"` da variante
visível). Ver `config/targets.yaml` para as 5 sitekeys catalogadas.
"""

from __future__ import annotations

import functools
import http.server
import tempfile
import threading
from pathlib import Path

PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Turnstile test page (captcha-security-eval, uso acadêmico privado)</title>
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
<script>
window.__turnstileResult = {{success: null, error: null, expired: null}};
function __turnstileOnSuccess(token) {{ window.__turnstileResult.success = token; }}
function __turnstileOnError(errorCode) {{ window.__turnstileResult.error = errorCode; }}
function __turnstileOnExpired() {{ window.__turnstileResult.expired = true; }}
</script>
</head>
<body>
<form id="turnstile-form">
<div class="cf-turnstile"
     data-sitekey="{sitekey}"
     data-callback="__turnstileOnSuccess"
     data-error-callback="__turnstileOnError"
     data-expired-callback="__turnstileOnExpired"></div>
</form>
</body>
</html>
"""


class TurnstileTestServer:
    """Serve uma página local com o widget Turnstile para uma `sitekey` dada.

    Uso como *context manager* — garante que o servidor HTTP é desligado
    mesmo se a navegação/coleta falhar no meio:

        with TurnstileTestServer(sitekey) as url:
            driver.get(url)
            ...
    """

    def __init__(self, sitekey: str, host: str = "127.0.0.1"):
        self.sitekey = sitekey
        self.host = host
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        self._httpd: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="turnstile_test_")
        html_path = Path(self._tmp_dir.name) / "index.html"
        html_path.write_text(PAGE_TEMPLATE.format(sitekey=self.sitekey), encoding="utf-8")

        # Suprime o log de acesso padrão (uma linha por request) — ruído
        # irrelevante pra um servidor de teste local efêmero.
        class _QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

        quiet_handler = functools.partial(_QuietHandler, directory=self._tmp_dir.name)
        self._httpd = http.server.ThreadingHTTPServer((self.host, 0), quiet_handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

        port = self._httpd.server_address[1]
        return f"http://{self.host}:{port}/"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *_exc_info) -> None:
        self.stop()
