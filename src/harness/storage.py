"""Persistência das amostras coletadas (Atividade 1).

Cada amostra vira uma pasta `data/raw/<alvo>/<timestamp>/` contendo:
  - challenge.png   (screenshot da área do desafio)
  - page.html       (HTML renderizado no momento da captura)
  - metadata.json   (timestamp, URL, alvo, rótulo humano opcional)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass
class SampleMetadata:
    target: str
    source_url: str
    collected_at: str
    label: str | None = None
    extra: dict = field(default_factory=dict)


def _sample_dir(target: str, collected_at: datetime) -> Path:
    stamp = collected_at.strftime("%Y%m%dT%H%M%S%f")
    sample_dir = DATA_ROOT / target / stamp
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def save_sample(
    target: str,
    source_url: str,
    screenshot_png: bytes,
    page_html: str,
    label: str | None = None,
    extra: dict | None = None,
) -> Path:
    """Grava uma amostra bruta coletada e retorna o diretório criado."""
    collected_at = datetime.now(UTC)
    sample_dir = _sample_dir(target, collected_at)

    (sample_dir / "challenge.png").write_bytes(screenshot_png)
    (sample_dir / "page.html").write_text(page_html, encoding="utf-8")

    metadata = SampleMetadata(
        target=target,
        source_url=source_url,
        collected_at=collected_at.isoformat(),
        label=label,
        extra=extra or {},
    )
    (sample_dir / "metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return sample_dir


def save_attempt_capture(
    target: str,
    technique: str,
    screenshot_png: bytes,
    outcome: str,
    extra: dict | None = None,
) -> Path:
    """Persiste o screenshot de uma tentativa real de `experiments/live.py`.

    Complementa `save_sample` (Atividade 1 — coleta manual isolada, com
    rótulo humano e HTML da página). Aqui o objetivo é outro: acumular, ao
    longo de lotes reais de experimento, um repertório de imagens de
    desafio com o resultado verificado pelo próprio widget (sucesso/falha),
    para no futuro avaliar se um tipo de desafio recicla um conjunto finito
    de padrões visuais (o que viabilizaria casamento de padrão) — pergunta
    em aberto até que esse acúmulo exista (ver "Trabalhos futuros" na
    monografia). Reaproveita o mesmo layout de diretório de `save_sample`
    (`data/raw/<alvo>/<timestamp>/`), sem gravar `page.html` — não
    necessário para esse propósito, e evitar esse custo por tentativa
    importa quando se acumula um lote de centenas.
    """
    collected_at = datetime.now(UTC)
    sample_dir = _sample_dir(target, collected_at)

    (sample_dir / "challenge.png").write_bytes(screenshot_png)

    metadata = {
        "target": target,
        "technique": technique,
        "outcome": outcome,
        "collected_at": collected_at.isoformat(),
        "extra": extra or {},
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return sample_dir
