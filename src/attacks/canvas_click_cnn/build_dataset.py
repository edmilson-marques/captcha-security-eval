"""Monta o dataset rotulado de "laços fechados" (hCaptcha) a partir de
amostras reais salvas em `data/raw/hcaptcha/` (Atividade 2, Nov/2026).

**Origem e limitação do rótulo, declarada explicitamente**: nenhuma
tentativa real do pipeline teve sucesso até agora, então não existe nenhum
rótulo de verdade produzido pelo próprio experimento. Os rótulos abaixo
(`LABELS`) foram atribuídos por inspeção visual direta das imagens
anotadas (candidato a candidato, com a imagem inteira do desafio como
contexto) feita pelo assistente de IA (Claude), não pelo autor do TCC —
rotulagem assistida, não 100% manual do autor, registrada aqui de forma
explícita por honestidade metodológica (ver Capítulo de Metodologia da
monografia).

**Revisão de 2026-11-04 (substitui o dataset original de 20 exemplos)**:
o usuário observou ao vivo este desafio marcando errado de novo. Diagnóstico
achou a causa raiz na SEGMENTAÇÃO (`detect_stroke_regions` com a máscara
default, por luminância) fundindo a maior parte de cada desafio real numa
única região sempre que o fundo decorativo tem textura de alto contraste —
em 4 amostras reais, incluindo um caso de **zero candidatos** detectados
(toda a imagem virou 1 blob, descartado pelo filtro de área máxima). Esse
problema contaminava tanto a extração de treino quanto a de inferência, já
que as duas reusam `detect_stroke_regions`. Corrigido com
`_closed_loop_mask` (Canny sobre o canal de saturação em vez de luminância,
ver sua docstring) — testado contra as 48 amostras reais de "laços
fechados" disponíveis: **nenhuma produz mais zero candidatos** (eram
3-11 por amostra, antes incluindo pelo menos 1 caso de zero). Como a
segmentação mudou, os índices de candidato do dataset antigo não
correspondem mais aos recortes atuais — o dataset inteiro foi re-extraído e
re-rotulado do zero (mesma metodologia de rotulagem assistida), não só
ampliado.

Dataset novo: 42 recortes (19 fechados, 23 abertos) de 11 amostras reais
(5 reaproveitadas da sessão original de 02/11, 6 novas de 04/11) — bem
mais balanceado que o original (16 fechados, 4 abertos) e quase o dobro do
tamanho. Mesmo critério de filtragem de sempre: de ~75 candidatos
inspecionados nessas 11 amostras, só os recortes limpos o bastante pra
rotular com confiança entraram em `LABELS` — o resto (fatias de ruído,
caixas fundindo dois traços, fragmentos parciais de um mesmo traço) foi
descartado, não rotulado como classe nenhuma. A segmentação por saturação
reduz a fusão mas não elimina: ainda ocorre ocasionalmente (ver README,
seção "Laços fechados"), por isso a filtragem manual continua necessária.

Esse mesmo processo **não** foi viável para "ícone diferente" (nenhuma
das 9 amostras reais rendeu um par de candidatos utilizável — o detector
por borda não acha o ícone camuflado por definição, já que camuflagem é
justamente baixo contraste) nem para "número do grupo" (0 candidatos via
OCR nas 3 amostras reais). Por isso o CNN desta sessão cobre só
"laços fechados" — ver Conclusão da monografia para o porquê dos outros
dois tipos não terem dado certo com a mesma abordagem.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from attacks.canvas_click_cnn.dataset_prep import RAW_HCAPTCHA_DIR, extract_candidates

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "labels" / "hcaptcha_closed_loop"

# (nome_da_amostra, índice_do_candidato) -> classe. Ver docstring do módulo
# sobre a origem e a limitação desses rótulos -- índices re-extraídos com
# `_closed_loop_mask` em 2026-11-04, não compatíveis com o dataset antigo.
LABELS: dict[tuple[str, int], str] = {
    ("20261102T171156010627", 4): "closed",
    ("20261102T171156010627", 2): "open",
    ("20261102T171156010627", 5): "open",
    ("20261102T171210446710", 2): "open",
    ("20261102T171210446710", 5): "open",
    ("20261102T171229621787", 1): "closed",
    ("20261102T171229621787", 3): "closed",
    ("20261102T171229621787", 0): "open",
    ("20261102T171229621787", 5): "open",
    ("20261102T171229621787", 6): "open",
    ("20261102T171244466558", 0): "closed",
    ("20261102T171244466558", 7): "closed",
    ("20261102T171244466558", 2): "open",
    ("20261102T171318476240", 2): "closed",
    ("20261102T171318476240", 6): "closed",
    ("20261102T171318476240", 0): "open",
    ("20261102T171318476240", 4): "open",
    ("20261102T171318476240", 5): "open",
    ("20261104T124420312728", 0): "closed",
    ("20261104T124420312728", 3): "closed",
    ("20261104T124420312728", 4): "closed",
    ("20261104T124420312728", 1): "open",
    ("20261104T124618245518", 3): "closed",
    ("20261104T124618245518", 2): "open",
    ("20261104T124654870498", 3): "closed",
    ("20261104T124654870498", 5): "closed",
    ("20261104T124654870498", 0): "open",
    ("20261104T124654870498", 1): "open",
    ("20261104T124654870498", 2): "open",
    ("20261104T125003297898", 2): "closed",
    ("20261104T125003297898", 3): "closed",
    ("20261104T125003297898", 5): "closed",
    ("20261104T125003297898", 0): "open",
    ("20261104T125003297898", 1): "open",
    ("20261104T125003297898", 6): "open",
    ("20261104T125304860477", 5): "open",
    ("20261104T125304860477", 6): "open",
    ("20261104T130540929584", 0): "closed",
    ("20261104T130540929584", 8): "closed",
    ("20261104T130540929584", 9): "closed",
    ("20261104T130540929584", 1): "open",
    ("20261104T130540929584", 7): "open",
}


def build(output_dir: Path = OUTPUT_DIR, raw_dir: Path = RAW_HCAPTCHA_DIR) -> dict[str, int]:
    """Recorta e salva, em `output_dir/<classe>/*.png` (layout `ImageFolder`),
    só os candidatos presentes em `LABELS`. Retorna a contagem por classe.

    Idempotente: pode ser rodado de novo a qualquer momento (ex. depois de
    rotular mais amostras reais acumuladas) sem duplicar arquivos, já que
    recria `output_dir` do zero a cada chamada.

    **Bug corrigido em 2026-11-04**: esta função dizia recriar `output_dir`
    do zero mas nunca de fato limpava o diretório -- só sobrescrevia
    arquivos de mesmo nome (`{amostra}__{índice}.png`). Quando a
    segmentação mudou (ver `dataset_prep._closed_loop_mask`) e os índices
    de candidato passaram a ser outros, os PNGs antigos (rótulos obsoletos,
    de uma segmentação diferente) ficaram no diretório ao lado dos novos em
    vez de serem substituídos -- descoberto porque o dataset final tinha 60
    arquivos em vez dos 42 esperados em `LABELS`.
    """
    shutil.rmtree(output_dir, ignore_errors=True)

    counts: dict[str, int] = {}
    cache: dict[str, list] = {}

    for (sample_name, index), label in LABELS.items():
        if sample_name not in cache:
            _criterion_type, candidates = extract_candidates(raw_dir / sample_name)
            cache[sample_name] = candidates

        candidate = next((c for c in cache[sample_name] if c.index == index), None)
        if candidate is None:
            raise ValueError(
                f"Candidato {index} não encontrado em {sample_name} -- a extração "
                "mudou desde que os rótulos foram atribuídos?"
            )

        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"{sample_name}__{index}.png"
        cv2.imwrite(str(out_path), candidate.crop)
        counts[label] = counts.get(label, 0) + 1

    return counts


if __name__ == "__main__":
    result = build()
    print(f"Dataset montado em {OUTPUT_DIR}: {result}")
