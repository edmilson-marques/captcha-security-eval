"""Solver de CAPTCHA baseado em modelo de visão (Atividade 2 — técnica "deep learning").

Em vez de treinar um classificador do zero (o que exigiria um dataset
rotulado em volume — ver nota em `image_grid_cnn/dataset.py`), esta técnica
envia o screenshot do desafio + a instrução original para um modelo
multimodal da Anthropic e recebe de volta uma resposta estruturada (JSON
validado via `output_config.format`/`client.messages.parse`) com o que
clicar. É uma técnica distinta das demais em `src/attacks/`: não precisa de
dados de treino próprios, mas depende de uma API paga externa.

**Requer `ANTHROPIC_API_KEY`** no ambiente (ou um perfil via `ant auth
login`) — não incluído neste repositório. **Não foi validada ao vivo nesta
sessão** (sem chave de API disponível) — os testes (`tests/test_llm_vision_solver.py`)
usam um cliente Anthropic mockado, cobrindo o contrato de chamada, não a
qualidade real das respostas do modelo. Antes de usar em experimentos de
verdade, validar manualmente contra amostras reais coletadas em
`data/raw/`.

**Escolha de modelo**: `claude-sonnet-5` por padrão (configurável via
`MODEL_ID`) — bom equilíbrio entre custo (~US$2–3 por milhão de tokens de
entrada, preço promocional vigente até 2026-08-31) e precisão em tarefas de
classificação/coordenada visual, bem mais barato que `claude-opus-5`
(~US$5/milhão). `claude-haiku-4-5` (~US$1/milhão) é uma alternativa ainda
mais barata — vale testar se a precisão compensa a economia, especialmente
para o caso mais simples (classificação de grade, sem coordenadas
geométricas finas). Custo estimado por tentativa (screenshot pequeno de
~300-500px recortado só na área do desafio + prompt curto, pensamento
desabilitado para não gastar tokens à toa numa tarefa de classificação
direta): da ordem de poucos milésimos de dólar por chamada — meça com
`client.messages.count_tokens(...)` contra as amostras reais do projeto
antes de rodar em lote.
"""

from __future__ import annotations

import base64

import anthropic
from pydantic import BaseModel, Field

MODEL_ID = "claude-sonnet-5"


class RecaptchaGridAnswer(BaseModel):
    """Resposta estruturada para um desafio de grade (reCAPTCHA v2 / hCaptcha)."""

    tile_indices: list[int] = Field(
        description=(
            "Índices (0-based, esquerda->direita, cima->baixo) dos tiles que "
            "contêm o alvo da instrução. Lista vazia se nenhum tile corresponder."
        )
    )
    reasoning: str = Field(description="Justificativa breve da escolha, em uma frase.")


class PuzzlePieceAnswer(BaseModel):
    """Resposta estruturada para um desafio de arrastar-e-encaixar (ex. hCaptcha)."""

    target_x: int = Field(
        description="Coordenada X (pixels, relativa ao canto superior esquerdo da imagem) onde soltar a peça."
    )
    target_y: int = Field(
        description="Coordenada Y (pixels, relativa ao canto superior esquerdo da imagem) onde soltar a peça."
    )
    confidence: str = Field(description="Confiança na resposta: 'high', 'medium' ou 'low'.")


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _image_block(screenshot_png: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(screenshot_png).decode("utf-8"),
        },
    }


def solve_recaptcha_grid(
    screenshot_png: bytes,
    instruction: str,
    grid_rows: int,
    grid_cols: int,
    client: anthropic.Anthropic | None = None,
) -> RecaptchaGridAnswer:
    """Pede ao modelo quais células de uma grade correspondem à instrução do desafio.

    `screenshot_png` deve conter só a área da grade de imagens (sem o
    cabeçalho de instrução nem os botões), para reduzir tokens de imagem e
    ambiguidade — a instrução textual já é passada separadamente em
    `instruction`.
    """
    client = client or _client()
    prompt = (
        f"Esta imagem mostra uma grade de {grid_rows}x{grid_cols} células "
        f"(numeradas de 0 a {grid_rows * grid_cols - 1}, da esquerda para a "
        f"direita e de cima para baixo) de um desafio CAPTCHA. Instrução "
        f'original do desafio: "{instruction}". Identifique os índices de '
        f"todas as células que correspondem à instrução."
    )
    response = client.messages.parse(
        model=MODEL_ID,
        max_tokens=1024,
        thinking={"type": "disabled"},
        messages=[
            {
                "role": "user",
                "content": [_image_block(screenshot_png), {"type": "text", "text": prompt}],
            }
        ],
        output_format=RecaptchaGridAnswer,
    )
    return response.parsed_output


def solve_puzzle_piece(
    screenshot_png: bytes,
    instruction: str,
    client: anthropic.Anthropic | None = None,
) -> PuzzlePieceAnswer:
    """Pede ao modelo onde soltar a peça de um desafio de arrastar-e-encaixar."""
    client = client or _client()
    prompt = (
        "Esta imagem mostra um desafio CAPTCHA de arrastar-e-encaixar (drag "
        f'and drop). Instrução original: "{instruction}". Identifique as '
        "coordenadas de pixel (x, y), relativas ao canto superior esquerdo "
        "da imagem, onde a peça destacada deve ser solta para se encaixar "
        "corretamente na imagem de fundo."
    )
    response = client.messages.parse(
        model=MODEL_ID,
        max_tokens=1024,
        thinking={"type": "disabled"},
        messages=[
            {
                "role": "user",
                "content": [_image_block(screenshot_png), {"type": "text", "text": prompt}],
            }
        ],
        output_format=PuzzlePieceAnswer,
    )
    return response.parsed_output
