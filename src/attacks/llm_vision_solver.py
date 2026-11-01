"""Solver de CAPTCHA baseado em modelo de visão (Atividade 2 — técnica "deep learning").

Em vez de treinar um classificador do zero (o que exigiria um dataset
rotulado em volume — ver nota em `image_grid_cnn/dataset.py`), esta técnica
envia o screenshot do desafio + a instrução original para um modelo
multimodal e recebe de volta uma resposta estruturada (JSON validado via
`client.responses.parse`/Pydantic) com o que clicar. É uma técnica distinta
das demais em `src/attacks/`: não precisa de dados de treino próprios, mas
depende de uma API paga externa.

Usa a API da OpenAI (Responses API). **Requer `OPENAI_API_KEY`** no
ambiente — não incluída neste repositório.

**Validado ao vivo em 2026-11-01** contra uma amostra real coletada em
`data/raw/recaptcha/20261101T190501120789/challenge.png` (desafio "selecione
os quadrados com motocicletas", grade 4x4): `solve_recaptcha_grid` retornou
`tile_indices=[5, 6, 9, 10]`, que bate exatamente com a área ocupada pela
motocicleta na imagem (conferido visualmente, de forma independente, como
gabarito). Custo medido da chamada: 516 tokens de entrada + 117 de saída
(80 deles de raciocínio interno do modelo) = **US$0,0011** — abaixo da
estimativa inicial. Os testes automatizados
(`tests/test_llm_vision_solver.py`) continuam usando um cliente OpenAI
mockado (não fazem chamada real/paga a cada execução da suíte); a validação
ao vivo acima foi manual, documentada aqui em vez de virar teste automatizado.

**Escolha de modelo** (confirmado em 2026-11-01 via
developers.openai.com/api/docs/pricing e /docs/models): `gpt-6.1-terra`
por padrão (configurável via `MODEL_ID`) — US$1/US$5 por milhão de tokens de
entrada/saída, bom equilíbrio custo-precisão, com janela de contexto de
1.05M e suporte a imagem via Responses API. `gpt-6-luna` (US$0.10/US$0.50)
é uma alternativa ainda mais barata — vale testar se a precisão compensa a
economia, especialmente para o caso mais simples (classificação de grade,
sem coordenadas geométricas finas). `gpt-6-astra` (flagship, US$10/US$50)
fica reservado para o caso de `terra` não ter precisão suficiente.
"""

from __future__ import annotations

import base64

from openai import OpenAI
from pydantic import BaseModel, Field

MODEL_ID = "gpt-6.1-terra"


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


def _client() -> OpenAI:
    return OpenAI()


def _image_content_block(screenshot_png: bytes) -> dict:
    encoded = base64.b64encode(screenshot_png).decode("utf-8")
    return {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}


def solve_recaptcha_grid(
    screenshot_png: bytes,
    instruction: str,
    grid_rows: int,
    grid_cols: int,
    client: OpenAI | None = None,
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
    response = client.responses.parse(
        model=MODEL_ID,
        max_output_tokens=1024,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    _image_content_block(screenshot_png),
                ],
            }
        ],
        text_format=RecaptchaGridAnswer,
    )
    return response.output_parsed


def solve_puzzle_piece(
    screenshot_png: bytes,
    instruction: str,
    client: OpenAI | None = None,
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
    response = client.responses.parse(
        model=MODEL_ID,
        max_output_tokens=1024,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    _image_content_block(screenshot_png),
                ],
            }
        ],
        text_format=PuzzlePieceAnswer,
    )
    return response.output_parsed
