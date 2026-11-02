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

**Extensões de 2026-11-01** (para cobrir os desafios do hCaptcha além da
grade — ver `harness/collectors/hcaptcha_collector.py`): `solve_canvas_points`
resolve desafios de clique livre (1 ou mais pontos) sobre um `<canvas>` sem
elementos DOM individuais (ex. "ícone diferente", "laços fechados");
`solve_puzzle_piece` passou a pedir também a posição de origem da peça
(`DragPuzzleAnswer`, antes só tinha o destino), necessária pra um arrasto de
verdade via `harness/humanized_actions.py`; `solve_recaptcha_grid_by_majority_vote`
reduz imprecisão visual chamando o classificador N vezes e combinando por
maioria simples, técnica equivalente à usada em frameworks de automação
internos não relacionados a este projeto (pesquisa conceitual, sem copiar
código). `solve_canvas_points_by_consensus` tenta a mesma ideia adaptada pra
coordenada contínua (clustering espacial em vez de maioria exata) — **testado
ao vivo em 2026-11-01 e não melhorou a taxa de sucesso de clique-sobre-canvas
(0/10, igual a antes do consenso)**, ver docstring de
`experiments.live.hcaptcha_llm_vision_attempt` para a discussão completa.
"""

from __future__ import annotations

import base64
import collections

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


class DragPuzzleAnswer(BaseModel):
    """Resposta estruturada para um desafio de arrastar-e-encaixar (ex. hCaptcha).

    Antes chamada `PuzzlePieceAnswer` e com só `target_x`/`target_y`: faltava
    a posição de origem da peça, necessária pra executar um arrasto de
    verdade (clicar-e-segurar na peça antes de mover até o alvo) — ver
    `harness/humanized_actions.drag_canvas_point`.
    """

    piece_x: int = Field(
        description="Coordenada X (pixels, canto superior esquerdo da imagem) da peça solta a ser arrastada."
    )
    piece_y: int = Field(
        description="Coordenada Y (pixels, canto superior esquerdo da imagem) da peça solta a ser arrastada."
    )
    target_x: int = Field(
        description="Coordenada X (pixels, canto superior esquerdo da imagem) onde soltar a peça."
    )
    target_y: int = Field(
        description="Coordenada Y (pixels, canto superior esquerdo da imagem) onde soltar a peça."
    )
    confidence: str = Field(description="Confiança na resposta: 'high', 'medium' ou 'low'.")


class CanvasPoint(BaseModel):
    """Um ponto de clique em pixels, canto superior esquerdo da imagem como origem.

    Objeto dedicado (em vez de `tuple[int, int]`) porque a API da OpenAI
    rejeita o schema JSON que o Pydantic gera para tupla de tamanho fixo
    (`items` ausente/`prefixItems`, incompatível com o modo estrito de
    structured outputs) — achado ao vivo em 2026-11-01, erro 400 "Invalid
    schema for response_format ... array schema missing items".
    """

    x: int = Field(description="Coordenada X em pixels.")
    y: int = Field(description="Coordenada Y em pixels.")


class CanvasPointsAnswer(BaseModel):
    """Resposta estruturada para desafios de clique em ponto(s) sobre um `<canvas>`.

    Cobre tanto desafios de clique único (ex. "clique no ícone que é
    diferente") quanto de seleção múltipla (ex. "clique em todos os laços
    fechados") com o mesmo schema — a diferença é só o tamanho de `points`.
    """

    elements_considered: int = Field(
        description=(
            "Quantidade de elementos distintos (ícones/formas/curvas) que "
            "você enumerou na imagem antes de decidir a resposta — serve só "
            "para auditoria, não precisa bater com `points`."
        )
    )
    points: list[CanvasPoint] = Field(
        description=(
            "Lista de pontos a clicar, relativos ao canto superior esquerdo da "
            "imagem. Uma lista com um único ponto para desafios de clique "
            "único; vários pontos para desafios de seleção múltipla. Lista "
            "vazia se nada corresponder."
        )
    )
    reasoning: str = Field(description="Justificativa breve da escolha, em uma frase.")


def _client() -> OpenAI:
    return OpenAI()


def _image_content_block(screenshot_png: bytes) -> dict:
    encoded = base64.b64encode(screenshot_png).decode("utf-8")
    return {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}


def _strip_ui_instructions(instruction: str) -> str:
    """Remove instruções de interface irrelevantes ao critério de seleção.

    O texto bruto extraído da página inclui frases como 'Clique em
    "Confirmar" quando não houver mais nenhuma.' ou 'Se não houver nenhum,
    clique em "Pular".' — isso é instrução de botão para um usuário humano,
    não faz parte do critério de classificação em si. Mantém só a parte
    anterior a essas frases (não é reformulação da tarefa, só remoção de
    ruído que não muda o significado do critério).
    """
    return instruction.split("Clique em")[0].split("Se não houver")[0].strip()


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
    `instruction` (o prompt usa só a parte relevante ao critério de seleção,
    ver `_strip_ui_instructions`).
    """
    client = client or _client()
    criterion = _strip_ui_instructions(instruction)
    prompt = (
        f"Esta imagem mostra uma grade de {grid_rows}x{grid_cols} células "
        f"(numeradas de 0 a {grid_rows * grid_cols - 1}, da esquerda para a "
        f"direita e de cima para baixo) de um desafio CAPTCHA. Instrução "
        f'original do desafio: "{criterion}". Identifique os índices de '
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
) -> DragPuzzleAnswer:
    """Pede ao modelo a origem e o destino da peça de um desafio de arrastar-e-encaixar."""
    client = client or _client()
    prompt = (
        "Esta imagem mostra um desafio CAPTCHA de arrastar-e-encaixar (drag "
        f'and drop). Instrução original: "{instruction}". Identifique (1) as '
        "coordenadas de pixel (x, y) da peça solta, destacada/em destaque, que "
        "precisa ser arrastada, e (2) as coordenadas de pixel (x, y) do local "
        "(recorte/contorno) onde essa peça deve ser solta para se encaixar "
        "corretamente na imagem de fundo. Ambas relativas ao canto superior "
        "esquerdo da imagem."
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
        text_format=DragPuzzleAnswer,
    )
    return response.output_parsed


_CANVAS_CRITERION_PROMPTS = {
    "icon_outlier": (
        "Esta imagem mostra um pequeno conjunto de ícones/formas sobre um "
        'fundo decorativo. Instrução original: "{criterion}". Siga este '
        "raciocínio antes de responder: (1) enumere cada ícone/forma "
        "distinto que você vê, com uma descrição breve de cor e formato de "
        "cada um; (2) compare os ícones enumerados entre si; (3) identifique "
        "qual deles é visualmente diferente dos demais (cor, formato ou "
        "padrão diferente — não é sobre a posição, é sobre a aparência). "
        "Retorne as coordenadas de pixel (x, y), relativas ao canto superior "
        "esquerdo da imagem, do centro do ícone diferente."
    ),
    "closed_loop": (
        "Esta imagem mostra um conjunto de curvas/traçados desenhados sobre "
        'um fundo decorativo. Instrução original: "{criterion}". Siga este '
        "raciocínio antes de responder: (1) enumere cada curva/traçado "
        "distinto que você vê; (2) para cada um, decida se ele é um laço "
        "FECHADO (a linha forma uma forma fechada, sem pontas soltas — "
        "começa e termina no mesmo lugar, delimitando uma área) ou um laço "
        "ABERTO (tem pelo menos uma ponta solta/lacuna visível, não delimita "
        "uma área fechada). Retorne as coordenadas de pixel (x, y), "
        "relativas ao canto superior esquerdo da imagem, do centro de cada "
        "curva classificada como FECHADA. Lista vazia se nenhuma for fechada."
    ),
}


def solve_canvas_points(
    screenshot_png: bytes,
    instruction: str,
    criterion_type: str | None = None,
    client: OpenAI | None = None,
) -> CanvasPointsAnswer:
    """Pede ao modelo em quais pontos clicar num desafio renderizado em `<canvas>`.

    Cobre desafios de clique único (ex. "ícone diferente") e de seleção
    múltipla (ex. "laços fechados") — não há grade de células aqui, a
    resposta é em coordenadas de pixel livres, não índices de tile.

    `criterion_type` seleciona um prompt especializado por natureza da
    tarefa (`"icon_outlier"` — comparação visual pareada; `"closed_loop"` —
    classificação de topologia de curva) em vez do prompt genérico anterior.
    **Achado em 2026-11-01**: tratar os dois tipos com o mesmo prompt
    genérico coincidiu com 0/23 sucessos reais mesmo com self-consistency —
    o raciocínio estruturado por tipo (enumerar -> comparar/classificar ->
    responder) é uma tentativa de correção, ainda não validada ao vivo.
    `criterion_type=None` (ou um valor não reconhecido) cai no prompt
    genérico original, mantendo compatibilidade com tipos de desafio ainda
    não catalogados.
    """
    client = client or _client()
    criterion = _strip_ui_instructions(instruction)
    template = _CANVAS_CRITERION_PROMPTS.get(criterion_type) if criterion_type else None
    if template:
        prompt = template.format(criterion=criterion)
    else:
        prompt = (
            "Esta imagem mostra um desafio CAPTCHA desenhado livremente (não é "
            f'uma grade de células). Instrução original: "{criterion}". '
            "Enumere os elementos distintos que você vê antes de responder. "
            "Identifique as coordenadas de pixel (x, y), relativas ao canto "
            "superior esquerdo da imagem, de cada ponto que deve ser clicado "
            "para satisfazer a instrução. Pode ser um único ponto ou vários."
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
        text_format=CanvasPointsAnswer,
    )
    return response.output_parsed


def solve_canvas_points_by_consensus(
    screenshot_png: bytes,
    instruction: str,
    criterion_type: str | None = None,
    n_votes: int = 3,
    cluster_radius: float = 40.0,
    client: OpenAI | None = None,
) -> CanvasPointsAnswer:
    """Chama `solve_canvas_points` `n_votes` vezes e combina por consenso espacial.

    Mesmo princípio de `solve_recaptcha_grid_by_majority_vote` (self-
    consistency: várias chamadas do mesmo classificador, maioria simples
    decide), mas adaptado para coordenadas de pixel contínuas em vez de
    índices discretos de tile — duas chamadas raramente concordam pixel a
    pixel, então "maioria exata" não funcionaria aqui. Em vez disso:

    1. Junta todos os pontos de todas as rodadas válidas numa lista só.
    2. Agrupa por proximidade: cada ponto entra no primeiro cluster cujo
       centróide atual esteja a até `cluster_radius` pixels de distância;
       senão vira um cluster novo (clustering guloso de passagem única,
       suficiente aqui — não precisa de k-means, os cliques tendem a vir bem
       separados ou bem próximos, sem caso ambíguo no meio).
    3. Mantém só os clusters com suporte de **mais da metade** das rodadas
       válidas (mesmo critério de maioria da votação de grade) e devolve o
       centróide de cada um como o ponto final.

    Tolerante a falha parcial (mesmo comportamento de
    `solve_recaptcha_grid_by_majority_vote`): se uma chamada falhar, a
    votação segue com as demais; só propaga o erro se todas falharem.
    `n_votes=1` desliga o consenso (chamada única, sem custo extra).

    `criterion_type` é repassado direto a cada chamada de
    `solve_canvas_points` (ver lá o prompt especializado por tipo). O
    `elements_considered` da resposta final é a média arredondada entre as
    rodadas válidas — só diagnóstico, não participa da lógica de consenso.

    **Achado ao vivo em 2026-11-01**: ao contrário da recusa de segurança
    observada em `solve_recaptcha_grid` (que preenche o schema com
    `tile_indices=[]` em vez de quebrar), uma recusa em `solve_canvas_points`
    às vezes faz `response.output_parsed` vir `None` (o modelo não preenche o
    schema de jeito nenhum) — descoberto porque uma rodada real quebrou com
    `'NoneType' object has no attribute 'points'`. Tratado aqui como mais uma
    forma de falha de rodada (mesmo balde dos erros de API), não uma falha
    fatal da função inteira.
    """
    if n_votes < 1:
        raise ValueError("n_votes deve ser >= 1.")

    answers: list[CanvasPointsAnswer] = []
    last_error: Exception | None = None
    for _ in range(n_votes):
        try:
            answer = solve_canvas_points(
                screenshot_png=screenshot_png,
                instruction=instruction,
                criterion_type=criterion_type,
                client=client,
            )
            if answer is None:
                raise ValueError(
                    "solve_canvas_points retornou None (provável recusa do "
                    "modelo sem preencher o schema)"
                )
            answers.append(answer)
        except Exception as exc:  # noqa: BLE001 - qualquer rodada pode falhar isoladamente
            last_error = exc

    if not answers:
        assert last_error is not None
        raise last_error

    clusters: list[list[tuple[int, int]]] = []
    for answer in answers:
        for point in answer.points:
            pt = (point.x, point.y)
            match = None
            for cluster in clusters:
                centroid_x = sum(p[0] for p in cluster) / len(cluster)
                centroid_y = sum(p[1] for p in cluster) / len(cluster)
                distance = ((pt[0] - centroid_x) ** 2 + (pt[1] - centroid_y) ** 2) ** 0.5
                if distance <= cluster_radius:
                    match = cluster
                    break
            if match is not None:
                match.append(pt)
            else:
                clusters.append([pt])

    majority_threshold = len(answers) / 2
    winning_points = [
        CanvasPoint(
            x=round(sum(p[0] for p in cluster) / len(cluster)),
            y=round(sum(p[1] for p in cluster) / len(cluster)),
        )
        for cluster in clusters
        if len(cluster) > majority_threshold
    ]
    combined_reasoning = " | ".join(answer.reasoning for answer in answers)
    avg_elements_considered = round(
        sum(answer.elements_considered for answer in answers) / len(answers)
    )
    return CanvasPointsAnswer(
        points=winning_points,
        reasoning=combined_reasoning,
        elements_considered=avg_elements_considered,
    )


def solve_recaptcha_grid_by_majority_vote(
    screenshot_png: bytes,
    instruction: str,
    grid_rows: int,
    grid_cols: int,
    n_votes: int = 3,
    client: OpenAI | None = None,
) -> RecaptchaGridAnswer:
    """Chama `solve_recaptcha_grid` `n_votes` vezes e combina por maioria simples.

    Técnica de "self-consistency": em vez de confiar numa única chamada
    (sensível a imprecisão visual pontual do modelo — ver achado nas 20
    tentativas reais documentado em `experiments/live.py`), pergunta a mesma
    coisa `n_votes` vezes e só mantém, na resposta final, os índices que
    foram escolhidos em **mais da metade** das rodadas válidas — nem união
    (que soma todos os palpites e piora falso-positivo) nem unanimidade (que
    descarta acertos por causa de uma única rodada ruim). Tolerante a falha:
    se uma chamada individual lançar exceção, a votação segue com as demais;
    só propaga o erro se todas falharem. `n_votes=1` desliga a votação (uma
    chamada só, sem custo extra) — útil quando o ganho de confiabilidade não
    compensa o custo de API extra.

    Desenho inspirado numa técnica equivalente que encontrei (pesquisa
    conceitual, sem copiar código) em frameworks de automação internos não
    relacionados a este projeto acadêmico — eles usam exatamente maioria
    simples sobre múltiplas chamadas do mesmo classificador de visão, pelo
    mesmo motivo.
    """
    if n_votes < 1:
        raise ValueError("n_votes deve ser >= 1.")

    answers: list[RecaptchaGridAnswer] = []
    last_error: Exception | None = None
    for _ in range(n_votes):
        try:
            answers.append(
                solve_recaptcha_grid(
                    screenshot_png=screenshot_png,
                    instruction=instruction,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    client=client,
                )
            )
        except Exception as exc:  # noqa: BLE001 - qualquer rodada pode falhar isoladamente
            last_error = exc

    if not answers:
        assert last_error is not None
        raise last_error

    votes = collections.Counter(idx for answer in answers for idx in answer.tile_indices)
    majority_threshold = len(answers) / 2
    winning_indices = sorted(idx for idx, count in votes.items() if count > majority_threshold)
    combined_reasoning = " | ".join(answer.reasoning for answer in answers)
    return RecaptchaGridAnswer(tile_indices=winning_indices, reasoning=combined_reasoning)
