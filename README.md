# captcha-security-eval

Pipeline experimental do TCC2 "Avaliação da Segurança de Mecanismos CAPTCHA Diante de
Técnicas Modernas de Web Scraping: um Estudo Comparativo entre Google reCAPTCHA, hCaptcha
e Cloudflare Turnstile" — Edmilson Marques Batista, UFPel, orientação Prof. Dr. Ulisses
Brisolara Corrêa.

Repositório **privado**, de uso acadêmico. Não é afiliado a nenhum repositório/organização
de trabalho do autor.

## Escopo ético

Os experimentos rodam **somente** contra páginas de demonstração oficiais dos próprios
fornecedores (ou contra uma página própria do aluno usando sitekey oficial em modo de
teste/managed), nunca contra sites de produção de terceiros. Ver `config/targets.yaml`
para as URLs e as notas sobre cada alvo (em especial a ausência de uma demo oficial
full-page do Cloudflare Turnstile). Validar volume/ritmo de requisições com o orientador
antes de cada rodada de coleta.

## Estrutura

```
config/targets.yaml       URLs de demo + metadados de cada CAPTCHA-alvo
src/harness/              Atividade 1 — driver Selenium, coletores, persistência de amostras
src/attacks/              Atividade 2 — CV clássica, OCR, CNN de grade de imagem, Turnstile comportamental
src/evasion/               Atividade 3 — proxy, user-agent, cookies, comportamento humano
src/experiments/           Atividades 4-6 — runner de experimentos e métricas
src/analysis/               Atividades 7-9 — comparação de técnicas e relatório final
tests/                       Atividade 7 — suíte pytest
data/raw/, data/results/     amostras e resultados (não versionados, ver .gitignore)
notebooks/                   análise exploratória para o relatório comparativo
```

Mapeamento completo atividade → código no plano
`~/.claude/plans/com-base-nos-documentos-precious-starlight.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## Coletando uma amostra (Atividade 1)

```bash
python -m harness.collect_cli --target recaptcha
python -m harness.collect_cli --target hcaptcha
python -m harness.collect_cli --target turnstile --url https://<sua-pagina-de-teste>
```

Cada amostra é salva em `data/raw/<alvo>/<timestamp>/` (screenshot, HTML e metadados).

## Solver de visão via LLM (custo real, já validado ao vivo)

`src/attacks/llm_vision_solver.py` implementa uma técnica alternativa de IA: envia o
screenshot do desafio + a instrução para um modelo multimodal da OpenAI (Responses API,
`gpt-6.1-terra`) e recebe de volta uma resposta estruturada (índices de tiles, ou
coordenadas de um puzzle). Requer `OPENAI_API_KEY` no ambiente (ex. num `.env` na raiz do
projeto, já coberto pelo `.gitignore` — nunca commitar a chave).

**Validado ao vivo em 2026-11-01** contra a amostra real
`data/raw/recaptcha/20261101T190501120789/challenge.png` ("selecione os quadrados com
motocicletas", grade 4x4): resposta `[5, 6, 9, 10]`, correta (conferida visualmente),
custo real **US$0,0011 por chamada**. Os testes automatizados continuam usando um cliente
mockado (não fazem chamada paga a cada execução da suíte) — a validação ao vivo foi
manual, ver docstring do módulo.

### Técnica ligada ao runner (`recaptcha_llm_vision_attempt`)

```bash
python -m experiments.run_live_cli --target recaptcha --technique llm_vision --n 1
```

**Primeira versão (passada única), 4 tentativas reais, 0/4 sucessos**: 3 caíram em
grades 3x3 "dinâmicas" (o reCAPTCHA troca o tile por uma foto nova ao ser clicado —
uma única passada não vence esse tipo por design); 1 foi super-segmentação visual.

**Corrigido com um loop de refinamento por rodadas** (desenho próprio — pesquisei antes
como um framework de automação interno não-relacionado trata CAPTCHA, e eles também
não tinham esse refinamento iterativo pós-clique, só uma checagem de staleness antes
do clique; então não tinha o que copiar, foi desenho novo): a cada rodada, refotografa
a grade, repergunta ao modelo e clica só os índices novos, até convergir. **3 tentativas
reais com o loop, ainda 0/3, mas por um motivo diferente e mais relevante**: o
mecanismo do loop funciona certo (2 de 3 convergiram corretamente na 2ª rodada), só que
em 1 das 3 o modelo (`gpt-6.1-terra`) **recusou responder por segurança**
(`"Não posso ajudar a resolver ou contornar um CAPTCHA"`) — uma barreira de moderação de
conteúdo que um classificador treinado do zero (`image_grid_cnn/`) não teria. Achado
relevante pra comparação entre técnicas do estudo.

**Achado maior: detecção de bot via user-agent mascarava o verdadeiro desempenho.**
Mais 5 tentativas (após só remover ruído de interface do prompt, sem tocar na moldura
da tarefa) deram de novo 0/5 — mas com o loop convergindo corretamente mesmo em casos
que não eram recusa, o que levantou a suspeita de bloqueio por detecção de automação
*antes* de qualquer avaliação da resposta. Confirmado: `navigator.userAgent` do Chrome
headless continha literalmente `"HeadlessChrome"` (achado e corrigido em
`src/harness/browser.py` — ver PR #1). **Depois da correção, 5 novas tentativas reais
deram 2/5 sucessos** — das 3 falhas restantes, 2 foram recusas de segurança do modelo e
1 foi imprecisão visual real (sem recusa, loop convergiu certo). Ou seja, com o sinal de
bot removido, a causa de falha passou a ser só o que interessa à comparação do estudo
(recusa de segurança vs. imprecisão visual), não mais um artefato do harness.

**Lote de 20 tentativas reais, para significância estatística**: **10/20 sucessos
(50%)**, confirmando a taxa da amostra pequena anterior. Reclassificando as 10 falhas
pelo formato real da saída (vazio vs. não-vazio, não pela palavra "não posso" no texto
de `reasoning`): **4/20 (20%) recusa verdadeira** (`tile_indices: []`) e **6/20 (30%)
resposta real rejeitada** (imprecisão visual, loop convergiu normalmente). Achado extra:
em 2 das 20, o modelo verbalizou recusa no `reasoning` mas ainda assim preencheu
`tile_indices` com uma resposta real (uma delas até teve sucesso) — uma "recusa
performática" que não corresponde a recusa efetiva; por isso o critério de recusa usado
é a saída vazia, não o texto livre. Detalhamento completo na docstring de
`recaptcha_llm_vision_attempt` em `src/experiments/live.py`.

**Self-consistency por votação de maioria: 15/20 sucessos (75%)**. Pesquisei
antes como frameworks de automação internos não relacionados resolvem o
mesmo problema de imprecisão visual (sem copiar código, só a técnica):
chamar o classificador N vezes por rodada e manter só os índices escolhidos
pela maioria simples (`solve_recaptcha_grid_by_majority_vote`, `n_votes=3`).
Efeito bem delimitado nas 20 tentativas: das 6/20 falhas por imprecisão
visual da rodada anterior, sobrou só **1/20** — a votação atacou exatamente
o que deveria. A recusa de segurança do modelo **não mudou** (4/20 em ambas
as rodadas), como esperado: votação é uma técnica de precisão visual, não
de contorno de recusa. Achado extra: quando a maioria das rodadas de voto
recusa, uma minoria que responde de verdade não atinge maioria e é
descartada — a recusa é "pegajosa" sob votação, comportamento correto do
desenho.

### hCaptcha baseline (`hcaptcha_baseline_attempt`) — mesmo lote de 20

```bash
python -m experiments.run_live_cli --target hcaptcha --technique baseline_no_select --n 20
```

**0/20 sucessos — esperado por desenho**: esta é a técnica "sem ataque" (clica
aceitar/pular às cegas), então falhar sempre que aparece um desafio real é o
comportamento correto, não um achado. O dado relevante é outro: o hCaptcha
apresentou desafio interativo real em **100%** das 20 tentativas (nunca
"passou direto") e esse desafio variou entre **três tipos distintos**, nenhum
deles grade de fotos: "Clique no ícone que é diferente" (10/20, o mais
comum), "Arraste o elemento para o local onde se encaixa" (7/20) e "Clique em
todos os laços fechados" (3/20). Detalhes em
`src/harness/collectors/hcaptcha_collector.py` e na docstring de
`hcaptcha_baseline_attempt`.

### hCaptcha via solver de visão (`hcaptcha_llm_vision_attempt`) — novo

```bash
python -m experiments.run_live_cli --target hcaptcha --technique llm_vision --n 20
```

Implementa solvers para os 4 tipos de desafio observados. Pesquisei antes como
frameworks de automação internos não relacionados resolvem esses mesmos
problemas (sem copiar código, só arquitetura): **não encontrei equivalente
para nenhum dos 3 tipos canvas** (ícone diferente, arrastar-e-encaixar, laços
fechados) — a estratégia observada lá nem tenta resolvê-los, só pede um novo
desafio repetidamente até sair um do tipo grade. Confirma que esses três
tipos são um gap real, não só deste projeto; o desenho dos solvers aqui é
original. O que aproveitei de lá foi a técnica de **execução** do clique: como
os três tipos canvas não têm elemento DOM individual por item (só um
`<canvas>` inteiro), a interação é só por coordenada de pixel — e eventos de
mouse sintetizados via JS puro podem ser rejeitados (`isTrusted=false`) mesmo
com a resposta certa, daí o clique/arrasto via `ActionChains` com trajetória
de Bézier (`src/harness/humanized_actions.py`) em vez de um "teleporte" direto.

**20 tentativas reais: 4/20 sucessos (20%)**, bem desigual entre os tipos
(nenhuma tentativa caiu no tipo grade nesta amostra):
- **Arrastar-e-encaixar: 4/7 (57%)** — resultado mais forte, valida a
  extração de origem+destino da peça combinada com o arrasto humanizado.
- **Clique em ponto(s) sobre canvas (ícone diferente + laços fechados):
  0/13 (0%)** — nenhum resolvido nesta amostra; sem instrumentação adicional
  não dá pra separar imprecisão do modelo de outra causa, mas o clique em si
  não falhou mecanicamente em nenhuma tentativa (mesma função usada com
  sucesso dentro do arrasto). Detalhamento completo na docstring de
  `hcaptcha_llm_vision_attempt` em `src/experiments/live.py`.

Resultado de uma única rodada de medição (bounded), não de uma otimização
iterada até um alvo de taxa de sucesso.

**Tentativa de melhorar o clique-sobre-canvas: não subiu.** A pedido do
usuário, troquei para `solve_canvas_points_by_consensus` — mesma técnica de
self-consistency que deu +25pp no reCAPTCHA, adaptada para coordenada
contínua via clustering espacial em vez de maioria exata de índice. Mais 20
tentativas reais: **5/20 (25%)** no total, mas **clique-sobre-canvas
continuou em 0/10 (0%)** — a melhora do agregado foi só variação amostral no
arrastar-e-encaixar. Achado no processo: uma recusa do modelo pode fazer
`solve_canvas_points` retornar `None` (não `points=[]` como a recusa da
grade) — corrigido para não derrubar a votação. Com **0/23** cliques-sobre-
canvas bem-sucedidos na sessão inteira (antes e depois do consenso), a causa
provavelmente não é só ruído pontual do modelo (votação deveria destravar
pelo menos alguns casos, como fez na grade).

**Instrumentação a nível de evento JS descarta bug de execução.** Injetei
listeners (`pointerdown`/`mousedown`/`pointerup`/`mouseup`/`click`) direto no
`<canvas>` do desafio real e cliquei em 3 pontos conhecidos via
`click_canvas_point`: sequência completa de eventos, **100% `isTrusted=true`**
e coordenada recebida idêntica, pixel a pixel, ao ponto pedido nos 3 casos —
sem desvio, sem erro de escala. O mecanismo de clique está correto; não há
bug de coordenada nem rejeição de evento a explicar o 0/23. A causa
remanescente é, com alta confiança agora, **precisão visual do próprio
modelo** para esses dois tipos específicos de desafio — sistemática o
bastante pra não ceder nem a self-consistency. Não testado: se um modelo mais
forte (`gpt-6-astra`) ou outro pré-processamento de imagem mudaria isso.
Detalhamento completo na docstring de `hcaptcha_llm_vision_attempt` e de
`harness/humanized_actions.py`.

### Ciclo de diagnóstico estruturado (/loop) — 3 correções + achado sobre camuflagem

A pedido do usuário, apliquei um roteiro formal de diagnóstico ao problema
acima (≥3 hipóteses de causa raiz, ≥3 correções por hipótese, escolha
justificada evitando overengineering). Implementei e testei (smoke test,
sem lote oficial ainda — combinado previamente não rodar lote grande nem
trocar de modelo sem perguntar antes):

1. **Captura nativa do canvas** (`toDataURL` em vez de `screenshot_as_png`)
   — valida sem `SecurityError`, ganho real de resolução (2x no caso
   observado). Gotcha: muda o espaço de coordenada, exige escalar antes de
   clicar (`native_scale` em `_canvas_native_screenshot`).
2. **Prompt especializado por tipo de critério** (`icon_outlier` /
   `closed_loop`) com raciocínio estruturado (enumerar → comparar/classificar
   → responder), em vez do prompt genérico único de antes.
3. **Campo `elements_considered`** na resposta, só para auditoria.

**Achado novo, mais importante que as 3 correções em si**: inspecionando as
imagens nativas diretamente, confirmei que "ícone diferente" mostra
consistentemente só **2 ícones** por desafio — não um grupo com maioria e um
intruso, como o nome sugere. Com só 2 candidatos, não há "maioria" pra
comparar; o padrão observado (nas 3 amostras inspecionadas e no próprio
raciocínio do modelo, que às vezes verbaliza "não há uma maioria de
referência") sugere que o critério real é **quebra de camuflagem de cor**
(qual dos dois ícones destoa da paleta do fundo decorativo vs. qual foi
colorido pra se camuflar nela) — uma habilidade perceptual distinta de
classificação semântica, que é o que o modelo já faz bem na grade do
reCAPTCHA. Isso refina a causa provável do 0/23: não é imprecisão genérica,
é um tipo de percepção visual específico que o prompt ainda não nomeia
explicitamente. Fica como próxima correção candidata, não implementada
ainda. Detalhamento completo na docstring de `hcaptcha_llm_vision_attempt`.

## Rodando o OCR (opcional)

`src/attacks/ocr.py` depende do binário de sistema `tesseract-ocr`. Se não estiver
instalado, `tests/test_ocr.py` é pulado automaticamente; para rodar de verdade:

```bash
sudo apt install -y tesseract-ocr tesseract-ocr-por
```

## Status

- [x] Atividade 1 — ambiente de testes + coletores (Ago/2026)
- [x] Atividade 2 — módulos de ataque: CV clássica, OCR, CNN de grade de imagem,
      probe comportamental do Turnstile, solver de visão via LLM (Ago-Set/2026)
- [x] Atividade 3 — técnicas de evasão: proxy, user-agent/fingerprint, cookies,
      comportamento humano (Ago-Set/2026)
- [~] Atividades 4-6 — infraestrutura pronta e **ligada de verdade** para reCAPTCHA e
      hCaptcha, ambos com `baseline_no_select` e `llm_vision` (`experiments/live.py`),
      rodadas reais via `run_live_cli.py`. reCAPTCHA: loop de refinamento por rodadas +
      self-consistency por votação de maioria (`n_votes`), depois de corrigir o
      vazamento de fingerprint (`"HeadlessChrome"` no user-agent, PR #1) —
      **15/20 sucessos reais (75%)**; falhas restantes só recusa de segurança do modelo
      (20%) ou imprecisão visual residual (5%), não mais detecção de bot (ver seção
      acima). hCaptcha: 4 tipos de desafio distintos observados ao vivo (grade, ícone
      diferente, arrastar-e-encaixar, laços fechados — nenhum grade de tiles exceto o
      primeiro, `image_grid_cnn/` não se aplica aos outros 3), cada um com solver
      próprio — puxado quase todo pelo arrastar-e-encaixar (9/15 nas duas rodadas, ~60%);
      clique em ponto sobre canvas (ícone diferente + laços fechados) ficou em **0/23**
      mesmo depois de tentar self-consistency por consenso espacial (mesma técnica que
      funcionou na grade do reCAPTCHA). Instrumentação a nível de evento JS (listeners no
      `<canvas>` real) descartou bug de execução — 100% `isTrusted=true`, coordenada
      exata em todos os pontos testados — então a causa provável é um limite de precisão
      visual do próprio modelo para esses dois tipos, não um bug de clique. **Falta**:
      ligar o Turnstile (precisa de página própria) e, se fizer sentido, testar um modelo
      mais forte ou outro pré-processamento de imagem para o clique-sobre-canvas. Rodar
      lotes maiores de coleta deve ter ritmo/volume combinado com o orientador (ver
      "Escopo ético" acima).
- [x] Atividade 7 — testes unitários (89 passed/3 skipped) + comparação das
      técnicas (`analysis/compare.py`) (Out-Nov/2026)
- [ ] Atividade 8 — manutenção do pipeline (reativa, conforme forem surgindo
      mudanças de DOM/bloqueios durante a coleta real — Out-Nov/2026)
- [x] Atividade 9 — relatório comparativo consolidado (`analysis/report.py`,
      tabela LaTeX + gráfico por alvo) (Nov/2026)
