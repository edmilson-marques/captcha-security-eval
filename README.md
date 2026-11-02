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

**Ciclo 2: prompt de camuflagem explícito — não melhorou.** Com autorização
do usuário, implementei a correção candidata (nomear "quebra de camuflagem
de cor" direto no prompt de `icon_outlier`) e rodei o lote oficial de 20.
**Resultado: 0/13 em "ícone diferente", 0/5 em "laços fechados"** — zero
melhora. Somando com o histórico da sessão, o clique-sobre-canvas está em
**0/41 no total**, mesmo depois de três correções independentes e bem
fundamentadas (resolução nativa, raciocínio estruturado, framing de
camuflagem). Isso é evidência mais forte de que o gargalo não é engenharia
de prompt — é uma limitação de capacidade perceptual do modelo
(`gpt-6.1-terra`) específica desses dois tipos de tarefa. Caminhos ainda não
tentados, não executados sem autorização: (i) modelo mais forte
(`gpt-6-astra`); (ii) abordagem de visão computacional clássica — calcular
contraste de cor (ex. Delta-E) entre o ícone e o fundo ao redor numericamente
em vez de pedir ao modelo um julgamento qualitativo, usando/estendendo
`src/attacks/classical_cv.py` (Atividade 2).

**Ciclo 3: `gpt-6-astra` (10x mais caro) — também não melhorou.** Com
autorização explícita do usuário, testei o modelo mais forte só no solver de
clique-sobre-canvas (`--canvas-model`, sem mexer no padrão usado pelos
solvers que já funcionam). No caminho, descobri um **6º tipo de desafio**
("Clique no número que pertence a este grupo: 789") com resposta
objetivamente verificável — diferente dos outros dois, não é julgamento
perceptual subjetivo — e adicionei um `criterion_type="digit_membership"`
dedicado antes de medir. **Resultado: 0/10** (0/6 ícone diferente, 0/4 número
do grupo). Somando com o histórico, clique-sobre-canvas fecha a sessão em
**0/51**. Um modelo substancialmente mais forte não acertar nem a tarefa
aparentemente mais objetiva (ler um dígito) enfraquece a hipótese de "modelo
insuficiente" — o padrão agora sugere algo na execução/verificação do
próprio desafio ainda não identificado, não necessariamente um teto de
capacidade do modelo. Bug real corrigido no caminho: `solve_puzzle_piece`
também podia retornar `None` numa recusa sem tratamento (mesmo padrão já
visto em `solve_canvas_points`).

**Achado que corrige a leitura acima — "número do grupo" é anti-OCR
deliberado, não uma tarefa fácil.** Inspecionando amostras reais direto do
canvas (sem custo de API), os dígitos aparecem ocultos atrás de um padrão
denso de listras verticais sobrepostas — a mesma técnica clássica de
ofuscação de CAPTCHA de texto distorcido (von Ahn et al., citada no Cap. 3
do TCC1), não texto limpo. Isso explica de forma mais completa por que
tanto o LLM quanto o OCR clássico (ver próxima seção) tiveram dificuldade
aqui — não é um teto de capacidade de leitura de dígito em geral.

### Técnica alternativa sem LLM: visão computacional clássica + OCR

A pedido explícito do usuário (parar de iterar em prompt do mesmo modelo e
tentar algo estruturalmente diferente), implementei uma técnica nova —
"classical_cv" — usando segmentação por borda (OpenCV) e OCR (Tesseract)
em vez de um modelo de visão, em `attacks/classical_canvas_solver.py`
(`solve_icon_outlier_classical`, `solve_closed_loop_classical`,
`solve_digit_membership_classical`). Mesmo escopo dos três subtipos canvas
que o solver via LLM não resolveu; grade e arrastar-e-encaixar continuam
com o solver via LLM (já funcionam bem).

```bash
python -m experiments.run_live_cli --target hcaptcha --technique classical_cv --n 20
```

**Dois bugs reais encontrados e corrigidos via prototipagem manual contra
amostras reais salvas, antes de qualquer tentativa ao vivo**: (1) um limiar
de saturação de cor global não funciona — traços diferentes do mesmo
desafio têm saturação bem diferente entre si; detecção por borda (Canny)
deu sinal bem mais forte. (2) A captura nativa do canvas inclui uma faixa
preta sólida no topo (área do cabeçalho, não desenhada no canvas) que
confundia a detecção — corrigido com recorte automático da transição.

**Lote oficial de 20 tentativas reais: 0/20** — revelou um **terceiro bug**,
só visível ao vivo: a borda/moldura do próprio desafio tem poucos pixels
(contorno fino) mas um bounding box gigante, passava no filtro de área e
"vencia" como maior contraste em praticamente todo desafio real, mascarando
os ícones de verdade (confirmado porque 4 desafios diferentes "detectavam"
a mesma coordenada). Corrigido com um filtro de bounding box.

**Lote de 500 tentativas com pausas de 30s a cada 10** (pedido do usuário,
`--pause-every`/`--pause-seconds`, ver `experiments.runner.run_experiment`)
— **interrompido pelo usuário em 271/500 ao ficar claro que não estava
funcionando: 0 sucessos** em 224 tentativas aplicáveis (0/113 ícone
diferente, 0/61 laços fechados, 0/50 número do grupo). Amostra bem maior
que qualquer lote anterior, confirmando com mais confiança estatística que,
mesmo com o bug da borda corrigido, CV clássica/OCR como implementada aqui
também não resolve nenhum dos três subtipos. **Conclusão do estudo**: os
três tipos de desafio canvas do hCaptcha permanecem sem solução nas duas
técnicas tentadas (LLM de visão, incluindo modelo mais forte; CV
clássica/OCR) — um resultado legítimo e documentável para a comparação,
não um problema em aberto a continuar perseguindo sem uma direção nova.

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
      próprio — puxado quase todo pelo arrastar-e-encaixar (16/24 nas quatro rodadas, ~67%);
      clique em ponto sobre canvas (ícone diferente, laços fechados, e um 6º tipo
      descoberto, "número que pertence a um grupo") ficou em **0/51 via LLM** (incluindo
      `gpt-6-astra`, 10x mais caro) **e 0/271 via CV clássica/OCR** (técnica alternativa sem
      LLM, `attacks/classical_canvas_solver.py` — lote de 500 interrompido pelo usuário em
      271 ao ficar claro que não funcionava), mesmo depois de corrigir 3 bugs reais de
      execução ao longo do caminho (detecção de bot, faixa preta do cabeçalho, borda/moldura
      do desafio mascarando os ícones). **Conclusão**: os três tipos canvas permanecem sem
      solução nas duas técnicas tentadas — resultado legítimo e documentável pro estudo
      comparativo. Arrastar-e-encaixar e grade continuam resolvidos pelo solver via LLM.
      **Falta**: ligar o Turnstile (precisa de página própria). Rodar lotes maiores de
      coleta deve ter ritmo/volume combinado com o orientador (ver "Escopo ético" acima).
- [x] Atividade 7 — testes unitários (89 passed/3 skipped) + comparação das
      técnicas (`analysis/compare.py`) (Out-Nov/2026)
- [ ] Atividade 8 — manutenção do pipeline (reativa, conforme forem surgindo
      mudanças de DOM/bloqueios durante a coleta real — Out-Nov/2026)
- [x] Atividade 9 — relatório comparativo consolidado (`analysis/report.py`,
      tabela LaTeX + gráfico por alvo) (Nov/2026)
