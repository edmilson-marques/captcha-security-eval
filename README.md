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

## Checklist de limitações metodológicas

`docs/checklist_limitacoes_processo.md` reúne os problemas achados durante a coleta/treino
que **não têm correção de código** (ruído anti-bot, erro composto em desafio encadeado,
significância estatística em lote pequeno, corrida entre forks de rotulagem, revisão de
contaminação por selo). Conferir antes de reportar qualquer número novo no TCC.

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

## Solver de visão via LLM (custo real, já validado ao vivo) — técnica atual pro reCAPTCHA

> **Idas e vindas de 2026-11-05, registradas por transparência**: por um período curto, o
> projeto tentou substituir este solver pelo CNN sem LLM da seção "Solver de grid via CNN"
> logo abaixo (grade do reCAPTCHA e do hCaptcha) e remover `solve_puzzle_piece` do
> arrastar-e-encaixar do hCaptcha. Ao vivo, o clique do CNN não se mostrou confiável
> (usuário observou "não está clicando no grid", causa ainda não diagnosticada) e a troca
> foi revertida ("coloca o v2 na openAI") — **este solver via LLM volta a ser a técnica
> usada/reportada** pro reCAPTCHA e pela grade dentro de `hcaptcha_combined_attempt`. O
> arrastar-e-encaixar do hCaptcha continua sem chamar `solve_puzzle_piece` em `combined`
> por ora (não fazia parte do pedido de reversão). O código do CNN de grid permanece no
> repositório, disponível como técnica `grid_cnn` isolada pro reCAPTCHA, só não é mais o
> caminho padrão.

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

**Segundo vazamento de fingerprint, mais profundo, achado só mais tarde (2026-11-05):**
a correção do `user-agent` acima não foi a história completa. Inspirado em boas práticas
já documentadas publicamente sobre como mecanismos anti-bot avançados identificam
automação (ex. o projeto `CircuitSavage/hcaptcha-hsj-hsw-reversed`, engenharia reversa
pública do script de fingerprint do hCaptcha), testei diretamente no harness deste
projeto se o Chrome headless também denuncia automação via **media queries CSS de
dispositivo de entrada** — um sinal que não depende de string nenhuma, não é mascarável
por `--user-agent` e, segundo a mesma fonte, **não é corrigível por comando de CDP**
(a emulação de `pointer`/`hover` do protocolo é silenciosamente ignorada pelo Chrome
headless). Confirmado ao vivo neste harness:

```
headless=True:  pointer=none      any-pointer=none   hover=none    # denuncia automação
headless=False: pointer=fine      any-pointer=fine    hover=hover   # indistinguível de pessoa
```

Ou seja, **todo lote histórico deste projeto rodado com `headless=True`** (a imensa
maioria, já que esse é o padrão de `BrowserConfig`) carrega esse segundo sinal de
automação, mesmo depois da correção do `user-agent` — um fator de confusão adicional,
não quantificado, que pode ter contribuído pro 0% histórico em "ícone diferente"/"laços
fechados"/"número do grupo" além da imprecisão de solver já documentada. Diferente do
`user-agent`, a correção aqui não é uma troca de string: é rodar com `headless=False`
(janela real ou virtual via Xvfb, não emulação), o que só passou a ser testado neste
projeto nas últimas rodadas ao vivo (`--headed`). Re-rodar os lotes históricos nessa
condição para isolar esse fator fica como trabalho futuro — não refeito agora por não
ter sido o escopo pedido no momento do achado.

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
também não resolve nenhum dos três subtipos.

**Troca de métrica de contraste de cor (redmean → Delta-E CIE76) — também
não melhorou.** `color_contrast` usava "redmean", uma aproximação de baixo
custo; trocada para Delta-E real (conversão BGR→CIELAB via `cv2.cvtColor`
em ponto flutuante, distância euclidiana em L\*a\*b\*, ver
`attacks/classical_cv.py::_bgr_to_lab`), única mudança isolada no
`solve_icon_outlier_classical` (usado só pelo subtipo "ícone diferente").
Suíte de testes re-executada antes de medir ao vivo: 128 passed, nenhum
teste quebrou (as duas propriedades testadas — ordenação relativa de
contraste e distância zero pra cores idênticas — valem tanto pra redmean
quanto pra Delta-E). **Lote real de 20 tentativas pós-troca: 0/20** no
total, com **9 tentativas de "ícone diferente" e 0 sucessos** — mesmo
resultado da métrica anterior. Cumulativo de "ícone diferente" via CV
clássica na sessão: **0/122** (113 com redmean + 9 com Delta-E). Esse
resultado reforça a leitura já registrada no ciclo de diagnóstico do
solver via LLM (ver seção acima): o gargalo não está na métrica de
contraste de cor escolhida (redmean vs. Delta-E perceptual), mas em algo
anterior a ela — provavelmente a segmentação por borda (`detect_stroke_
regions`) não isola de forma confiável os dois ícones candidatos antes de
qualquer comparação de cor acontecer.

**Instrumentação de persistência + checagem da hipótese de repertório de
imagens.** A pedido do usuário, `experiments/live.py` passou a gravar em
disco (via `harness.storage.save_attempt_capture`, mesmo layout de
`data/raw/<alvo>/<timestamp>/` usado pela Atividade 1) o screenshot nativo
e o resultado verificado de cada tentativa real de clique-sobre-canvas —
antes disso, essas imagens eram processadas só em memória e descartadas, e
o "repertório" real em disco era de apenas 3 amostras manuais (nenhuma
delas desse tipo de desafio). Motivação: avaliar se os três subtipos
reciclam um conjunto finito de padrões visuais, o que viabilizaria
casamento de padrão em vez de julgamento por imagem a cada tentativa.
**Resultado da checagem, com as 24 primeiras imagens reais acumuladas**
(9 ícone diferente, 12 laços fechados, 3 número do grupo; lote real de
25 tentativas, 0/25 sucessos — mesmo padrão de sempre): comparação par a
par dentro de cada subtipo (miniatura 64×64 em escala de cinza, diferença
média absoluta de pixel) não encontrou nenhum par com diferença média
abaixo de 5 — a menor diferença observada foi 8,75, a maior passou de 53.
**Não há evidência de reciclagem de um conjunto fixo de imagens** nessa
amostra inicial: o ambiente de demo parece renderizar cada desafio de
forma genuinamente distinta. Isso enfraquece a hipótese de casamento de
padrão como caminho viável (pelo menos nessa escala de amostra) e reforça
o treino de um classificador supervisionado (que generaliza sobre
variação visual, não depende de repetição exata) como a alternativa mais
promissora ainda não tentada neste estudo.

**Classificador CNN para "laços fechados" (`attacks/canvas_click_cnn/`) —
construído e testado ao vivo, ainda sem melhora.** A pedido do usuário,
implementei o classificador supervisionado sugerido acima, escopado só
pro subtipo "laços fechados" (único com dado rotulável — ver abaixo).
Pipeline: `dataset_prep.py` extrai os recortes de candidato das amostras
reais salvas (mesma segmentação de `detect_stroke_regions`);
`build_dataset.py` grava, no layout `ImageFolder`, só os recortes que
inspecionei individualmente e considerei limpos o bastante pra rotular
(20 de ~52 candidatos detectados — o resto é artefato de segmentação
conhecido: fatia de ruído, caixa fundindo dois traços, candidato
redundante de borda dupla); `train.py` reaproveita a arquitetura/loop de
treino já existente em `image_grid_cnn/` (nunca fora, de fato, específica
de grade). **Rotulagem assistida por IA, não pelo autor** — registrado
explicitamente por honestidade metodológica (ver docstring de
`build_dataset.py`). Durante a rotulagem, encontrei e corrigi 1 rótulo
contaminado (caixa capturando 2 traços na mesma imagem) ao conferir o
recorte isolado, não só a imagem anotada de contexto — checagem que vale
a pena repetir sempre que uma nova rotulagem for feita.

Integrado como nova técnica `classical_cv_cnn` (`run_live_cli.py`), que
usa o CNN só pra "laços fechados" e mantém a heurística clássica pros
outros dois subtipos. **Lote real de 18 tentativas: 0/18** no total, com
apenas **3 tentativas de "laços fechados" (0/3)** — amostra pequena
demais pra tirar conclusão estatística, mas sem sinal de melhora. Achado
de diagnóstico no processo: o dataset de treino é desbalanceado (16
fechados / 4 abertos), e uma das 3 tentativas reais clicou em 4 pontos
(provável viés do classificador pra prever "fechado" com mais frequência
dada essa proporção no treino) — suspeito que isso contribua pro erro,
não testado ainda com balanceamento de classe. Com a instrumentação de
persistência (seção anterior), esse mesmo lote já somou mais 24 imagens
reais ao repertório, incluindo amostras de "laços fechados" que podem
virar rótulo novo numa próxima rodada.

**Conclusão do estudo**: os três tipos de desafio canvas do hCaptcha
permanecem sem solução em nenhuma das três técnicas tentadas (LLM de
visão, incluindo modelo mais forte; CV clássica/OCR, incluindo duas
métricas de contraste de cor distintas; CNN supervisionado pra laços
fechados) — um resultado legítimo e documentável para a
comparação, não um problema em aberto a continuar perseguindo sem uma
direção nova.

### Achado maior: "ícone diferente" é um desafio animado, não estático

Observando o navegador aberto ao vivo (a pedido do usuário), foi
identificado que o desafio "ícone diferente" **não é uma imagem parada**:
é uma animação contínua em que os ícones do `<canvas>` aparecem, somem
(fade) e se deslocam ao longo do tempo. Confirmado capturando 8
screenshots nativos do mesmo desafio ao longo de ~4s, antes de qualquer
clique: em metade das capturas os dois ícones visíveis eram **idênticos**
(nenhum "diferente" presente naquele instante), e um terceiro ícone
visualmente distinto só apareceu por ~1s dentro do ciclo. Isso refina a
causa-raiz do 0% documentado acima: as três técnicas tentadas processam
sempre **um único screenshot**, que pode capturar um instante sem os dois
ícones certos pra comparar — um fator nunca antes isolado do julgamento
em si (camuflagem de cor, segmentação).

**Solver temporal implementado** (`attacks/temporal_canvas_solver.py`):
captura uma rajada de 8 screenshots (~4s) em vez de um só, localiza
"slots" espaciais de alta variância temporal, e para cada slot mede em
quantos frames há conteúdo real (comparação de cor contra o anel
imediatamente ao redor, no mesmo frame — não uma estatística agregada ao
longo do tempo). O slot que aparece em **menos** frames é tratado como o
ícone-alvo. Dois problemas reais de generalização foram encontrados e
corrigidos durante o desenvolvimento, via testes sintéticos adversariais
(`tests/test_temporal_canvas_solver.py`): (1) uma estimativa de fundo por
mediana/mínimo temporal inverte a contagem quando o ícone comum fica
parado numa posição fixa; (2) um componente conexo gigante (fusão de
duas ocorrências próximas) precisa ser descartado por faixa de tamanho
relativo à imagem, senão domina a contagem. **Validado offline com
sucesso** contra a captura real original e contra os dois casos
sintéticos adversariais.

**Resultado ao vivo, porém, ainda negativo**: 10 tentativas reais
genuínas de "ícone diferente" (após corrigir um bug relacionado, ver
abaixo) com o solver temporal: **0/10**, mesmo com o solver encontrando e
clicando um candidato em quase todas. Ou seja, a técnica passa em todos
os testes offline (1 exemplo real + 2 casos sintéticos adversariais),
mas ainda não produziu sucesso real — amostra pequena demais pra
descartar de vez, mas não há sinal de melhora ainda sobre as três
técnicas anteriores.

**Bug real encontrado e corrigido no processo, independente do resultado
acima**: `_infer_canvas_criterion_type` classificava qualquer prompt
contendo a palavra "padrão" como `"icon_outlier"` — o que forçava um
**7º tipo de desafio ainda não catalogado** ("Coloque o item correto
para completar o padrão", uma grade de arrastar-e-soltar com frutas/
flores, visualmente nada parecido com os dois ícones do "ícone
diferente") a ser tratado como se fosse clique-sobre-canvas de ícone.
Descoberto duas vezes de forma independente: rotulando dados reais pro
CNN de laços fechados, e de novo num lote ao vivo do
`hcaptcha_combined_attempt` (3 de 6 tentativas marcadas "icon_outlier"
eram esse outro desafio). Corrigido em `_infer_canvas_criterion_type` e
no despacho de `hcaptcha_classical_cv_attempt`/`hcaptcha_combined_attempt`
(critério não reconhecido agora vira `technique_scope: "not_applicable"`,
não mais um "icon_outlier" por padrão). **Limitação que fica**: números
históricos de "ícone diferente" documentados antes desta correção (nas
seções acima) podem incluir uma fração pequena e não quantificada de
tentativas que eram, na real, esse 7º tipo — não dá pra separar
retroativamente sem reprocessar os `.jsonl` brutos originais.

### Técnica combinada: tenta resolver todos os tipos de desafio

A pedido do usuário ("coloca todos os métodos juntos, pra solucionar
todos os tipos possíveis"), `hcaptcha_combined_attempt`
(técnica `combined` no `run_live_cli.py`) roteia cada tipo de desafio pra
a melhor técnica já medida neste estudo, em vez de marcar alguns tipos
como fora de escopo: grade e arrastar-e-encaixar via LLM (as técnicas
mais fortes medidas, 75%/57%); laços fechados via CNN; ícone diferente
via o solver temporal (acima); número do grupo via OCR clássico. Nenhuma
combinação aqui muda a taxa de sucesso dos tipos que já não funcionavam —
o ponto é ter uma única técnica que **tenta** todos os 6-7 tipos
observados, não uma técnica nova de resolução.

**Achado extra, ainda não corrigido**: testando ao vivo com o navegador
aberto, uma tentativa real de **grade de imagens do hCaptcha** (primeira
vez neste estudo inteiro que esse tipo foi de fato tentado, não só
observado) travou com um erro real do Selenium
(`element click intercepted`, coordenada Y fora de qualquer limite
razoável) — possivelmente a grade reorganiza/anima os tiles no meio da
rodada de refinamento por maioria. Diferente do 0% "por imprecisão
visual" documentado nos outros tipos, esse é um bug de execução distinto,
registrado aqui mas não investigado a fundo.

### Lote real de 100 tentativas (`combined`) — 2 tipos de desafio novos descobertos

A pedido do usuário, rodado um lote de 100 tentativas reais (pausas de
30s a cada 10, mesmo ritmo já usado no lote de 500 da CV clássica).
**Resultado: 0/100** — mas o achado real não é a taxa, é a distribuição
de tipo de desafio encontrada, bem mais diversa do que o catalogado até
aqui:

| Tipo (prompt real) | Tentativas | % | Técnica aplicada |
|---|---|---|---|
| "Clique no número que pertence a este grupo: 789" (digit_membership) | 41 | 41% | OCR clássico |
| **"Por favor, clique nos elementos que são idênticos"** (novo) | 31 | 31% | nenhuma -- `technique_scope: not_applicable` |
| "Clique em todos os loops fechados" (closed_loop) | 16 | 16% | CNN |
| Grade de imagens | 5 | 5% | LLM (não aplicável nesta amostra -- nenhuma chegou a clicar) |
| Erros de infraestrutura (internet caiu 3x, sessão do Chrome morreu 1x) | 4 | 4% | -- |
| "Clique na figura que quebra o padrão na grelha" (icon_outlier) | 1 | 1% | solver temporal |
| "Encontre o objeto que leva para acampar" (novo) | 1 | 1% | nenhuma -- fora de escopo |

**"Clique nos elementos que são idênticos" é um 8º tipo de desafio, nunca
catalogado antes, e é tão ou mais comum que "número do grupo" (31% vs
41% nesta amostra de 100)** — provavelmente o inverso estrutural do
"ícone diferente" (achar os elementos iguais, não o destoante), e
nenhuma técnica tenta resolvê-lo ainda. "Encontre o objeto que leva para
acampar" (1x) sugere um 9º tipo, de busca semântica de objeto em cena —
amostra única demais pra caracterizar.

**Achado estatístico colateral**: com uma amostra de 100 em vez de 20,
fica claro que o "ícone diferente" catalogado nas seções acima (e toda a
investigação da animação/solver temporal) é, na verdade, um dos tipos
**mais raros** do hCaptcha (1/100 nesta amostra) — a maior parte do
esforço de pesquisa desta sessão foi num tipo de desafio
proporcionalmente pouco frequente. "Número do grupo" e o novo "elementos
idênticos" dominam a distribuição real e seguem sem nenhuma técnica
eficaz.

Variação de texto também confirmada: o mesmo tipo de desafio aparece com
frases sinônimas diferentes entre tentativas (ex. "loops fechados" aqui
vs. "laços fechados" em lotes anteriores) -- `_infer_canvas_criterion_type`
já cobre essa variação para os tipos catalogados, mas qualquer prompt
novo precisa ser adicionado manualmente à heurística.

### Três correções integradas (voto adaptativo, clique de grade, "elementos idênticos") e novo lote de 100

A pedido do usuário, três melhorias foram implementadas em paralelo (uma
por alvo) e integradas na mesma sessão: `solve_recaptcha_grid_adaptive_vote`
(só escala votos extras quando a votação empata, não quando o modelo
recusa -- usado agora por padrão nos três lugares que resolvem grade:
`recaptcha_llm_vision_attempt`, `hcaptcha_llm_vision_attempt`,
`hcaptcha_combined_attempt`), `click_grid_tiles` (corrige o
`element click intercepted` da grade do hCaptcha -- rebusca os elementos
a cada clique em vez de reaproveitar uma lista que fica obsoleta quando a
grade re-renderiza um tile) e `solve_identical_elements_classical` (8º
tipo, "Clique nos elementos que são idênticos", zero solver antes).

**Novo lote real de 100 tentativas** (de um lote de 150 planejado,
interrompido no meio por infraestrutura, não por bug de código) com a
técnica `combined` já com as três correções ativas:

| Tipo | Tentativas | Sucessos |
|---|---|---|
| Ícone diferente (solver temporal) | 38 | 0 |
| Número do grupo | 29 | 0 |
| Arrastar-e-encaixar | 21 | 5 (24%) |
| Laços fechados | 9 | 0 |
| Grade de imagens | 2 | 0 |
| "Objeto de acampar" (fora de escopo) | 1 | 0 |

**Dois achados reais deste lote**:

1. **A distribuição de tipo de desafio do hCaptcha não é estável entre
   sessões.** No lote de 100 anterior, "ícone diferente" foi 1% da amostra
   e "elementos idênticos" foi 31%; neste lote, "ícone diferente" foi
   **38%** e "elementos idênticos" não apareceu nenhuma vez. Qualquer
   percentual de distribuição de tipo reportado neste README deve ser lido
   como uma amostra de um momento específico, não uma proporção fixa do
   hCaptcha.
2. **Com uma amostra bem maior (38 em vez de 10), o solver temporal de
   "ícone diferente" confirma 0% de sucesso** — reforça, com mais
   confiança estatística, a conclusão já registrada de que a técnica passa
   em todos os testes offline mas não produz sucesso real contra o
   desafio ao vivo. **Arrastar-e-encaixar caiu pra 24% (5/21)** nesta
   amostra, bem abaixo do ~57-67% histórico de amostras menores (N=7-8) —
   inspecionado individualmente, nenhuma tentativa mostrou erro ou dado
   anômalo (coordenadas e confiança do modelo parecem normais), então o
   mais provável é variação real de amostra pequena-pra-média, não uma
   regressão introduzida pelas correções desta sessão (nenhuma delas toca
   no caminho de arrastar-e-encaixar).

"Elementos idênticos" e o clique de grade corrigido não caíram tentativa
nenhuma de verdade neste lote especificamente (zero ocorrências de
"idênticos"; só 2 de grade, amostra pequena demais pra validar a
correção do bug com confiança) — ainda pendente uma medição real
dedicada a esses dois.

### Bug real no solver temporal de "ícone diferente" — parte do "0%" registrado pode ter sido crash, não erro de solução

Reproduzindo o pipeline ao vivo depois da reversão de "laços fechados"
(`testa`), uma das 3 tentativas de um lote pequeno quebrou com
`ValueError: all input arrays must have the same shape` dentro de
`_temporal_std_map` (`attacks/temporal_canvas_solver.py`). Causa raiz:
`solve_icon_outlier_temporal` chamava `_crop_black_header` **uma vez por
frame** da rajada; como a detecção do fim da faixa preta do cabeçalho
depende de um limiar de brilho médio por linha, ruído de renderização bem
perto do limiar faz a transição ser detectada numa linha diferente em
frames diferentes (confirmado isolando o mecanismo: duas linhas com
brilho 10 e 20, limiar 15.0, geram recortes de 460 e 461 linhas a partir
da mesma imagem) — `np.stack` então recebe frames de altura diferente e
quebra. Corrigido calculando o recorte **uma única vez, a partir do
primeiro frame**, e aplicando a mesma faixa a todos (o cabeçalho é DOM
estático dentro de uma mesma rajada, não deveria mudar de posição entre
frames de qualquer forma). Teste de regressão em
`tests/test_temporal_canvas_solver.py::test_tolerates_black_header_detected_at_slightly_different_rows`
reproduz o crash exato antes do fix e passa depois.

Como `experiments.runner.run_experiment` captura exceção por tentativa e
grava `outcome="failure"` com `extra={"error": ...}` (sem derrubar o
lote), este crash **não aparecia como um tipo de erro separado nos lotes
anteriores — ele virava só mais uma "falha"** indistinguível de um clique
errado. Isso significa que uma fração desconhecida do "0% confirmado numa
amostra de 38" documentado acima pode ter sido este crash, não a lógica
do solver realmente errando a posição do ícone — o tamanho exato dessa
fração não é recuperável retroativamente (os `.jsonl` antigos guardam só
`str(exceção)` quando falham assim, não um campo estruturado que
diferencie "crash" de "clique errado" nos registros já efetivamente
checados). Com o fix, o mesmo lote de 3 tentativas teve **1 sucesso real
em "ícone diferente"** (`criterion_type: icon_outlier`) — amostra única
demais pra revisar a conclusão de 0%, mas motivo suficiente pra reabrir a
pergunta com um lote novo e maior antes de tratar aquele número como
definitivo.

### "Laços fechados" — 4 causas-raiz encontradas, nenhum fix limpo (investigação revertida)

Observando o navegador aberto num lote ao vivo, o usuário notou a técnica
de CV clássica (`solve_closed_loop_classical`) marcando errado contra um
desafio real de 5 traços (1 laço de fato fechado, 4 abertos, incluindo
duas curvas em "S" com várias dobras). Diagnosticar essa amostra revelou
**4 causas-raiz distintas**, cada correção tentada resolvendo uma à custa
de regredir outra:

1. **A máscara de borda (`_stroke_edge_mask`, usada por `detect_stroke_regions`)
   funde traços próximos numa única região** — na amostra real, 3 traços
   próximos viraram 1 bbox só, e `classify_loop_closure` não consegue
   avaliar fechamento de um blob que na verdade é 3 curvas distintas.
2. **Alternativa por limiar de saturação (HSV) segmenta melhor, mas corta
   o trecho de baixa saturação de traços com degradê de cor** (ex. de
   amarelo a roxo no mesmo laço) — um laço genuinamente fechado perdia
   exatamente o trecho de cor mais "lavada" que o fechava, aparecendo como
   aberto pro classificador.
3. **União das duas máscaras (borda OU saturação) melhora a segmentação da
   amostra real**, mas sem o fechamento morfológico que causava o problema
   (1) — e a ausência desse fechamento **reduz a espessura efetiva do
   traço detectado**, o que desloca sistematicamente a métrica de "fração
   cercada" pra baixo. O círculo sintético usado no teste de regressão
   (`tests/test_classical_canvas_solver.py::test_solve_closed_loop_classical_returns_only_closed_curves`),
   que sempre pontuou corretamente como fechado (0.58) com a máscara de
   borda, caiu pra 0.31 só por trocar de máscara — abaixo até do limiar
   original (0.5), quebrando um caso que já funcionava.
4. **A própria métrica de "fração cercada" (flood-fill a partir da borda
   do bbox) pontua curvas abertas em "S" acima do limiar neutro** só pela
   ondulação da curva, que cerca um pouco de área mesmo sem nunca fechar
   de verdade — uma limitação geométrica do método de classificação em si,
   independente de qual máscara alimenta ele.

Qualquer limiar fixo testado (0.5 → 0.73) corrigia a amostra real
observada à custa de regredir o teste sintético já validado — não foi
encontrado um ajuste que generalizasse sem essa troca. **Decisão:
reverter todo o código de volta ao estado original** (máscara só de
borda, `closure_threshold=0.5`, `hcaptcha_combined_attempt` de volta pro
classificador CNN) em vez de enviar uma correção estreita, calibrada numa
única amostra, que resolve um caso visível e quebra outro já coberto por
teste. As 4 causas ficam documentadas aqui — e na docstring de
`solve_closed_loop_classical` — como o achado mais completo desta sessão
sobre por que "laços fechados" resiste tanto ao CNN (18 tentativas reais,
0% — ver acima) quanto à heurística clássica: o gargalo não é falta de
técnica de classificação, é a segmentação anterior a qualquer
classificação, numa família de desafios desenhada justamente pra
dificultar separar traço de traço (curvas próximas, cores em degradê,
geometria ondulada).

**Atualização, mesmo dia**: a conclusão acima (nenhum fix limpo) foi sobre a
métrica clássica por threshold de fechamento. Investigando os três
problemas abaixo com 3 sub-agentes em paralelo, um deles achou e corrigiu
dois bugs novos e reais no classificador **CNN** (ver seção seguinte) —
completamente independentes das 4 causas acima, que continuam válidas
especificamente para `solve_closed_loop_classical`.

### Bug arquitetural maior: critério ficava obsoleto entre rodadas de retentativa

Assistindo o navegador aberto, o usuário notou ao vivo que **"elementos
idênticos" nunca chegava a ser tentado, "laços fechados" marcava errado, e
"número do grupo" nunca era tentado** — três sintomas aparentemente
distintos, investigados em paralelo por 3 sub-agentes (um por sintoma).

A investigação do sub-agente de "número do grupo" achou a causa raiz
**compartilhada pelos três**: `hcaptcha_combined_attempt`,
`hcaptcha_llm_vision_attempt` e `hcaptcha_classical_cv_attempt` liam
`prompt` e calculavam `criterion_type` **uma única vez, antes** do laço de
retentativa (`while outcome_state == "retry"`, ver `_submit_round_and_check`
acima). A suposição documentada até aqui — "hCaptcha reapresenta o *mesmo
tipo* de desafio com conteúdo regenerado" — estava **errada**: inspecionando
as imagens reais salvas por rodada, o regenerar-ao-submeter-errado troca o
**tipo inteiro** do desafio, não só o conteúdo. Uma tentativa rotulada
"digit_membership" na 2ª/3ª rodada podia estar, de fato, olhando pra uma
tela de "laços fechados" ou "elementos idênticos" — o código continuava
aplicando a lógica do tipo da 1ª rodada a um desafio visualmente diferente,
produzindo exatamente os três sintomas observados: OCR rodando sobre um
canvas sem nenhum dígito (lista vazia — "nem tentou"), classificador de
laço fechado avaliando formas de cristal/polígono de um desafio de
"elementos idênticos" (resposta sem sentido — "marcou errado"), e por aí em
diante.

**Corrigido** nas três funções: `prompt` e `criterion_type` agora são
relidos a cada rodada do laço, não só antes dele. Efeito colateral
descoberto no caminho: `native_screenshot = frames[-1]` nos ramos de
rajada (`icon_outlier`/`digit_membership`) guardava um array numpy já
decodificado em vez dos bytes PNG que `save_attempt_capture` espera —
corrigido reencodando (`cv2.imencode`) antes de salvar; sem isso, toda
imagem de debug salva desses dois tipos era um dump de pixels corrompido,
não um PNG de verdade.

**Risco pra qualquer número de taxa de sucesso já reportado neste README
envolvendo mais de 1 rodada** (`rounds_used > 1`): pode estar subestimado,
já que rodadas 2/3 aplicavam lógica do tipo errado em vez de simplesmente
pular. Não é recuperável retroativamente (os `.jsonl` antigos não guardam
o prompt por rodada, só o da 1ª leitura) — novas medições a partir de agora
já vêm corrigidas.

### "Elementos idênticos" — corrigido (invariância a rotação/escala)

Com o critério já correto a cada rodada, ficou claro que
`solve_identical_elements_classical` nunca agrupava nada contra o desafio
real: a versão anterior comparava forma via IoU de silhueta relativa ao
bbox, que o desafio real derrota de propósito **girando e redimensionando**
as cópias idênticas de formas diferentes (confirmado numa amostra real: duas
formas genuinamente idênticas, só em ângulos diferentes, davam IoU de
0.02-0.07 — abaixo de qualquer limiar razoável). Trocado por
`cv2.matchShapes` (distância de Hu, invariante a rotação/escala/translação)
— na mesma amostra, o par de verdade ficou em 0.0106 contra 0.12-0.88 de
todos os outros pares, margem de mais de 10x. A amostragem de cor também
mudou de pixels de borda (sensível a qual trecho de uma forma multifacetada
é amostrado, que muda com a rotação) para a média dos pixels **dentro** do
contorno preenchido. **Validado ao vivo**: num lote de 15 tentativas após o
fix, as 3 ocorrências reais de "elementos idênticos" geraram cliques de
verdade (antes: lista vazia sempre) — ainda sem sucesso medido (amostra
pequena demais), mas o sintoma relatado ("nem foi tentado") está resolvido.

### "Laços fechados" via CNN — bug de canal de cor (BGR/RGB) e segmentação fundida

Investigando por que o CNN ainda "marcava errado" mesmo depois da
retentativa ler o critério certo a cada rodada, um sub-agente achou **dois
bugs novos, independentes das 4 causas já documentadas pra métrica
clássica**:

1. **Troca de canal BGR/RGB entre treino e inferência**: `solve_closed_loop_cnn`
   alimentava o modelo com recortes BGR crus, mas `predict()` assume
   entrada já em RGB (baseado em PIL) — o treino passava por arquivos PNG
   (corretamente RGB), então o modelo via cores espelhadas em produção
   desde que o CNN foi criado. Provavelmente responsável por boa parte do
   0/18 histórico, independente de qualquer problema de segmentação.
2. **A fusão de traços próximos numa única região** (causa-raiz #1 já
   documentada pra métrica clássica) afeta a entrada do CNN do mesmo jeito:
   nas 48 amostras reais de "laços fechados" acumuladas nesta sessão, pelo
   menos uma gerava zero candidatos (desafio inteiro fundido num blob só,
   descartado pelo filtro de área máxima) e a maioria sub-segmentava (2-4
   regiões onde havia 5+ traços visíveis).

Corrigido: nova máscara (`_closed_loop_mask`, borda no canal de **saturação**
HSV, não luminância — diferente da abordagem "limiar de saturação" já
tentada e documentada como falha, porque reage a *transições* de saturação,
não ao valor absoluto, sobrevivendo a casos onde o fundo é mais saturado que
o traço); conversão BGR→RGB corrigida antes de `predict()`; bug adicional
encontrado em `build_dataset.build()` (não limpava arquivos antigos antes
de reconstruir — corrigido com `shutil.rmtree`). Dataset reconstruído e
re-rotulado do zero com a nova segmentação: 42 exemplos de 11 amostras reais
(19 fechados/23 abertos) — bem mais balanceado que os 20 originais
(16/4). Checkpoint retreinado.

**Validação**: nas 48 amostras reais, zero agora geram 0 candidatos (antes,
pelo menos 1). Holdout de 5 seeds (75/25): 72% de acurácia média (50-80%)
sobre recortes reais nunca vistos — sinal real acima do chance (50%), amostra
pequena. Lote ao vivo (n=15): 2 ocorrências de "laços fechados", ambas
geraram pontos de clique reais e não-vazios (antes podiam vir vazios/sem
sentido) — ambas ainda falharam (amostra pequena demais pra medir melhora
de taxa de sucesso, mas o mecanismo está mecanicamente correto agora, o
que não era o caso antes).

### "Número do grupo" — sintoma era o bug arquitetural, dificuldade de OCR continua

Com `criterion_type` relido a cada rodada, "número do grupo" agora é
corretamente identificado e tentado toda vez que aparece (não é mais "nem
tentou"). A dificuldade de fundo já documentada (dígito oculto atrás de
padrão denso de listras, OCR com baixa taxa de leitura mesmo com rajada de
frames) continua — num lote de validação de 15, as 4 ocorrências reais
tiveram `points: []` (nenhum frame da rajada teve uma leitura do dígito-alvo
legível o bastante). Não é um bug novo, é a dificuldade já registrada em
`solve_digit_membership_temporal` se manifestando sem mais ser mascarada
pelo bug arquitetural acima.

### Flakiness residual do Selenium observada nos lotes de validação

Dois erros de infraestrutura apareceram em mais de um lote de validação
desta rodada, ambos não investigados a fundo: `stale element reference`
(1x no lote headed de 5, 1x no lote de 15) e `element click intercepted:
Element is not clickable at point (469, -9458)` (coordenada Y impossível —
mesma assinatura do bug já corrigido em `click_grid_tiles`, ver
`attacks/grid_interaction.py`, mas sobrevivendo às 2 retentativas já
implementadas lá). Como a mitigação existente (rebuscar elementos + até 2
retentativas de 0.3s) não eliminou o sintoma, a grade pode ficar instável
por mais tempo que o orçamento de retentativa cobre em alguns casos —
fica registrado como trabalho futuro, não investigado nesta sessão.

### Cloudflare Turnstile (`turnstile_behavioral_attempt`) — ligado

A Cloudflare não mantém uma página de demo oficial full-page (diferente do
Google/hCaptcha) — nota já registrada em `config/targets.yaml`. A
alternativa documentada pela própria Cloudflare pra testar uma integração
sem precisar de conta/domínio próprio são as *sitekeys de teste* publicadas
em developers.cloudflare.com/turnstile/troubleshooting/testing/, que
funcionam em qualquer origem: `1x00000000000000000000AA` (sempre passa),
`2x00000000000000000000AB` (sempre bloqueia), `3x00000000000000000000FF`
(sempre força desafio interativo).

```bash
python -m experiments.run_live_cli --target turnstile --technique behavioral_probe --n 5
python -m experiments.run_live_cli --target turnstile --technique behavioral_probe --n 1 \
    --sitekey 3x00000000000000000000FF
```

**Investigação ao vivo do DOM** (`harness/turnstile_test_server.py`,
docstring completa): o widget renderiza a UI visível (checkbox, texto,
logo) sem nenhum `<iframe>` e sem nenhum elemento acessível via
`querySelectorAll`/`innerHTML` no DOM principal — confirmado inspecionando
`document.documentElement.outerHTML` e testando `element.shadowRoot` em
todos os elementos da página (sempre `null`). Consistente com **Shadow DOM
fechado** (técnica legítima anti-adulteração do próprio navegador, não uma
limitação da ferramenta). Por isso a detecção de resultado não tenta
inspecionar DOM: usa os *callbacks* JS oficiais do Turnstile
(`data-callback`/`data-error-callback`/`data-expired-callback`),
verificados ao vivo contra as 3 sitekeys antes de escrever a técnica.

`harness/turnstile_test_server.py` serve localmente (via
`http.server.ThreadingHTTPServer` numa thread daemon) uma página mínima
embutindo o script oficial do widget com a sitekey escolhida, pra que o
Selenium navegue até ela como faria com qualquer alvo real.
`turnstile_behavioral_attempt` navega até essa página, aguarda o widget
resolver, e lê `window.__turnstileResult` (preenchido pelos callbacks)
pra mapear o resultado: token presente → `success`; código de erro
presente → `blocked`; nenhum dos dois (desafio interativo pendente sem
resposta) → `failure`.

**Validado ao vivo contra as 3 sitekeys de teste** — mapeamento correto em
todos os casos: sempre-passa → `success` (token `XXXX.DUMMY.TOKEN.XXXX`),
sempre-bloqueia → `blocked` (código de erro `600010`), sempre-força-desafio
→ `failure` (`widget_state: interactive_challenge_pending`).

**Limitação importante, documentada desde o início**: sitekeys de teste têm
comportamento FIXO — servem pra validar que o pipeline de coleta/medição
funciona de verdade, não pra medir a robustez real do Turnstile (que reage
a sinais comportamentais genuínos do navegador, não a uma sitekey de
teste). Medir robustez real exigiria uma sitekey Cloudflare própria do
aluno em modo Managed — passo que só o aluno pode fazer (precisa de
conta/domínio Cloudflare próprios), já apontado em `config/targets.yaml`.

## Solver de grid via CNN (sem LLM) — técnica experimental, não é mais a padrão

A pedido do usuário ("tira o v2 da OpenAI... treinar o CNN de verdade pra substituir"),
`attacks/image_grid_cnn/` passou de módulo nunca treinado (ver seu `dataset.py` original)
pra um classificador real, binário (carro/não-carro), treinado sobre células de grid reais.
**Atualização, mesmo dia**: ao testar ao vivo com o navegador aberto, o usuário observou
que o clique nos tiles previstos não acontecia de forma confiável (causa ainda não
diagnosticada) e pediu a reversão pro solver via LLM ("coloca o v2 na openAI") — ver nota
no topo da seção anterior. O código abaixo continua funcional e disponível como técnica
isolada (`--technique grid_cnn`), só não é mais chamado por `hcaptcha_combined_attempt`
nem é a técnica recomendada pro reCAPTCHA.

**Por que só "carros"**: um lote de 30 tentativas reais de coleta
(`recaptcha_baseline_attempt`, zero custo de API — a técnica só navega e captura o grid,
não tenta resolver) revelou pelo menos 9 categorias distintas pedidas pelo reCAPTCHA
(carros, motocicletas, semáforos, ônibus, bicicletas, faixas de pedestre, pontes, escadas,
montanhas) — treinar um classificador que generalizasse pra todas exigiria dado rotulado
em volume pra cada uma, fora de escopo aqui. "Carros" teve mais amostras reais nesse lote
(7 de 30) e virou o recorte tratável, mesmo espírito do classificador fechado/aberto de
`attacks/canvas_click_cnn/`.

**Dataset**: 60 células reais rotuladas (19 "carro", 41 "nao_carro") de 7 grids 3x3 reais —
rotulagem assistida por IA (Claude), mesma ressalva metodológica já registrada pro CNN de
laços fechados. 3 células descartadas por ambiguidade genuína entre carro e caminhão/van/
ônibus, não rotuladas como classe nenhuma.

```bash
python -m attacks.image_grid_cnn.build_dataset        # monta o dataset rotulado
python -m attacks.image_grid_cnn.train_recaptcha_carros  # treina e salva o checkpoint
python -m experiments.run_live_cli --target recaptcha --technique grid_cnn --n 10 --headed
```

**Escopo explícito, por desenho**: `solve_recaptcha_grid_cnn` só tenta resolver quando a
instrução pede "carro"/"carros" — qualquer outra categoria retorna `None` e a tentativa é
marcada `technique_scope: "not_applicable"`, nunca uma classificação arriscada sem dado de
treino. Essa é a mesma técnica agora usada em `hcaptcha_combined_attempt` pro tipo "grade"
do hCaptcha (reaproveitando o mesmo checkpoint, já que o problema — classificar célula de
grid — é o mesmo nos dois alvos).

**Resultado real, lote de 20 tentativas ao vivo (2026-11-05)**: 6 das 20 caíram em "carros"
(aplicável), **0/6 sucessos**. Amostra pequena demais pra conclusão estatística forte, mas
é o resultado real medido — bem menor que o 75% histórico do solver via LLM sobre o mesmo
alvo (ver seção anterior, mantida como comparação). Achado esperado: um classificador
binário treinado em 60 exemplos reais não tem o mesmo poder de generalização visual que um
modelo multimodal de propósito geral — a troca aqui prioriza não depender de API paga de
terceiro sobre taxa de sucesso bruta, uma decisão de escopo explícita do projeto, não um
resultado de pesquisa que diz que CNN pequeno "funciona pior" em geral.

**Estado atual em `hcaptcha_combined_attempt` após a reversão**: a grade voltou a usar
`solve_recaptcha_grid_adaptive_vote` (LLM), igual antes de 2026-11-05. O
arrastar-e-encaixar (antes ~57%-67% real via `solve_puzzle_piece`) **continua** sem
nenhuma técnica de resolução (`technique_scope: "not_applicable"`) -- esse ponto específico
não fazia parte do pedido de reversão, só a grade.

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
      acima). **75% continua sendo o resultado atual do projeto pro reCAPTCHA** — uma
      tentativa de substituir por um CNN sem LLM (2026-11-05) foi revertida no mesmo dia
      por clique não confiável ao vivo (ver "Solver de grid via CNN"). hCaptcha: 4 tipos de
      desafio distintos observados ao vivo (grade, ícone
      diferente, arrastar-e-encaixar, laços fechados — nenhum grade de tiles exceto o
      primeiro, `image_grid_cnn/` não se aplica aos outros 3), cada um com solver
      próprio — puxado quase todo pelo arrastar-e-encaixar (16/24 nas quatro rodadas, ~67%
      via LLM histórico — `combined` não chama mais essa técnica específica por pedido do
      usuário, arrastar-e-encaixar fica `not_applicable` por ora; a grade do hCaptcha
      dentro de `combined` voltou a usar o solver via LLM, igual reCAPTCHA);
      clique em ponto sobre canvas (ícone diferente, laços fechados, e um 6º tipo
      descoberto, "número que pertence a um grupo") ficou em **0/51 via LLM** (incluindo
      `gpt-6-astra`, 10x mais caro) **e 0/316 via CV clássica/OCR** (técnica alternativa sem
      LLM, `attacks/classical_canvas_solver.py` — lote de 500 interrompido pelo usuário em
      271 ao ficar claro que não funcionava, mais 20 após trocar a métrica de contraste de
      cor de "ícone diferente" de redmean para Delta-E CIE76 real, mais 25 depois de
      instrumentar a persistência das imagens reais em disco), mesmo depois de corrigir
      3 bugs reais de execução ao longo do caminho (detecção de bot, faixa preta do
      cabeçalho, borda/moldura do desafio mascarando os ícones), trocar a métrica de
      contraste de cor por uma perceptualmente mais fiel, e checar (sem achar evidência)
      a hipótese de que os desafios reciclassem um conjunto fixo de imagens, **e 0/18 via
      classificador CNN supervisionado** (`attacks/canvas_click_cnn/`, só "laços fechados",
      20 exemplos reais rotulados — amostra pequena demais pra conclusão estatística, mas
      sem sinal de melhora). **Conclusão**: os três tipos canvas permanecem sem solução em
      nenhuma das três técnicas tentadas, incluindo duas métricas de contraste de cor
      distintas na via CV clássica — resultado legítimo e documentável pro estudo
      comparativo. Arrastar-e-encaixar e grade continuam resolvidos pelo solver via
      LLM.
      Turnstile: **ligado** via `turnstile_behavioral_attempt` (`experiments/live.py`) +
      `harness/turnstile_test_server.py` — como a Cloudflare não tem demo full-page oficial,
      serve localmente uma página com cada uma das 3 sitekeys de teste publicadas pela
      própria Cloudflare (sempre-passa/sempre-bloqueia/sempre-força-desafio) e detecta o
      resultado via os callbacks JS oficiais do widget (DOM é Shadow DOM fechado, sem
      iframe nem elemento inspecionável — achado confirmado ao vivo), já que sitekeys de
      teste têm comportamento fixo e só validam o pipeline, não a robustez real (que exigiria
      sitekey própria do aluno em modo Managed). Rodar lotes maiores de coleta deve ter
      ritmo/volume combinado com o orientador (ver "Escopo ético" acima).
- [x] Atividade 7 — testes unitários (128 passed, 0 skipped com tesseract instalado) +
      comparação das técnicas (`analysis/compare.py`) (Out-Nov/2026)
- [ ] Atividade 8 — manutenção do pipeline (reativa, conforme forem surgindo
      mudanças de DOM/bloqueios durante a coleta real — Out-Nov/2026)
- [x] Atividade 9 — relatório comparativo consolidado (`analysis/report.py`,
      tabela LaTeX + gráfico por alvo) (Nov/2026)
