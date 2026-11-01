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

**Resultado real medido em 2026-11-01 (4 tentativas): 0/4 sucessos.** Achado
documentado, não bug: 3 de 4 caíram em grades 3x3 "dinâmicas" (o reCAPTCHA troca o
tile por uma foto nova ao ser clicado — a implementação atual só faz uma passada,
então não vence esse tipo por design); a 4ª foi uma grade 4x4 estática onde o
próprio modelo super-segmentou (`[8,9,12,13]` quando só `9,13` tinham a moto).
Ver docstring de `recaptcha_llm_vision_attempt` em `src/experiments/live.py` para o
detalhamento e a melhoria de próximo passo (loop de refinamento iterativo).

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
- [~] Atividades 4-6 — infraestrutura pronta e **ligada de verdade** para o reCAPTCHA:
      `baseline_no_select` e `llm_vision` (`experiments/live.py`), rodadas reais via
      `run_live_cli.py`. hCaptcha só tem `baseline_no_select` por enquanto (os dois tipos
      de desafio observados ao vivo — arrastar-e-encaixar, "laços fechados" — não são
      grade de tiles). **Falta**: loop de refinamento iterativo do reCAPTCHA para grades
      3x3 dinâmicas (achado real: 0/4 sucessos do `llm_vision` até agora, 3 por essa
      limitação — ver seção acima), e ligar o Turnstile (precisa de página própria).
      Rodar lotes maiores de coleta deve ter ritmo/volume combinado com o orientador
      (ver "Escopo ético" acima).
- [x] Atividade 7 — testes unitários (72 passed/3 skipped) + comparação das
      técnicas (`analysis/compare.py`) (Out-Nov/2026)
- [ ] Atividade 8 — manutenção do pipeline (reativa, conforme forem surgindo
      mudanças de DOM/bloqueios durante a coleta real — Out-Nov/2026)
- [x] Atividade 9 — relatório comparativo consolidado (`analysis/report.py`,
      tabela LaTeX + gráfico por alvo) (Nov/2026)
