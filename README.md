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

## Solver de visão via LLM (opcional, custo real)

`src/attacks/llm_vision_solver.py` implementa uma técnica alternativa de IA: envia o
screenshot do desafio + a instrução para um modelo multimodal da Anthropic e recebe de
volta uma resposta estruturada (índices de tiles, ou coordenadas de um puzzle). Requer
`ANTHROPIC_API_KEY` no ambiente — **não incluída no repositório, e não testada ao vivo**
(sem chave disponível na sessão em que foi escrita; os testes usam um cliente mockado).
Custo real por chamada — ver docstring do módulo para o modelo escolhido e a estimativa.
Antes de usar em experimentos de verdade, validar manualmente contra amostras reais.

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
- [~] Atividades 4-6 — **infraestrutura pronta** (`experiments/runner.py` +
      `experiments/metrics.py`, testável por injeção de dependência, sem rede).
      **Falta**: ligar um `attempt_fn` de verdade por (alvo, técnica) — isto é,
      abrir o navegador real contra a demo, aplicar a técnica/evasão e
      interpretar o resultado — e então rodar os lotes de coleta em si contra
      Google/hCaptcha/Cloudflare. Isso é trabalho exploratório ao vivo (a
      estrutura do DOM de cada desafio só se confirma inspecionando a página
      de verdade) e deve ser feito com ritmo/volume combinado com o orientador
      (ver "Escopo ético" acima) — não foi automatizado sem supervisão.
- [x] Atividade 7 — testes unitários (72 passed/3 skipped) + comparação das
      técnicas (`analysis/compare.py`) (Out-Nov/2026)
- [ ] Atividade 8 — manutenção do pipeline (reativa, conforme forem surgindo
      mudanças de DOM/bloqueios durante a coleta real — Out-Nov/2026)
- [x] Atividade 9 — relatório comparativo consolidado (`analysis/report.py`,
      tabela LaTeX + gráfico por alvo) (Nov/2026)
