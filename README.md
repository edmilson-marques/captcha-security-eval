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

## Status

- [x] Atividade 1 — ambiente de testes + coletores (Ago/2026)
- [ ] Atividade 2 — módulos de ataque (Ago-Set/2026)
- [ ] Atividade 3 — técnicas de evasão (Ago-Set/2026)
- [ ] Atividades 4-6 — execução dos experimentos por alvo (Set-Out/2026)
- [ ] Atividade 7 — testes unitários completos + comparação das técnicas (Out-Nov/2026)
- [ ] Atividade 8 — manutenção do pipeline (Out-Nov/2026)
- [ ] Atividade 9 — relatório comparativo consolidado (Nov/2026)
