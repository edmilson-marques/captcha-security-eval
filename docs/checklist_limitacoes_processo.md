# Checklist de limitações não-automatizáveis

Complemento aos fixes de código (`dataset_prep.validate_and_merge_labels`,
`dataset_prep.plan_labeling_chunks`, `train.cross_validate(validation_eligible=...)`,
`dataset_prep.is_blank_capture`) pros problemas achados ao vivo durante a coleta/treino
(Nov/2026). Os itens abaixo **não têm correção de código** -- são ruído inerente ao
experimento ou julgamento que só um humano revisando o resultado consegue aplicar.
Repetir essa checagem antes de reportar qualquer número novo no TCC.

## 1. Ruído anti-bot do alvo ao vivo

A demo pública do reCAPTCHA roda detecção de automação real (fingerprint, timing,
histórico de IP) -- uma fração dos "erros" de um lote ao vivo é rejeição por suspeita de
bot, não erro de classificação da CNN. Não dá pra automatizar a distinção a partir do
resultado agregado (sucesso/falha) sozinho.

- [ ] Ao reportar uma taxa de acerto ao vivo, registrar também a taxa de desafios que
      nem chegaram a abrir um grid (indício de bloqueio antecipado).
- [ ] Nunca comparar taxa de acerto ao vivo direto com acurácia de validação (CV) como se
      medissem a mesma coisa -- são números de natureza diferente (ver item 3).
- [ ] Se a taxa de rejeição subir de forma abrupta entre lotes do mesmo dia, suspeitar de
      bloqueio por volume antes de suspeitar do modelo.

## 2. Efeito de erro composto em desafios encadeados

Quando o Google reemite um novo desafio na mesma tentativa (`max_challenges` > 1), um erro
de classificação no primeiro desafio pode mudar qual desafio aparece em seguida -- a
tentativa inteira falha por um motivo que não aparece isolado em nenhuma métrica por
categoria. Não dá pra atribuir automaticamente "qual categoria causou a falha" numa
tentativa com múltiplos desafios encadeados.

- [ ] Ao analisar uma tentativa falha com mais de um desafio na sequência, revisar
      manualmente o screenshot de cada desafio antes de atribuir a falha a uma categoria.
- [ ] Reportar separadamente "taxa de acerto por desafio individual" e "taxa de sucesso
      da tentativa completa" -- a segunda é sempre menor e mistura efeitos.

## 3. Significância estatística em lotes pequenos

Lotes de 20-100 tentativas ao vivo têm margem de erro grande o bastante pra mascarar
diferenças reais ou inventar diferenças que não existem (ver teste BGR/RGB: 14,0% vs
9,6% em dois lotes de 100, z≈0,87, não-significativo). Nenhum script decide sozinho se
uma mudança "funcionou" -- precisa do teste.

- [ ] Toda comparação antes/depois de uma mudança de código usa teste de proporção de
      duas amostras (two-proportion z-test) antes de qualquer afirmação de melhora/piora.
- [ ] Reportar o z (ou p-valor) junto do percentual, não só o percentual bruto.
- [ ] Lote <100 tentativas: tratar qualquer diferença abaixo de ~10 pontos percentuais
      como possível ruído até um lote maior confirmar.

## 4. Risco de corrida entre forks paralelos de rotulagem

Rodar múltiplos sub-agentes de rotulagem em paralelo sobre arquivos compartilhados (ex.
um script auxiliar comum) pode gerar sobrescrita cruzada -- já observado uma vez nesta
sessão (`make_sheets.py` sobrescrito por um fork irmão). `plan_labeling_chunks` evita
chunks desbalanceados, mas não evita colisão de arquivo.

- [ ] Cada fork de rotulagem paralelo grava em um diretório/arquivo próprio, nunca
      compartilha um script ou planilha auxiliar com outro fork rodando ao mesmo tempo.
- [ ] Mesclar os resultados dos forks só depois que todos terminaram, nunca durante.

## 5. Revisão manual de contaminação por selo de "check"

`dataset_prep.has_checkmark_overlay` existe só como **pré-triagem** (ver docstring) --
teve falso-positivo grave (~5-10% das células reais) quando usado como filtro automático
e foi revertido. Decidir se uma célula sinalizada é selo real ou falso-positivo (parede/
carro azul saturado) exige olhar a imagem.

- [ ] Rodar `has_checkmark_overlay` só pra gerar uma lista curta de candidatos.
- [ ] Abrir cada candidato visualmente antes de descartar a célula do dataset -- nunca
      descartar em lote sem revisão.

## 6. Origem do rótulo (vale para o dataset inteiro)

Rotulagem assistida por IA (Claude), não pelo autor do TCC (ressalva já registrada em
`build_dataset.py`/`category_models.py`). Isso não é um "bug" corrigível -- é uma
limitação metodológica a declarar no texto do TCC toda vez que a acurácia do dataset for
citada como se fosse rótulo humano independente.

- [ ] Toda menção à acurácia do classificador no texto do TCC cita essa ressalva (nota
      de rodapé ou parágrafo de limitações), não só na documentação de código.
