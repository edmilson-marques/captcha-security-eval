"""Dataset de desafios em grade de imagem — reCAPTCHA v2 / hCaptcha (Atividade 2, Ago-Set/2026).

Observação importante sobre a origem dos dados: a Atividade 1 (`src/harness`)
coleta o desafio **inteiro** (o grid completo, ex. 3x3 ou 4x4 células) como um
único screenshot em `data/raw/<alvo>/<timestamp>/challenge.png`. O recorte
desse screenshot em células individuais, cada uma salva isoladamente e
rotulada pela classe-alvo (ex. `travessia_de_pedestres/`, `onibus/`,
`nenhum/`), é um passo manual de preparação de dataset que ainda será feito
pelo aluno quando houver amostras reais em volume suficiente — não é
implementado aqui. Este módulo assume que esse recorte/rotulagem já existe em
disco, na convenção `root/<nome_da_classe>/*.png` (mesma convenção usada pelo
`torchvision.datasets.ImageFolder`), e também é a convenção usada pelo dataset
sintético gerado nos testes (ver `tests/test_image_grid_cnn_inference.py`).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from attacks.image_grid_cnn.model import INPUT_SIZE

# Transform default compartilhado entre treino (`dataset.py`) e inferência
# (`infer.py`), para garantir que uma imagem seja pré-processada exatamente da
# mesma forma nos dois caminhos. Normalização simples (média/desvio 0.5 por
# canal) — não são as estatísticas do ImageNet, já que o modelo é treinado do
# zero sobre os próprios dados do projeto.
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def preprocess_image(image: Image.Image) -> Tensor:
    """Aplica o transform default do projeto a uma imagem PIL já em modo RGB.

    Reutilizada por `infer.py` para garantir o mesmo pré-processamento usado
    no treinamento.
    """
    return DEFAULT_TRANSFORM(image)


class ImageGridDataset(Dataset):
    """Dataset de células de grid rotuladas, no layout `root/<classe>/*.png`.

    Encapsula `torchvision.datasets.ImageFolder` para já acoplar o
    resize/normalize default do projeto (`DEFAULT_TRANSFORM`), já que os
    dados reais rotulados ainda não existem (ver docstring do módulo).
    """

    def __init__(self, samples_dir: str | Path, transform=None) -> None:
        self.samples_dir = Path(samples_dir)
        self._image_folder = ImageFolder(
            root=str(self.samples_dir),
            transform=transform or DEFAULT_TRANSFORM,
        )

    @property
    def classes(self) -> list[str]:
        """Nomes das classes, na ordem dos índices usados pelos rótulos."""
        return self._image_folder.classes

    def __len__(self) -> int:
        return len(self._image_folder)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self._image_folder[index]
