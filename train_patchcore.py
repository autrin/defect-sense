from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
import torch


def main() -> None:
    torch.set_float32_matmul_precision("high")

    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category="bottle",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )

    engine = Engine(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
    )

    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)
    print(results)


if __name__ == "__main__":
    main()