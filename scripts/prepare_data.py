"""Deduplica por id, hace el split 75/25 y lo guarda en data/processed/."""

import logging

from src import config
from src.data import load_test, load_train, prepare_splits


def main() -> None:
    """Genera train.parquet y test.parquet y muestra un resumen."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prepare_splits()

    train = load_train()
    # Solo se lee el test para reportar tamano y tasa de fraude, no para decidir nada.
    test = load_test()
    for nombre, df in (("train", train), ("test", test)):
        tasa = df[config.TARGET_COLUMN].mean()
        print(f"{nombre}: {df.height} filas, tasa de fraude {tasa:.2%}")


if __name__ == "__main__":
    main()
