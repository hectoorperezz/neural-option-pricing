import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.generator import OptionDataset
from src.training import PriceLoss, Trainer


def test_trainer_fits_one_epoch_and_tracks_best_state() -> None:
    features = torch.linspace(0.0, 1.0, 32).reshape(-1, 1)
    prices = 2.0 * features
    dataset = OptionDataset(
        features=features,
        prices=prices,
        deltas=None,
        raw_inputs=features,
        input_names=("moneyness",),
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = nn.Linear(1, 1)
    trainer = Trainer(
        model=model,
        loss_fn=PriceLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=loader,
        validation_loader=loader,
    )

    history = trainer.fit(epochs=3)

    assert len(history) == 3
    assert trainer.best_state_dict is not None
    assert trainer.best_validation_mae < float("inf")
    trainer.load_best()
