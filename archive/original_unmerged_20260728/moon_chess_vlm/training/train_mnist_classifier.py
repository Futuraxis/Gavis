"""独立的 MNIST/手写数字分类训练脚本。

这个脚本不使用 PPO，而是使用普通监督学习来完成数字分类。

支持两种数据来源：
1. 本地 MNIST IDX 文件
2. sklearn 自带 digits 数据集（8x8），适合作为零配置 smoke test
"""

from __future__ import annotations

import argparse
import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}


class ClassifierNet(nn.Module):
    """简单的 MLP 分类器。"""

    def __init__(self, input_dim: int, num_classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MNISTIDXDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """读取本地 MNIST IDX 文件。"""

    def __init__(self, images_path: Path, labels_path: Path) -> None:
        images = _read_idx_images(images_path).astype(np.float32) / 255.0
        labels = _read_idx_labels(labels_path).astype(np.int64)
        self.features = torch.from_numpy(images.reshape(images.shape[0], -1))
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


@dataclass(slots=True)
class DatasetBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    source_name: str


def build_dataloaders(
    *,
    data_dir: Path,
    batch_size: int,
    dataset: str,
) -> DatasetBundle:
    if dataset in {"mnist", "auto"} and _has_local_mnist(data_dir):
        train_dataset = MNISTIDXDataset(
            images_path=_resolve_idx_path(data_dir / MNIST_FILES["train_images"]),
            labels_path=_resolve_idx_path(data_dir / MNIST_FILES["train_labels"]),
        )
        test_dataset = MNISTIDXDataset(
            images_path=_resolve_idx_path(data_dir / MNIST_FILES["test_images"]),
            labels_path=_resolve_idx_path(data_dir / MNIST_FILES["test_labels"]),
        )
        return DatasetBundle(
            train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
            input_dim=train_dataset.features.shape[1],
            source_name="local_mnist_idx",
        )

    if dataset in {"sklearn-digits", "auto"}:
        try:
            from sklearn.datasets import load_digits
        except ModuleNotFoundError as exc:
            if dataset == "sklearn-digits":
                raise RuntimeError(
                    "未安装 scikit-learn，无法使用 sklearn digits。请安装 scikit-learn，或提供本地 MNIST IDX 文件。"
                ) from exc
        else:
            digits = load_digits()
            features = torch.tensor(digits.data / 16.0, dtype=torch.float32)
            labels = torch.tensor(digits.target, dtype=torch.long)
            full_dataset = TensorDataset(features, labels)
            train_size = int(len(full_dataset) * 0.8)
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = random_split(
                full_dataset,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )
            return DatasetBundle(
                train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
                input_dim=features.shape[1],
                source_name="sklearn_digits",
            )

    raise RuntimeError(
        "没有找到可用数据集。请提供本地 MNIST IDX 文件，或安装 scikit-learn 后使用 sklearn digits。"
    )


def train(
    *,
    data_dir: Path,
    dataset: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    save_path: Path | None,
    device: str | None = None,
) -> dict[str, float | str]:
    bundle = build_dataloaders(data_dir=data_dir, batch_size=batch_size, dataset=dataset)
    resolved_device = _resolve_device(device)
    model = ClassifierNet(input_dim=bundle.input_dim).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for features, labels in bundle.train_loader:
            features = features.to(resolved_device)
            labels = labels.to(resolved_device)

            logits = model(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size_actual
            total_examples += batch_size_actual

        train_loss = total_loss / max(total_examples, 1)
        test_accuracy = evaluate(model, bundle.test_loader, resolved_device)
        print(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "test_accuracy": round(test_accuracy, 6),
                "dataset": bundle.source_name,
            }
        )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": bundle.input_dim,
                "dataset": bundle.source_name,
            },
            save_path,
        )

    final_accuracy = evaluate(model, bundle.test_loader, resolved_device)
    return {
        "dataset": bundle.source_name,
        "test_accuracy": final_accuracy,
        "device": resolved_device.type,
    }


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
    return correct / max(total, 1)


def _resolve_device(device: str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def _has_local_mnist(data_dir: Path) -> bool:
    return all(_resolve_idx_path(data_dir / name).exists() for name in MNIST_FILES.values())


def _resolve_idx_path(path: Path) -> Path:
    if path.exists():
        return path
    gz_path = Path(f"{path}.gz")
    if gz_path.exists():
        return gz_path
    return path


def _read_idx_images(path: Path) -> np.ndarray:
    with _open_maybe_gzip(path) as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"非法 MNIST 图像文件: {path}")
        buffer = handle.read(count * rows * cols)
        return np.frombuffer(buffer, dtype=np.uint8).reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with _open_maybe_gzip(path) as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"非法 MNIST 标签文件: {path}")
        buffer = handle.read(count)
        return np.frombuffer(buffer, dtype=np.uint8)


def _open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def main() -> None:
    parser = argparse.ArgumentParser(description="训练一个普通的手写数字分类模型。")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument(
        "--dataset",
        choices=["auto", "mnist", "sklearn-digits"],
        default="auto",
        help="优先使用本地 MNIST；没有的话可退回 sklearn digits。",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/mnist_classifier.pt"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    result = train(
        data_dir=args.data_dir,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
