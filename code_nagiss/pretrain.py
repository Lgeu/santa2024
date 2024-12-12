import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from constants import LIST_NUM_WORDS, LIST_WORD_TO_ID, PATH_SAVE
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
from util import load_score_memo


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.se = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.se(out)
        out += x
        out = self.relu(out)
        return out


class SantaNet(nn.Module):
    def __init__(self, vocab_size: int, channels: int, num_blocks: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, channels)
        self.conv_stem = nn.Conv1d(
            channels, channels, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.relu_stem = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )
        self.fc = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        x = self.embedding(x)  # (B, L, channels)
        x = x.transpose(1, 2)  # (B, channels, L)
        x = self.conv_stem(x)
        x = self.relu_stem(x)
        for block in self.blocks:
            x = block(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, channels)
        x = self.fc(x)  # (B, 1)
        return x


class ScoreDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = self.X[idx].long()
        y = self.y[idx]
        return X, y


def prepare_dataset(
    training_data: list[dict[bytes, float]], problem_id: int
) -> tuple[dict[str, int], Dataset]:
    word_to_id = LIST_WORD_TO_ID[problem_id]
    length = LIST_NUM_WORDS[problem_id]
    if problem_id in [1, 2]:
        raise ValueError("problem_id 1 and 2 are not supported")
    num_data = len(training_data[problem_id])
    X = torch.empty((num_data, length), dtype=torch.int8)
    y = torch.empty(num_data, dtype=torch.float)
    for idx, (compressed_text, score) in enumerate(
        tqdm(training_data[problem_id].items(), mininterval=30)
    ):
        assert len(compressed_text) == length
        X[idx] = torch.tensor(list(compressed_text), dtype=torch.int8)
        y[idx] = math.log(score)
    print(f"{word_to_id=}")
    dataset = ScoreDataset(X, y)
    return word_to_id, dataset


def train_model(problem_id: int = 5) -> None:
    _, training_data = load_score_memo()
    path_pretrain = PATH_SAVE / Path("pretrain")
    path_pretrain.mkdir(parents=True, exist_ok=True)
    num_epochs = 20
    batch_size = 4096
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word_to_id, dataset = prepare_dataset(training_data, problem_id)
    del training_data

    # 5% を validation に
    total_size = len(dataset)
    val_size = int(total_size * 0.05)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, shuffle=False
    )

    model = SantaNet(vocab_size=len(word_to_id), channels=128, num_blocks=12)
    model = model.to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.AdamW(model.parameters(), lr=0.005)

    for epoch in range(num_epochs):
        # 学習
        model.train()
        running_loss = 0.0
        count = 0
        pbar = tqdm(
            train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", mininterval=30
        )
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)  # (B, 1)
            loss = F.l1_loss(outputs.squeeze(-1), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            count += X.size(0)
            pbar.set_postfix(refresh=False, loss=loss.item())

        avg_loss = running_loss / count

        # 評価
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds: torch.Tensor = model(X).squeeze(-1)
                all_preds.extend(preds.detach().cpu().tolist())
                all_targets.extend(y.detach().cpu().tolist())
        corr, _ = spearmanr(all_targets, all_preds)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, "
            f"Val Spearman: {corr:.4f}"
        )

        torch.save(
            {"model": model.state_dict(), "word_to_id": word_to_id},
            path_pretrain / f"model_{problem_id}_epoch_{epoch+1}.pt",
        )


if __name__ == "__main__":
    train_model()
