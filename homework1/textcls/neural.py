"""Fine-tune BERT-base-uncased with a specified context and training budget."""

import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import write_json
from .metrics import evaluate, save_test_outputs

MAX_LENGTH = 64
EPOCHS = 3


class EncodedDataset(Dataset):
    def __init__(self, encodings, labels, indices):
        self.encodings = encodings
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        return {**{key: value[row] for key, value in self.encodings.items()},
                "labels": self.labels[row]}


def precision_context(use_bf16, device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_bf16 else nullcontext()


@torch.no_grad()
def predict(model, loader, device, use_bf16=False):
    model.eval()
    predictions = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        with precision_context(use_bf16, device):
            logits = model(**batch).logits
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return np.asarray(predictions)


def training_class_weights(labels, indices, n_classes):
    counts = np.bincount(labels[indices], minlength=n_classes)
    return len(indices) / (n_classes * counts)


def encode_documents(texts, tokenizer, max_length, position="prefix"):
    if position == "prefix":
        return tokenizer(texts, truncation=True, padding="max_length",
                         max_length=max_length, return_tensors="pt")
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    tokenized = tokenizer(texts, add_special_tokens=False, truncation=False, verbose=False)
    rows = []
    for tokens in tokenized["input_ids"]:
        if len(tokens) > budget:
            if position == "tail":
                tokens = tokens[-budget:]
            elif position == "head_tail":
                head = (budget + 1) // 2
                tokens = tokens[:head] + tokens[-(budget - head):]
            else:
                raise ValueError(f"Unknown document position: {position}")
        rows.append(tokenizer.prepare_for_model(tokens, return_attention_mask=True,
                                               return_token_type_ids=True, verbose=False))
    return tokenizer.pad(rows, padding="max_length", max_length=max_length, return_tensors="pt")


def run_bert(frame, labels, names, split, args, *, method="bert",
             max_length=MAX_LENGTH, balanced=False, epochs=EPOCHS,
             position="prefix", encodings=None):
    start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    directory = Path(args.output) / method
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    if encodings is None:
        encodings = encode_documents(frame["text"].tolist(), tokenizer, max_length, position)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.bert_model, num_labels=len(names),
        id2label={i: name for i, name in enumerate(names)},
        label2id={name: i for i, name in enumerate(names)},
    ).to(device)
    if balanced:
        weights = training_class_weights(labels, split["train"], len(names))
        loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
        )
        write_json(directory / "class_weights.json", dict(zip(names, weights.tolist())))
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        key: DataLoader(
            EncodedDataset(encodings, labels, indices),
            batch_size=args.batch_size if key == "train" else args.eval_batch_size,
            shuffle=key == "train", generator=generator if key == "train" else None,
            pin_memory=device.type == "cuda",
        )
        for key, indices in split.items()
    }
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=2e-5)
    total_steps = epochs * len(loaders["train"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    best_f1 = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, seen = 0.0, 0
        for step, batch in enumerate(loaders["train"], 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with precision_context(use_bf16, device):
                if balanced:
                    inputs = {key: value for key, value in batch.items() if key != "labels"}
                    loss = loss_function(model(**inputs).logits, batch["labels"])
                else:
                    loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch_rows = len(batch["labels"])
            loss_sum += loss.item() * batch_rows
            seen += batch_rows
            if step % 100 == 0:
                print(f"BERT epoch {epoch}/{epochs} step {step}/{len(loaders['train'])} "
                      f"loss={loss_sum / seen:.4f}", flush=True)
        val_prediction = predict(model, loaders["validation"], device, use_bf16)
        val_metrics = evaluate(labels[split["validation"]], val_prediction, len(names))
        entry = {"epoch": epoch, "train_loss": loss_sum / seen,
                 "training_examples": seen, "training_steps": len(loaders["train"]),
                 "validation_accuracy": val_metrics["accuracy"],
                 "validation_macro_f1": val_metrics["macro_f1"],
                 "seconds": time.perf_counter() - epoch_start}
        history.append(entry)
        write_json(directory / "training_history.json", history)
        print(f"BERT {entry}", flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_entry = val_metrics["macro_f1"], entry
            model.save_pretrained(directory / "best_model")
            tokenizer.save_pretrained(directory / "best_model")
    del optimizer, scheduler, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(directory / "best_model").to(device)
    test_prediction = predict(model, loaders["test"], device, use_bf16)
    test_metrics = evaluate(labels[split["test"]], test_prediction, len(names))
    save_test_outputs(args.output, method, frame, split["test"],
                      labels[split["test"]], test_prediction, names)
    result = {
        "method": method, "validation_accuracy": best_entry["validation_accuracy"],
        "validation_macro_f1": best_entry["validation_macro_f1"],
        "test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"],
        "seconds": time.perf_counter() - start, "selected_epoch": best_entry["epoch"],
        "pretrained_model": "google-bert/bert-base-uncased", "model_path": args.bert_model,
        "max_length": max_length, "epochs": epochs, "input_position": position, "learning_rate": 2e-5,
        "training_seed": args.seed,
        "class_weighting": "inverse-frequency" if balanced else "uniform",
        "batch_size": args.batch_size, "weight_decay": 0.01,
        "warmup_ratio": 0.1, "precision": "bf16" if use_bf16 else "fp32",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    write_json(directory / "metrics.json", result)
    yield result
