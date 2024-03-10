import math
import time

import matplotlib.pyplot as plt
import torch
from utils.batchifier import get_batch
from utils.modeler import repackage_hidden

from .evaluation import evaluate


def train_model(
    model,
    train_batched_data,
    valid_batched_data,
    epochs,
    batch_size,
    bptt,
    criterion,
    optimizer,
    clip_threshold,
    log_interval,
    lr,
    dropout,
):
    train_losses = []
    valid_losses = []
    train_perplexities = []
    valid_perplexities = []
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        start_time = time.time()
        hidden = model.init_hidden(batch_size)

        for batch_idx, i in enumerate(range(0, train_batched_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_batched_data, i, bptt)
            model.zero_grad()

            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, model.ntokens), targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0 and batch_idx > 0:
                cur_loss = total_loss / log_interval
                cur_ppl = math.exp(cur_loss)
                elapsed = time.time() - start_time

                val_loss = evaluate(
                    model, valid_batched_data, criterion, batch_size, bptt
                )
                val_ppl = math.exp(val_loss)

                print(
                    f"epoch {epoch:3d}: {batch_idx:5d}/{len(train_batched_data) // bptt:5d} "
                    f"batches | lr {lr:02.2f} | dropout {dropout:.2f} | ms/batch {elapsed * 1000 / log_interval:5.2f} | "
                    f"train loss {cur_loss:5.2f} | train ppl {cur_ppl:8.2f} | "
                    f"valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}"
                )

                train_losses.append(cur_loss)
                valid_losses.append(val_loss)
                train_perplexities.append(cur_ppl)
                valid_perplexities.append(val_ppl)

                total_loss = 0
                start_time = time.time()

    hyperparams_str = f"bptt={bptt}_lr={lr}_clip={clip_threshold}_dropout={dropout}"
    plot_title = f"Training and Validation Losses and Perplexities ({hyperparams_str})"

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.title(plot_title)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(train_perplexities, label="Training Perplexity")
    plt.plot(valid_perplexities, label="Validation Perplexity")
    plt.xlabel("Steps")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.grid(True)

    filename = f"images/rnn_train_val_losses_ppl_{hyperparams_str}.png"
    plt.savefig(filename)

    final_train_loss = train_losses[-1] if train_losses else None
    final_val_loss = valid_losses[-1] if valid_losses else None
    final_train_ppl = (
        math.exp(final_train_loss) if final_train_loss is not None else None
    )
    final_val_ppl = math.exp(final_val_loss) if final_val_loss is not None else None

    return final_train_loss, final_val_loss, final_train_ppl, final_val_ppl
