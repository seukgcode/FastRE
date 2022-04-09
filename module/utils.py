from typing import List, Tuple, Union, Set
import numpy as np
import paddle
from paddle import fluid


def find(source: List[str], target: List[str]) -> Tuple[int, int]:
    """Fina the start and end position of entities in text."""
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i, i + target_len
    return -1, -1


def pad(batch: List, padding: Union[int, np.ndarray], padding_type: str, max_len: int = None) -> np.ndarray:
    """Pad the data in one batch."""
    if max_len is None:
        length_batch = [len(seq) for seq in batch]
        max_len = max(length_batch)
    return np.array(
        [np.concatenate([seq, [padding] * (max_len - len(seq))]) if len(seq) < max_len else seq for seq in batch],
        dtype=padding_type)


def pos(x: Union[paddle.Tensor, List]) -> paddle.Tensor:
    """Get position ids."""
    if isinstance(x, list) and len(x) == 2:
        x, r = x
    else:
        r = 0
        r = fluid.dygraph.to_variable(np.array([r], dtype="int64"))
    batch_size = x.shape[0]
    sent_len = x.shape[1]
    pid = fluid.layers.range(0, sent_len, step=1, dtype="int64")
    pid = fluid.layers.unsqueeze(pid, [0])
    pid = fluid.layers.expand(pid, [int(batch_size), 1])
    pid = fluid.layers.abs(pid - r)
    return pid


def gather(seq: paddle.Tensor, idx: paddle.Tensor) -> paddle.Tensor:
    """Gather the subject start and end features."""
    idx = idx.astype("int32")
    batch_idx = fluid.layers.range(0, seq.shape[0], step=1, dtype="int32")
    batch_idx = fluid.layers.unsqueeze(batch_idx, [1])
    idx = fluid.layers.concat([batch_idx, idx], 1)
    r = fluid.layers.gather_nd(seq, idx)
    return r


def partial(pred_set: Set, gold_set: Set) -> Tuple[Set, Set]:
    """Under partial match."""
    pred = {(i[0].split()[0], i[1], i[2].split()[0]) for i in pred_set}
    gold = {(i[0].split()[0], i[1], i[2].split()[0]) for i in gold_set}
    return pred, gold


def cross_entropy(probs: paddle.Tensor, labels: paddle.Tensor, mask: paddle.Tensor, eps: float = 1e-7) -> paddle.Tensor:
    """Cross entropy, including gradient clip."""
    probs = fluid.layers.clip(probs, eps, 1 - eps)
    logits = fluid.layers.log(probs / (1 - probs))
    loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, labels)
    loss = fluid.layers.reduce_sum(loss * mask)
    loss = loss / fluid.layers.reduce_sum(mask)
    return loss


def at_loss(logits: paddle.Tensor, labels: paddle.Tensor, mask: paddle.Tensor) -> paddle.Tensor:
    """Adaptive thresholding Loss"""
    t = fluid.layers.zeros(shape=[labels.shape[0], labels.shape[1], 1], dtype="float32")
    labels = fluid.layers.concat([t, labels], axis=-1)
    th_label = fluid.layers.zeros_like(labels)
    th_label[:, :, 0] = 1.0
    p_mask = labels + th_label
    n_mask = 1 - labels
    logit1 = logits - (1 - p_mask) * 1e30
    loss1 = - fluid.layers.reduce_sum(fluid.layers.log(fluid.layers.softmax(logit1) + 1e-7) * labels, dim=-1)
    logit2 = logits - (1 - n_mask) * 1e30
    loss2 = - fluid.layers.reduce_sum(fluid.layers.log(fluid.layers.softmax(logit2) + 1e-7) * th_label, dim=-1)
    loss = fluid.layers.reduce_sum(loss1 + loss2) / fluid.layers.reduce_sum(mask)
    return loss
