"""
Targeted, asserted, idempotent patches for the ProgressiveTransformersSLP repository
(verified against commit ff8b28cb0817d177f63dfa18c7c322806ee5dd96).

Every edit checks that the code it expects is really there and stops with a clear message if the
repository has changed. Original files are copied to <repo>/_originals before the first edit.
"""
import re
import shutil
from pathlib import Path

MARK = "# [isign-patch]"
TESTED_COMMIT = "ff8b28cb0817d177f63dfa18c7c322806ee5dd96"


def _replace_block(src, start_marker, end_marker, new_lines, what):
    """Replace the lines from the one containing start_marker through the one containing end_marker."""
    lines = src.split("\n")
    starts = [i for i, l in enumerate(lines) if start_marker in l]
    ends = [i for i, l in enumerate(lines) if end_marker in l]
    assert len(starts) == 1 and len(ends) == 1, \
        f"{what}: expected exactly one start and one end line, found {len(starts)}/{len(ends)}"
    s, e = starts[0], ends[0]
    assert 0 <= e - s <= 12, f"{what}: unexpected block size ({e - s + 1} lines)"
    return "\n".join(lines[:s] + new_lines + lines[e + 1:])


def _once(src, old, what):
    assert src.count(old) == 1, f"{what}: expected exactly one occurrence, found {src.count(old)}"


# ------------------------------------------------------------------------------------ vocabulary.py
def patch_vocabulary(s, ctx):
    s2 = re.sub(r"^from torchtext[^\n]*\n", "", s, flags=re.M)
    assert s2 != s, "no torchtext import found"
    assert "dataset: Dataset" in s2, "build_vocab signature not found"
    s2 = s2.replace("dataset: Dataset", "dataset")
    new_lines = [
        f"        known = set(self.itos)  {MARK} O(1) membership test instead of scanning the list",
        "        for t in tokens:",
        "            new_index = len(self.itos)",
        "            if t not in known:",
        "                known.add(t)",
        "                self.itos.append(t)",
        "                self.stoi[t] = new_index",
    ]
    return _replace_block(s2, "for t in tokens:", "self.stoi[t] = new_index", new_lines, "vocabulary.add_tokens")


# ------------------------------------------------------------------------------------ helpers.py
def patch_helpers(s, ctx):
    s2 = re.sub(r"^from torchtext[^\n]*\n", "", s, flags=re.M)
    assert s2 != s, "no torchtext import found"
    old = "from torch.utils.tensorboard import SummaryWriter\n"
    _once(s2, old, "tensorboard import")
    new = ("try:\n"
           "    from torch.utils.tensorboard import SummaryWriter\n"
           "except Exception:  # only needed for training logs\n"
           "    SummaryWriter = None\n")
    return s2.replace(old, new)


# ------------------------------------------------------------------------------------ transformer_layers.py
def patch_transformer_layers(s, ctx):
    old = "max_len: int = 200000"
    _once(s, old, "positional-encoding max_len")
    return s.replace(old, f"max_len: int = {ctx['pe_max_len']}")


# ------------------------------------------------------------------------------------ batch.py
def patch_batch(s, ctx):
    old = "        self.trg_mask = None\n"
    _once(s, old, "trg_mask initialisation")
    s = s.replace(old, old + f"        self.frame_mask = None  {MARK}\n"
                            "        self.coord_mask = None\n"
                            "        self.frame_lengths = None\n")

    new_block = [
        f"            {MARK} masks come from true lengths / validity flags, never from coordinate values",
        "            seq_len = self.trg_input.shape[1]",
        "            self.frame_lengths = getattr(torch_batch, \"frame_lengths\", None)",
        "            if self.frame_lengths is not None:",
        "                input_lengths = (self.frame_lengths.cpu() - 1).clamp_min(0)",
        "                valid_steps = torch.arange(seq_len).unsqueeze(0) < input_lengths.unsqueeze(1)",
        "            else:",
        "                valid_steps = torch.ones((self.trg_input.shape[0], seq_len), dtype=torch.bool)",
        "            self.frame_mask = valid_steps  # [batch, T]: True for real (non-padded) target frames",
        "            # decoder padding mask [batch, 1, T, T]: True where the key frame is real",
        "            self.trg_mask = valid_steps.unsqueeze(1).unsqueeze(2).expand(-1, 1, seq_len, -1).contiguous()",
        "            landmark_valid = getattr(torch_batch, \"landmark_valid\", None)",
        "            if landmark_valid is not None:",
        "                if self.future_prediction != 0:",
        "                    raise NotImplementedError(\"landmark validity masks are not supported with future_prediction\")",
        "                # validity of the frames being predicted (shifted by one like self.trg): [B, T, 75] -> [B, T, 225]",
        "                lv = landmark_valid[:, 1:, :]",
        "                self.coord_mask = lv.unsqueeze(-1).expand(-1, -1, -1, 3).reshape(lv.shape[0], lv.shape[1], -1)",
        "                assert self.coord_mask.shape[1] == self.trg.shape[1], \"validity mask and target length differ\"",
    ]
    s = _replace_block(s, "Target Pad is dynamic", "self.trg_mask = (F.pad(", new_block, "batch.py mask block")

    old = "        self.src_mask = self.src_mask.cuda()\n"
    _once(s, old, "_make_cuda src_mask line")
    s = s.replace(old, old + "        self.src_lengths = self.src_lengths.cuda()\n")

    old = "            self.trg_mask = self.trg_mask.cuda()\n"
    _once(s, old, "_make_cuda trg_mask line")
    return s.replace(old, old +
                     "            self.frame_mask = self.frame_mask.cuda()\n"
                     "            if self.coord_mask is not None:\n"
                     "                self.coord_mask = self.coord_mask.cuda()\n"
                     "            if self.frame_lengths is not None:\n"
                     "                self.frame_lengths = self.frame_lengths.cuda()\n")


# ------------------------------------------------------------------------------------ loss.py
NEW_LOSS_CODE = '''def build_element_mask(targets, frame_mask=None, coord_mask=None):
    """
    Boolean mask [batch, time, width] of the target elements that may contribute to the loss.
      frame_mask  [batch, time]         False for padded frames
      coord_mask  [batch, time, width]  False for invalid landmark coordinates. A mask that is one
                                        column narrower than the targets (225 vs 226) leaves the
                                        frame-counter column valid.
    """
    batch, steps, width = targets.shape
    mask = torch.ones((batch, steps, width), dtype=torch.bool, device=targets.device)
    if frame_mask is not None:
        if frame_mask.dim() != 2 or tuple(frame_mask.shape) != (batch, steps):
            raise ValueError("frame_mask must have shape [batch, time] = %s, got %s"
                             % ((batch, steps), tuple(frame_mask.shape)))
        mask = mask & frame_mask.to(device=targets.device, dtype=torch.bool).unsqueeze(-1)
    if coord_mask is not None:
        cm = coord_mask.to(device=targets.device, dtype=torch.bool)
        if cm.dim() != 3 or tuple(cm.shape[:2]) != (batch, steps):
            raise ValueError("coord_mask must have shape [batch, time, width] with batch/time = %s, got %s"
                             % ((batch, steps), tuple(cm.shape)))
        if cm.shape[2] == width - 1:
            cm = torch.cat([cm, torch.ones((batch, steps, 1), dtype=torch.bool, device=targets.device)], dim=2)
        elif cm.shape[2] != width:
            raise ValueError("coord_mask width %d does not match target width %d (or %d without the counter)"
                             % (cm.shape[2], width, width - 1))
        mask = mask & cm
    return mask


class RegLoss(nn.Module):
    """
    Regression loss. With `frame_mask` and/or `coord_mask` only valid elements contribute: padded frames
    and unreliable landmark coordinates are excluded, the mean is taken over the valid elements, and an
    all-invalid batch gives exactly 0 (never a division by zero). Without masks the original behaviour
    (zeros in the target are padding) is kept for regression testing.
    """

    def __init__(self, cfg, target_pad=0.0):
        super(RegLoss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss(reduction="none")
        elif self.loss == "mse":
            self.criterion = nn.MSELoss(reduction="none")
        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss(reduction="none")

        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

    # pylint: disable=arguments-differ
    def forward(self, preds, targets, frame_mask=None, coord_mask=None):

        if frame_mask is None and coord_mask is None:
            # legacy behaviour: every zero in the target is treated as padding
            loss_mask = (targets != self.target_pad).to(preds.dtype)
            loss = self.criterion(preds * loss_mask, targets * loss_mask).mean()
        else:
            if preds.shape != targets.shape:
                raise ValueError("predictions %s and targets %s differ in shape"
                                 % (tuple(preds.shape), tuple(targets.shape)))
            keep = build_element_mask(targets, frame_mask, coord_mask)
            n_valid = keep.sum().clamp_min(1)
            loss = self.criterion(preds, targets).masked_fill(~keep, 0.0).sum() / n_valid

        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss

'''


def patch_loss(s, ctx):
    start = s.index("class RegLoss(")
    end = s.index("class XentLoss(")
    assert start < end, "unexpected class order in loss.py"
    s = s[:start] + NEW_LOSS_CODE + s[end:]
    old = "from torch import nn, Tensor"
    assert old in s, "torch import line not found in loss.py"
    return s.replace(old, "import torch\n" + old, 1)


# ------------------------------------------------------------------------------------ model.py
def patch_model(s, ctx):
    old = "        batch_loss = loss_function(skel_out, batch.trg)\n"
    _once(s, old, "loss call in get_loss_for_batch")
    new = (f"        {MARK} pass the validity masks so padded frames / unreliable landmarks never enter the loss\n"
           "        frame_mask = getattr(batch, \"frame_mask\", None)\n"
           "        coord_mask = getattr(batch, \"coord_mask\", None)\n"
           "        if (frame_mask is not None or coord_mask is not None) and self.future_prediction == 0:\n"
           "            batch_loss = loss_function(skel_out, batch.trg, frame_mask=frame_mask, coord_mask=coord_mask)\n"
           "        else:\n"
           "            batch_loss = loss_function(skel_out, batch.trg)\n")
    return s.replace(old, new)


# ------------------------------------------------------------------------------------ search.py
def patch_search(s, ctx):
    new_block = [
        f"        {MARK} nothing is padded while generating, so every generated frame is visible",
        "        padding_mask = torch.ones(",
        "            (ys.size(0), 1, ys.size(1), ys.size(1)), dtype=torch.bool, device=ys.device)",
    ]
    return _replace_block(s, "# Cut padding mask to required size", "padding_mask = (F.pad(", new_block, "search.py")


PATCHES = [
    ("vocabulary.py", patch_vocabulary),
    ("helpers.py", patch_helpers),
    ("transformer_layers.py", patch_transformer_layers),
    ("batch.py", patch_batch),
    ("loss.py", patch_loss),
    ("model.py", patch_model),
    ("search.py", patch_search),
]


def apply_all(repo_dir, pe_max_len, backup_dir=None, log=print):
    """Back up and patch the repository files in place. Safe to call again (already patched files are skipped)."""
    repo_dir = Path(repo_dir)
    backup_dir = Path(backup_dir) if backup_dir else repo_dir / "_originals"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ctx = {"pe_max_len": int(pe_max_len)}
    messages = []
    for filename, fn in PATCHES:
        path = repo_dir / filename
        src = path.read_text(encoding="utf-8")
        if MARK in src:
            messages.append(f"{filename:24s} already patched")
            continue
        backup = backup_dir / filename
        if not backup.exists():
            shutil.copy2(path, backup)
        try:
            new = fn(src, ctx)
        except AssertionError as exc:
            raise AssertionError(f"Patch for {filename} did not apply: {exc}") from exc
        path.write_text(new.rstrip("\n") + f"\n\n{MARK} applied\n", encoding="utf-8")
        compile(path.read_text(encoding="utf-8"), filename, "exec")     # syntax check right away
        messages.append(f"{filename:24s} patched (original saved in {backup_dir.name}/)")
    for m in messages:
        log(m)
    return messages
