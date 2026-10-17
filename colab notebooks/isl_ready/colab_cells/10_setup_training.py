# TRAINING SETUP — datasets, vocabulary, configuration, loaders and model for a real run.
# Needs the state created by 06_restore_and_diagnose.py and 07_convert.py. Refuses to continue unless the latest
# pre-training report says READY (set FORCE_TRAIN = True to override on purpose).
import copy, importlib, json, shutil, sys, time
import torch
import yaml

_o = globals().get("ISL_OVERRIDES", {})


def _opt(name, default):
    """A setting can be given as a plain variable, or through ISL_OVERRIDES, or falls back to the default."""
    return globals().get(name, _o.get(name, default))


RUN_NAME = _opt("RUN_NAME", "main")
FORCE_TRAIN = _opt("FORCE_TRAIN", False)
ALLOW_CPU_TRAINING = _opt("ALLOW_CPU_TRAINING", False)


def reset_imports():
    """Drop cached copies of repository / project modules so this always sees the current files."""
    stems = {p.stem for d in (CODE_DIR, REPO_DIR) for p in d.glob("*.py") if not p.stem.startswith("__")}
    for s in stems - {"isign_pose", "isign_report", "repo_patches"}:
        sys.modules.pop(s, None)
    for d in (CODE_DIR, REPO_DIR):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    importlib.invalidate_caches()


reset_imports()
import isign_data as idata
import isign_train as it
from constants import PAD_TOKEN
from loss import RegLoss
from model import build_model
from vocabulary import Vocabulary

# ---------- the readiness decision ----------
report_path = DRIVE_OUT / "pre_training_report.json"
decision = json.loads(report_path.read_text(encoding="utf-8")).get("decision") if report_path.exists() else "no report yet"
print("latest pre-training decision:", decision)
if decision != "READY FOR CONTROLLED TRAINING":
    if not FORCE_TRAIN:
        raise RuntimeError(f"The latest pre-training report says '{decision}'. Fix that first (run the checks), or set FORCE_TRAIN = True "
                           "to train anyway.")
    print("WARNING: FORCE_TRAIN is on; training although the decision is not READY.")

# ---------- device ----------
DEVICE, DEVICE_INFO = it.select_device(require_cuda=REQUIRE_GPU_FOR_TRAINING and not ALLOW_CPU_TRAINING)
IS_CUDA = DEVICE.type == "cuda"
print(json.dumps(DEVICE_INFO))

# ---------- data and vocabulary (training split only) ----------
train_set = idata.ISignShardDataset(PROC_ROOT / "train", NORM, max_frames=QUALITY["max_frames"])
val_set = idata.ISignShardDataset(PROC_ROOT / "val", NORM, max_frames=QUALITY["max_frames"])
test_set = idata.ISignShardDataset(PROC_ROOT / "test", NORM, max_frames=QUALITY["max_frames"])   # opened, not evaluated

RUN_DIR = DRIVE_OUT / "runs" / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
vocab_path = RUN_DIR / "src_vocab.txt"
if vocab_path.exists():                      # a resumed run must keep exactly the word ids it started with
    src_vocab = Vocabulary(file=str(vocab_path))
    new_words = {t for toks in train_set.token_lists() for t in toks} - set(src_vocab.itos)
    print(f"using the saved vocabulary ({len(src_vocab)} entries); {len(new_words)} training words are not in it and map to <unk>")
else:
    src_vocab = idata.build_vocab(train_set.token_lists())
    src_vocab.to_file(str(vocab_path))
    print(f"built the vocabulary from the training split: {len(src_vocab)} entries")
pad_id = src_vocab.stoi[PAD_TOKEN]

# ---------- configuration ----------
cfg = yaml.safe_load((REPO_DIR / "Configs" / "Base.yaml").read_text())
cfg["data"].update({"src": "text", "train": str(PROC_ROOT / "train"), "dev": str(PROC_ROOT / "val"), "test": str(PROC_ROOT / "test")})
cfg["data"].pop("src_vocab", None)
cfg["model"]["trg_size"] = ip.N_COORDS
cfg["training"].update({
    "use_cuda": bool(IS_CUDA), "batch_size": BATCH_SIZE, "epochs": MAX_EPOCHS, "learning_rate": LEARNING_RATE,
    "early_stop_patience": EARLY_STOP_PATIENCE, "val_every_epochs": 1, "continue": True, "overwrite": False,
    "model_dir": str(RUN_DIR), "random_seed": SEED,
})
cfg["normalization"] = NORM
cfg_path = RUN_DIR / "config.yaml"
if cfg_path.exists():
    old_cfg = yaml.safe_load(cfg_path.read_text())
    if old_cfg.get("model") != cfg["model"] or old_cfg.get("normalization") != cfg["normalization"]:
        raise RuntimeError(f"{cfg_path} was made with a different model or normalisation; use another RUN_NAME for a new experiment.")
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
(RUN_DIR / "normalization.json").write_text(json.dumps(NORM, indent=1, sort_keys=True), encoding="utf-8")

# ---------- loaders and model ----------
train_loader = idata.make_loader(train_set, src_vocab, BATCH_SIZE, shuffle=True, seed=SEED, num_workers=NUM_WORKERS)
val_loader = idata.make_loader(val_set, src_vocab, BATCH_SIZE, num_workers=NUM_WORKERS)
test_loader = idata.make_loader(test_set, src_vocab, BATCH_SIZE, num_workers=0)

it.set_seed(SEED)
model = build_model(cfg, src_vocab=src_vocab, trg_vocab=Vocabulary(tokens=[]))
it.prepare_model(model, DEVICE)
criterion = RegLoss(cfg)

n_params = sum(p.numel() for p in model.parameters())
train_frames = int(train_set.store.lengths.sum())
print(f"\nrun '{RUN_NAME}' in {RUN_DIR}")
print(f"train/val/test sequences: {len(train_set)}/{len(val_set)}/{len(test_set)} | train frames {train_frames:,} | "
      f"{len(train_loader)} steps per epoch at batch size {BATCH_SIZE}")
print(f"model: {n_params / 1e6:.1f} M parameters on {DEVICE} | source vocabulary {len(src_vocab)} | loss {cfg['training']['loss']}")
