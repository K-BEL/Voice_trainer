import argparse  # noqa: D100
import copy
import csv
import gc
import glob
import inspect
import math
import os
from pathlib import Path
import random
import shutil
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import data as data_utils
from utils import get_config
from utils.data import DynBatchDataset
from utils.training import save_states_gan as save_states
from vocoder import load_hifigan
from vocoder.hifigan.denoiser import Denoiser

from models.common.loss import calc_feature_match_loss, extract_chunks
try:
	from models.common.loss import PatchDiscriminatorCond as PatchDiscriminatorClass
	critic_uses_conditioning = True
except ImportError:
	from models.common.loss import PatchDiscriminator as PatchDiscriminatorClass
	critic_uses_conditioning = False
from models.fastpitch import net_config
from models.fastpitch.fastpitch.attn_loss_function import AttentionBinarizationLoss
from models.fastpitch.fastpitch.data_function import TTSCollate, batch_to_gpu
from models.fastpitch.fastpitch.loss_function import FastPitchLoss
from models.fastpitch.fastpitch.model import FastPitch

try:
	from text import tokenizer_raw
except ImportError:
	tokenizer_raw = None


def build_critic(device):  # noqa: ANN001, ANN201, D103
	# Different upstream snapshots expose different discriminator signatures.
	if critic_uses_conditioning:
		candidates = ((2, 32), (32,), ())  # noqa: C408
	else:
		# Non-conditional PatchDiscriminator usually expects 1 input channel.
		candidates = ((1, 32), (1,), (32,), ())  # noqa: C408
	for args in candidates:
		try:
			return PatchDiscriminatorClass(*args).to(device)
		except TypeError:
			continue
	return PatchDiscriminatorClass().to(device)


def critic_forward(critic, chunks, cond_vecs):  # noqa: ANN001, ANN201, D103
	if critic_uses_conditioning:
		return critic(chunks, cond_vecs)
	try:
		return critic(chunks, cond_vecs)
	except TypeError:
		return critic(chunks)


def _sanitize_tensor(t):  # noqa: ANN001, ANN201, D103
	if torch.is_tensor(t) and t.is_floating_point():
		return torch.nan_to_num(t, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
	return t


def _is_finite_tensor(t):  # noqa: ANN001, ANN201, D103
	return bool(torch.isfinite(t).all().item())


def _sanitize_gradients(parameters):  # noqa: ANN001, ANN201, D103
	changed = 0
	for param in parameters:
		if param.grad is None or not torch.is_tensor(param.grad):
			continue
		if not torch.isfinite(param.grad).all():
			param.grad = torch.nan_to_num(
				param.grad,
				nan=0.0,
				posinf=0.0,
				neginf=0.0,
			)
			changed += 1
	return changed


def _prepare_tmp_dir(root_dir):  # noqa: ANN001, ANN201, D103
	"""Use workspace temp dir when system /tmp is too small."""
	system_tmp = Path("/tmp")
	workspace_tmp = Path(root_dir) / ".tmp"
	try:
		system_free = shutil.disk_usage(system_tmp).free
	except Exception:  # noqa: BLE001
		system_free = 0
	if system_free < 512 * 1024 * 1024:
		workspace_tmp.mkdir(parents=True, exist_ok=True)
		os.environ["TMPDIR"] = workspace_tmp.as_posix()
		print(f"Using TMPDIR={workspace_tmp} (system /tmp is constrained)")  # noqa: T201


def _resolve_num_workers(default_workers):  # noqa: ANN001, ANN201, D103
	override = os.environ.get("VT_NUM_WORKERS")
	if override is not None:
		return max(0, int(override))
	# Safer default for constrained cloud containers.
	return max(0, min(int(default_workers), 2))


def remove_silence_safe(energy_per_frame: torch.Tensor, thresh: float = -10.0):  # noqa: D103
	# Some torchaudio/torchcodec combinations return shapes that make the
	# legacy remove_silence implementation ambiguous (non-scalar bool tensors).
	if energy_per_frame.ndim > 1:
		for _ in range(energy_per_frame.ndim - 1):
			energy_per_frame = energy_per_frame.mean(0)

	keep = (energy_per_frame > thresh).reshape(-1)
	i = keep.numel() - 1
	while i > 0 and not bool(keep[i].item()):
		keep[i] = True
		i -= 1
	return keep


_orig_torchaudio_load = data_utils.torchaudio.load


def torchaudio_load_mono_safe(filepath, *args, **kwargs):  # noqa: ANN001, ANN201, D103
	wave, sr = _orig_torchaudio_load(filepath, *args, **kwargs)
	if wave.ndim == 2 and wave.size(0) > 1:
		# Legacy pipeline expects mono waveform [1, T].
		wave = wave.mean(dim=0, keepdim=True)
	return wave, sr


# Patch legacy utility function for compatibility with newer stacks.
data_utils.remove_silence = remove_silence_safe
data_utils.torchaudio.load = torchaudio_load_mono_safe


def build_train_dataset(config):  # noqa: ANN001, ANN201, D103
	def _convert_csv_labels_to_pipe(csv_path):  # noqa: ANN001, ANN201, D103
		csv_file = Path(csv_path)
		pipe_file = csv_file.with_suffix(".pipe.txt")
		pipe3_mid_file = csv_file.with_suffix(".pipe3mid.txt")
		pipe3_file = csv_file.with_suffix(".pipe3.txt")
		comma_file = csv_file.with_suffix(".comma.txt")
		comma3_mid_file = csv_file.with_suffix(".comma3mid.txt")
		comma3_file = csv_file.with_suffix(".comma3.txt")
		if (
			pipe_file.exists()
			and pipe3_mid_file.exists()
			and pipe3_file.exists()
			and comma_file.exists()
			and comma3_mid_file.exists()
			and comma3_file.exists()
		):
			return {
				"pipe2": pipe_file.as_posix(),
				"pipe3mid": pipe3_mid_file.as_posix(),
				"pipe3tail": pipe3_file.as_posix(),
				"comma2": comma_file.as_posix(),
				"comma3mid": comma3_mid_file.as_posix(),
				"comma3tail": comma3_file.as_posix(),
			}

		with csv_file.open("r", encoding="utf-8", newline="") as src, pipe_file.open(
			"w", encoding="utf-8", newline=""
		) as dst, pipe3_mid_file.open(
			"w", encoding="utf-8", newline=""
		) as dst3_mid, pipe3_file.open(
			"w", encoding="utf-8", newline=""
		) as dst3_tail, comma_file.open(
			"w", encoding="utf-8", newline=""
		) as dst_comma, comma3_mid_file.open(
			"w", encoding="utf-8", newline=""
		) as dst_comma3_mid, comma3_file.open(
			"w", encoding="utf-8", newline=""
		) as dst_comma3_tail:
			reader = csv.DictReader(src)
			for row in reader:
				audio = (row.get("audio") or "").strip()
				caption = (row.get("caption") or "").replace("\n", " ").strip()
				if audio:
					dst.write(f"{audio}|{caption}\n")
					dst3_mid.write(f"{audio}|0|{caption}\n")
					dst3_tail.write(f"{audio}|{caption}|0\n")
					dst_comma.write(f"{audio},{caption}\n")
					dst_comma3_mid.write(f"{audio},0,{caption}\n")
					dst_comma3_tail.write(f"{audio},{caption},0\n")
		print(f"Built pipe metadata at {pipe_file}")  # noqa: T201
		print(f"Built pipe3-mid metadata at {pipe3_mid_file}")  # noqa: T201
		print(f"Built pipe3 metadata at {pipe3_file}")  # noqa: T201
		return {
			"pipe2": pipe_file.as_posix(),
			"pipe3mid": pipe3_mid_file.as_posix(),
			"pipe3tail": pipe3_file.as_posix(),
			"comma2": comma_file.as_posix(),
			"comma3mid": comma3_mid_file.as_posix(),
			"comma3tail": comma3_file.as_posix(),
		}

	def _build_pitch_dict_file(f0_folder_path, f0_dict_path):  # noqa: ANN001, ANN201, D103
		def _normalize_pitch_tensor(pitch_tensor):  # noqa: ANN001, ANN201, D103
			pitch_tensor = torch.as_tensor(pitch_tensor, device="cpu").float().reshape(-1)
			return pitch_tensor

		def _pitch_dict_needs_rebuild(f0_dict_file):  # noqa: ANN001, ANN201, D103
			if not f0_dict_file.exists():
				return True
			try:
				existing = torch.load(f0_dict_file, map_location="cpu")
				if not isinstance(existing, dict) or not existing:
					return True
				first_value = next(iter(existing.values()))
				first_tensor = torch.as_tensor(first_value, device="cpu")
				# Upstream loader does self.f0_dict[wav_name][None] and expects [1, T].
				return first_tensor.ndim != 1
			except Exception:  # noqa: BLE001
				return True

		f0_folder = Path(f0_folder_path)
		f0_dict_file = Path(f0_dict_path)
		if not f0_folder.exists():
			return
		if not _pitch_dict_needs_rebuild(f0_dict_file):
			return

		f0_dict_file.parent.mkdir(parents=True, exist_ok=True)
		pitch_dict = {}
		for pitch_file in f0_folder.rglob("*.pth"):
			rel = pitch_file.relative_to(f0_folder).as_posix()
			audio_rel = rel[:-4] if rel.endswith(".pth") else rel
			basename = Path(audio_rel).name
			stem = Path(audio_rel).stem
			pitch_tensor = _normalize_pitch_tensor(
				torch.load(pitch_file, map_location="cpu")
			)

			# Support common lookup styles used by different dataset versions.
			for key in (
				audio_rel,
				f"./{audio_rel}",
				basename,
				stem,
				audio_rel.replace("/", os.sep),
				basename.replace("/", os.sep),
			):
				pitch_dict[key] = pitch_tensor

		torch.save(pitch_dict, f0_dict_file)
		print(f"Built missing pitch dict at {f0_dict_file}")  # noqa: T201

	dataset_kwargs = {
		"txtpath": config.train_labels,
		"wavpath": config.train_wavs_path,
		# Older tts-arabic-pytorch datasets require a named text field among:
		# arabic / phonemes / buckwalter. "raw" is not accepted there.
		"label_pattern": r"(?P<filename>[^,]*),(?P<arabic>.*)",
		"f0_folder_path": config.f0_folder_path,
		"f0_mean": config.f0_mean,
		"f0_std": config.f0_std,
		"max_lengths": config.max_lengths,
		"batch_sizes": config.batch_sizes,
	}
	supported_params = set(inspect.signature(DynBatchDataset.__init__).parameters)
	if "f0_dict_path" in supported_params:
		fallback_f0_dict_path = Path("./data/pitch_dict.pt").resolve().as_posix()
		dataset_kwargs["f0_dict_path"] = fallback_f0_dict_path
		_build_pitch_dict_file(
			dataset_kwargs["f0_folder_path"],
			dataset_kwargs["f0_dict_path"],
		)
	filtered_kwargs = {
		key: value for key, value in dataset_kwargs.items() if key in supported_params
	}

	def _dataset_len_safe(ds):  # noqa: ANN001, ANN201, D103
		try:
			return len(ds)
		except Exception:  # noqa: BLE001
			return -1

	def _try_build(kwargs):  # noqa: ANN001, ANN201, D103
		try:
			ds = DynBatchDataset(**kwargs)
			return ds, _dataset_len_safe(ds)
		except FileNotFoundError as err:
			print(f"Dataset build skipped (missing file): {err}")  # noqa: T201
			return None, -1
		except Exception as err:  # noqa: BLE001
			print(f"Dataset build failed for current format: {err}")  # noqa: T201
			return None, -1

	dataset, dataset_len = _try_build(filtered_kwargs)
	if dataset_len > 0:
		return dataset

	txtpath = Path(dataset_kwargs["txtpath"])
	candidate_configs = [
		(
			filtered_kwargs.get("txtpath", dataset_kwargs["txtpath"]),
			config.label_pattern,
		),
		(
			filtered_kwargs.get("txtpath", dataset_kwargs["txtpath"]),
			dataset_kwargs["label_pattern"],
		),
	]
	if txtpath.suffix.lower() == ".csv" and txtpath.exists():
		converted_paths = _convert_csv_labels_to_pipe(txtpath)
		candidate_configs.extend([
			(converted_paths["pipe2"], r"(?P<filename>[^|]*)\|(?P<arabic>.*)"),
			(converted_paths["pipe3mid"], r"(?P<filename>[^|]*)\|(?P<speaker>\d+)\|(?P<arabic>.*)"),
			(converted_paths["pipe3tail"], r"(?P<filename>[^|]*)\|(?P<arabic>[^|]*)\|(?P<speaker>\d+)"),
			(converted_paths["comma2"], r"(?P<filename>[^,]*),(?P<arabic>.*)"),
			(converted_paths["comma3mid"], r"(?P<filename>[^,]*),(?P<speaker>\d+),(?P<arabic>.*)"),
			(converted_paths["comma3tail"], r"(?P<filename>[^,]*),(?P<arabic>[^,]*),(?P<speaker>\d+)"),
		])

	seen = set()
	for candidate_txtpath, pattern in candidate_configs:
		if "txtpath" in supported_params:
			filtered_kwargs["txtpath"] = candidate_txtpath
		key = (filtered_kwargs.get("txtpath"), pattern)
		if key in seen:
			continue
		seen.add(key)
		if "label_pattern" in supported_params:
			filtered_kwargs["label_pattern"] = pattern
		print(  # noqa: T201
			"Dataset empty; retrying with "
			f"txtpath={filtered_kwargs.get('txtpath')} "
			f"label_pattern={pattern}"
		)
		dataset, dataset_len = _try_build(filtered_kwargs)
		if dataset_len > 0:
			return dataset

	return dataset


device = "cuda:0"
torch.cuda.set_device(device)

try:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--config",
		type=str,
		default="configs/nawar_fp_adv_raw.yaml",
		help="Path to yaml config file",
	)
	args = parser.parse_args()
	config_path = args.config
except:  # noqa: E722
	config_path = "./configs/nawar_fp_adv_raw.yaml"


config = get_config(config_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_prepare_tmp_dir(Path(__file__).resolve().parents[3].as_posix())

# make checkpoint folder if nonexistent
if not os.path.isdir(config.checkpoint_dir):  # noqa: PTH112
	os.makedirs(os.path.abspath(config.checkpoint_dir))  # noqa: PTH100, PTH103
	print(f"Created checkpoint folder @ {config.checkpoint_dir}")  # noqa: T201


train_dataset = build_train_dataset(config)


collate_fn = TTSCollate()

config.batch_size = 1
sampler, shuffle, drop_last = None, True, True
train_loader = DataLoader(
	train_dataset,
	batch_size=config.batch_size,
	collate_fn=lambda x: collate_fn(x[0]),
	shuffle=shuffle,
	drop_last=drop_last,
	sampler=sampler,
	num_workers=_resolve_num_workers(config.num_workers),
	pin_memory=True,
)

(
	text_padded,
	input_lengths,
	mel_padded,
	output_lengths,
	len_x,
	pitch_padded,
	energy_padded,
	speaker,
	attn_prior_padded,
	audiopaths,
) = next(iter(train_loader))


net_config["n_speakers"] = 1600

model = FastPitch(**net_config).to(device)

optimizer = torch.optim.AdamW(
	model.parameters(),
	lr=config.g_lr,
	betas=(config.g_beta1, config.g_beta2),
	weight_decay=config.weight_decay,
)

try:
	criterion = FastPitchLoss(dur_loss_toofast_scale=1.0)
except TypeError:
	criterion = FastPitchLoss()
attention_kl_loss = AttentionBinarizationLoss()


critic = build_critic(device)

optimizer_d = torch.optim.AdamW(
	critic.parameters(),
	lr=config.d_lr,
	betas=(config.d_beta1, config.d_beta2),
	weight_decay=config.weight_decay,
)
chunk_len = 128
resume_optimizers = os.environ.get("VT_RESUME_OPTIMIZERS", "0") == "1"
resume_progress = os.environ.get("VT_RESUME_PROGRESS", "0") == "1"
gan_warmup_iters = int(os.environ.get("VT_GAN_WARMUP_ITERS", "1000"))
keep_ckpts = int(os.environ.get("VT_KEEP_CKPTS", "3"))

# resume from existing checkpoint
n_epoch, n_iter = 0, 0

if config.restore_model != "":
	state_dicts = torch.load(config.restore_model, map_location=device)
	model.load_state_dict(state_dicts["model"])
	if "model_d" in state_dicts:
		try:
			critic.load_state_dict(state_dicts["model_d"], strict=False)
		except RuntimeError as err:
			print(f"Skipping incompatible discriminator model state: {err}")  # noqa: T201
	if resume_optimizers and "optim" in state_dicts:
		try:
			optimizer.load_state_dict(state_dicts["optim"])
		except ValueError as err:
			print(f"Skipping incompatible generator optimizer state: {err}")  # noqa: T201
	if resume_optimizers and "optim_d" in state_dicts:
		try:
			optimizer_d.load_state_dict(state_dicts["optim_d"])
		except ValueError as err:
			print(f"Skipping incompatible discriminator optimizer state: {err}")  # noqa: T201
	if resume_progress and "epoch" in state_dicts:
		n_epoch = state_dicts["epoch"]
	if resume_progress and "iter" in state_dicts:
		n_iter = state_dicts["iter"]
else:
	model_sd = torch.load("G:/models/fastpitch/nvidia_fastpitch_210824+cfg.pt")
	model.load_state_dict(
		{k.removeprefix("module."): v for k, v in model_sd["state_dict"].items()},
	)

tb_logging_enabled = os.environ.get("VT_DISABLE_TB", "0") != "1"
writer = SummaryWriter(config.log_dir) if tb_logging_enabled else None


def safe_save_states(filename, model, critic, optimizer, optimizer_d, n_iter, epoch):  # noqa: ANN001, ANN201, D103
	try:
		save_states(
			filename,
			model,
			critic,
			optimizer,
			optimizer_d,
			n_iter,
			epoch,
			net_config,
			config,
		)
		return True
	except (RuntimeError, OSError) as err:
		print(f"Checkpoint save failed for {filename}: {err}")  # noqa: T201
		# Best-effort retry for transient I/O hiccups.
		time.sleep(1.0)
		gc.collect()
		torch.cuda.empty_cache()
		try:
			save_states(
				filename,
				model,
				critic,
				optimizer,
				optimizer_d,
				n_iter,
				epoch,
				net_config,
				config,
			)
			print(f"Checkpoint save retry succeeded for {filename}")  # noqa: T201
			return True
		except (RuntimeError, OSError) as retry_err:
			print(f"Checkpoint retry failed for {filename}: {retry_err}")  # noqa: T201
			return False


def cleanup_old_checkpoints(checkpoint_dir, keep=3):  # noqa: ANN001, ANN201, D103
	"""Delete old numbered checkpoints, keeping only the most recent `keep`.

	The rolling `states.pth` is never deleted. Only `states_XXXXX.pth` files
	are considered. Set `keep=0` to disable cleanup entirely.
	"""
	if keep <= 0:
		return
	pattern = os.path.join(checkpoint_dir, "states_*.pth")  # noqa: PTH118
	files = sorted(glob.glob(pattern), key=os.path.getmtime)
	to_delete = files[:-keep] if len(files) > keep else []
	for f in to_delete:
		try:
			os.remove(f)  # noqa: PTH107
			print(f"Cleaned up old checkpoint: {os.path.basename(f)}")  # noqa: T201, PTH119
		except OSError as err:
			print(f"Failed to delete {f}: {err}")  # noqa: T201


def warmup_cosine_lambda(warmup_iters, total_iters, min_lr_ratio=0.01):  # noqa: ANN001, ANN201, D103
	"""Return an lr_lambda function for LambdaLR: linear warmup then cosine decay."""

	def _lr_lambda(current_iter):  # noqa: ANN001, ANN201
		if current_iter < warmup_iters:
			return max(1e-6, current_iter / max(1, warmup_iters))
		progress = (current_iter - warmup_iters) / max(1, total_iters - warmup_iters)
		return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

	return _lr_lambda


def _split_labels_file(labels_path, val_ratio=0.1, seed=42):  # noqa: ANN001, ANN201, D103
	"""Split a labels file into train and val portions. Returns (train_path, val_path)."""
	labels = Path(labels_path)
	train_path = labels.parent / (labels.stem + "_train" + labels.suffix)
	val_path = labels.parent / (labels.stem + "_val" + labels.suffix)

	if train_path.exists() and val_path.exists():
		return train_path.as_posix(), val_path.as_posix()

	with labels.open("r", encoding="utf-8") as f:
		lines = f.readlines()

	# Handle CSV with header
	header = None
	if labels.suffix.lower() == ".csv":
		header = lines[0]
		lines = lines[1:]

	rng = random.Random(seed)  # noqa: S311
	rng.shuffle(lines)

	split_idx = max(1, int(len(lines) * (1 - val_ratio)))
	train_lines = lines[:split_idx]
	val_lines = lines[split_idx:]

	with train_path.open("w", encoding="utf-8") as f:
		if header:
			f.write(header)
		f.writelines(train_lines)

	with val_path.open("w", encoding="utf-8") as f:
		if header:
			f.write(header)
		f.writelines(val_lines)

	print(f"Split dataset: {len(train_lines)} train, {len(val_lines)} val")  # noqa: T201
	return train_path.as_posix(), val_path.as_posix()


def build_val_dataset(config, val_labels_path):  # noqa: ANN001, ANN201, D103
	"""Build a validation dataset using a split labels file."""
	val_config = copy.deepcopy(config)
	val_config.train_labels = val_labels_path
	try:
		return build_train_dataset(val_config)
	except Exception as err:  # noqa: BLE001
		print(f"Could not build validation dataset: {err}")  # noqa: T201
		return None


# ─── AMP (Mixed Precision) ────────────────────────────────────────
use_amp = os.environ.get("VT_AMP", "1") == "1"
scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)
if use_amp:
	print("Mixed precision (AMP) enabled")  # noqa: T201

# ─── LR Schedulers (warmup + cosine decay) ────────────────────────
lr_warmup_iters = int(os.environ.get("VT_WARMUP_ITERS", "1000"))
lr_min_ratio = float(os.environ.get("VT_LR_MIN_RATIO", "0.01"))
# Estimate total iterations for cosine schedule
_est_iters_per_epoch = max(1, len(train_loader))
total_iters_est = _est_iters_per_epoch * config.epochs
lr_lambda_fn = warmup_cosine_lambda(lr_warmup_iters, total_iters_est, lr_min_ratio)
scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fn)
scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lr_lambda_fn)
# Fast-forward schedulers if resuming
for _ in range(n_iter):
	scheduler_g.step()
	scheduler_d.step()

# ─── Validation dataset ───────────────────────────────────────────
val_split = float(os.environ.get("VT_VAL_SPLIT", "0.1"))
val_dataset = None
val_loader = None
if val_split > 0:
	try:
		_, val_labels_path = _split_labels_file(config.train_labels, val_ratio=val_split)
		val_dataset = build_val_dataset(config, val_labels_path)
		if val_dataset is not None and len(val_dataset) > 0:
			val_loader = DataLoader(
				val_dataset,
				batch_size=1,
				collate_fn=lambda x: collate_fn(x[0]),
				shuffle=False,
				drop_last=False,
				num_workers=0,
				pin_memory=True,
			)
			print(f"Validation dataset: {len(val_dataset)} samples")  # noqa: T201
		else:
			print("Validation dataset empty, disabling validation")  # noqa: T201
	except Exception as err:  # noqa: BLE001
		print(f"Skipping validation: {err}")  # noqa: T201

# ─── Early stopping ───────────────────────────────────────────────
patience = int(os.environ.get("VT_PATIENCE", "10"))
best_val_loss = float("inf")
patience_counter = 0

# ─── Training banner ──────────────────────────────────────────────
print("=" * 60)  # noqa: T201
print(f"  Training: epochs={config.epochs}, iter={n_iter}, device={device}")  # noqa: T201
print(f"  AMP={use_amp}, LR warmup={lr_warmup_iters}, patience={patience}")  # noqa: T201
print(f"  Keep ckpts={keep_ckpts}, GAN warmup={gan_warmup_iters}")  # noqa: T201
if val_loader is not None:
	print(f"  Validation: every {config.n_save_states_iter} iters, split={val_split}")  # noqa: T201
print("=" * 60)  # noqa: T201

epoch_start_time = time.time()
model.train()

early_stop = False
for epoch in range(n_epoch, config.epochs):
	if early_stop:
		break
	train_dataset.shuffle()
	epoch_start_time = time.time()
	epoch_losses = []
	for batch in train_loader:
		x, y, _ = batch_to_gpu(batch)
		x = [_sanitize_tensor(v) for v in x]
		y = tuple(_sanitize_tensor(v) for v in y)
		if x[6] is None:
			# Legacy dataset variants may not provide speaker IDs.
			x[6] = torch.zeros(x[0].size(0), dtype=torch.long, device=x[0].device)
		x = tuple(x)

		# ── Forward pass (AMP) ──────────────────────────────────
		with torch.amp.autocast("cuda", enabled=use_amp):
			y_pred = model(x)

		mel_out, *_, attn_soft, attn_hard, _, _ = y_pred

		(
			text_padded,
			input_lengths,
			mel_padded,
			output_lengths,
			pitch_padded,
			energy_padded,
			speaker,
			attn_prior,
			audiopaths,
		) = x

		# extract chunks for critic
		Nchunks = mel_out.size(0)
		tar_len_ = min(output_lengths.min().item(), chunk_len)
		mel_ids = torch.randint(0, mel_out.size(0), (Nchunks,)).cuda(non_blocking=True)
		ofx_perc = torch.rand(Nchunks).cuda(non_blocking=True)
		out_lens = output_lengths[mel_ids]

		ofx = (
			(ofx_perc * (out_lens + tar_len_) - tar_len_ / 2)
			.clamp(out_lens * 0, out_lens - tar_len_)
			.long()
		)

		chunks_org = extract_chunks(
			mel_padded,
			ofx,
			mel_ids,
			tar_len_,
		)  # mel_padded: B F T
		chunks_gen = extract_chunks(
			mel_out.transpose(1, 2),
			ofx,
			mel_ids,
			tar_len_,
		)  # mel_out: B T F

		chunks_org_ = (chunks_org.unsqueeze(1) + 4.5) / 2.5
		chunks_gen_ = (chunks_gen.unsqueeze(1) + 4.5) / 2.5

		with torch.no_grad():
			speakers_input = speaker[mel_ids]
			speaker_vecs = model.speaker_emb.weight[speakers_input]
			speaker_vecs = torch.nn.functional.normalize(speaker_vecs, p=2, dim=1)
			cond_vecs = speaker_vecs

		# ── Discriminator step (AMP) ────────────────────────────
		with torch.amp.autocast("cuda", enabled=use_amp):
			d_org, fmaps_org = critic_forward(
				critic, chunks_org_.requires_grad_(True), cond_vecs  # noqa: FBT003
			)
			d_gen, _ = critic_forward(critic, chunks_gen_.detach(), cond_vecs)
			loss_d = 0.5 * (d_org - 1).square().mean() + 0.5 * d_gen.square().mean()

		if not _is_finite_tensor(loss_d):
			print("Skipping batch due to non-finite discriminator loss")  # noqa: T201
			optimizer_d.zero_grad(set_to_none=True)
			optimizer.zero_grad(set_to_none=True)
			continue

		critic.zero_grad()
		scaler_d.scale(loss_d).backward()
		scaler_d.step(optimizer_d)
		scaler_d.update()

		# ── Generator step (AMP) ────────────────────────────────
		with torch.amp.autocast("cuda", enabled=use_amp):
			loss, meta = criterion(y_pred, y)
			d_gen2, fmaps_gen = critic_forward(critic, chunks_gen_, cond_vecs)
			loss_score = (d_gen2 - 1).square().mean()
			loss_fmatch = calc_feature_match_loss(fmaps_gen, fmaps_org)

		gan_weight = config.gan_loss_weight if n_iter >= gan_warmup_iters else 0.0
		feat_weight = config.feat_loss_weight if n_iter >= gan_warmup_iters else 0.0
		loss += gan_weight * loss_score
		loss += feat_weight * loss_fmatch

		binarization_loss = attention_kl_loss(attn_hard, attn_soft)
		if not _is_finite_tensor(binarization_loss):
			binarization_loss = torch.zeros_like(loss)
		loss += 1.0 * binarization_loss
		if not _is_finite_tensor(loss):
			print("Skipping batch due to non-finite generator loss")  # noqa: T201
			optimizer.zero_grad(set_to_none=True)
			continue

		optimizer.zero_grad()
		scaler_g.scale(loss).backward()
		scaler_g.unscale_(optimizer)
		grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
		if _is_finite_tensor(grad_norm):
			scaler_g.step(optimizer)
		else:
			print("Skipping optimizer step due to non-finite gradient norm")  # noqa: T201
			optimizer.zero_grad(set_to_none=True)
			scaler_g.update()
			continue
		scaler_g.update()

		# ── LR scheduler step ───────────────────────────────────
		scheduler_g.step()
		scheduler_d.step()

		# ── LOGGING ─────────────────────────────────────────────
		meta["loss_d"] = loss_d.detach()
		meta["score"] = loss_score.detach()
		meta["fmatch"] = loss_fmatch.detach()
		meta["kl_loss"] = binarization_loss.detach()
		epoch_losses.append(meta["loss"].item())

		current_lr = optimizer.param_groups[0]["lr"]
		print(  # noqa: T201
			f"[E{epoch}/{config.epochs} I{n_iter}] "
			f"loss: {meta['loss'].item():.4f} gnorm: {grad_norm:.2f} "
			f"lr: {current_lr:.2e}"
		)

		if writer is not None:
			try:
				for k, v in meta.items():
					writer.add_scalar(f"train/{k}", v.item(), n_iter)
				writer.add_scalar("train/lr", current_lr, n_iter)
				writer.add_scalar("train/amp_scale", scaler_g.get_scale(), n_iter)
			except OSError as err:
				print(f"Disabling TensorBoard logging due to write error: {err}")  # noqa: T201
				try:
					writer.close()
				except Exception:  # noqa: BLE001
					pass
				writer = None

		# ── Checkpoint saving ───────────────────────────────────
		if n_iter % config.n_save_states_iter == 0:
			safe_save_states(
				"states.pth",
				model,
				critic,
				optimizer,
				optimizer_d,
				n_iter,
				epoch,
			)

			# ── Validation ──────────────────────────────────────
			if val_loader is not None:
				model.eval()
				val_losses = []
				with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
					for val_batch in val_loader:
						try:
							vx, vy, _ = batch_to_gpu(val_batch)
							vx = [_sanitize_tensor(v) for v in vx]
							vy = tuple(_sanitize_tensor(v) for v in vy)
							if vx[6] is None:
								vx[6] = torch.zeros(vx[0].size(0), dtype=torch.long, device=vx[0].device)
							vx = tuple(vx)
							vy_pred = model(vx)
							vloss, _ = criterion(vy_pred, vy)
							if _is_finite_tensor(vloss):
								val_losses.append(vloss.item())
						except Exception:  # noqa: BLE001
							continue
				if val_losses:
					avg_val_loss = sum(val_losses) / len(val_losses)
					print(f"  ── val_loss: {avg_val_loss:.4f} (best: {best_val_loss:.4f})")  # noqa: T201
					if writer is not None:
						try:
							writer.add_scalar("val/loss", avg_val_loss, n_iter)
						except OSError:
							pass
					# Early stopping check
					if avg_val_loss < best_val_loss:
						best_val_loss = avg_val_loss
						patience_counter = 0
						safe_save_states(
							"best_model.pth",
							model,
							critic,
							optimizer,
							optimizer_d,
							n_iter,
							epoch,
						)
						print("  ── Saved best_model.pth ✓")  # noqa: T201
					else:
						patience_counter += 1
						if patience > 0 and patience_counter >= patience:
							print(f"  ── Early stopping triggered (patience={patience})")  # noqa: T201
							early_stop = True
							break
				model.train()

		if n_iter % config.n_save_backup_iter == 0 and n_iter > 0:
			safe_save_states(
				f"states_{n_iter}.pth",
				model,
				critic,
				optimizer,
				optimizer_d,
				n_iter,
				epoch,
			)
			cleanup_old_checkpoints(config.checkpoint_dir, keep=keep_ckpts)

		n_iter += 1

	# ── Epoch summary ───────────────────────────────────────────
	epoch_time = time.time() - epoch_start_time
	avg_epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
	print(  # noqa: T201
		f"── Epoch {epoch} done in {epoch_time:.0f}s | "
		f"avg_loss: {avg_epoch_loss:.4f} | "
		f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
		f"patience: {patience_counter}/{patience}"
	)


safe_save_states(
	"states.pth",
	model,
	critic,
	optimizer,
	optimizer_d,
	n_iter,
	epoch,
)

if writer is not None:
	try:
		writer.close()
	except Exception:  # noqa: BLE001
		pass


idx = 0
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.imshow(
	y_pred[0][idx, : y[2][idx], :].detach().cpu().t(),
	aspect="auto",
	origin="lower",
)
ax2.imshow(y[0][idx, :, : y[2][idx]].detach().cpu(), aspect="auto", origin="lower")


vocoder = load_hifigan(config.vocoder_state_path, config.vocoder_config_path)
vocoder = vocoder.cuda()
denoiser = Denoiser(vocoder)

model.eval()
with torch.inference_mode():
	(mel_out, dec_lens, dur_pred, pitch_pred, energy_pred) = model.infer(x[0][0:1])

	wave = vocoder(mel_out[0])

plt.imshow(mel_out[0].cpu(), aspect="auto", origin="lower")

plt.plot(wave[0].cpu())


if tokenizer_raw is not None:
	phrase = "أَتَاحَتْ لِلبَائِعِ المُتَجَوِّلِ أنْ يَكُونَ جَاذِباً لِلمُوَاطِنِ الأقَلِّ دَخْلاً"

	token_ids = x[0][idx : idx + 1]
	token_ids = torch.LongTensor(tokenizer_raw(" " + phrase + ". "))[None].cuda()

	with torch.inference_mode():
		(mel_out, dec_lens, dur_pred, pitch_pred, energy_pred) = model.infer(
			token_ids,
			pace=1,
			speaker=1,
		)

		wave = vocoder(mel_out[0])
		wave_ = denoiser(wave, 0.003)
		wave_ /= wave_.abs().max()
else:
	print("tokenizer_raw not found; skipping final phrase synthesis demo.")  # noqa: T201
