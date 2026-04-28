import argparse  # noqa: D100
import inspect
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
	for args in ((2, 32), (32,), ()):  # noqa: C408
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


def build_train_dataset(config):  # noqa: ANN001, ANN201, D103
	def _build_pitch_dict_file(f0_folder_path, f0_dict_path):  # noqa: ANN001, ANN201, D103
		f0_folder = Path(f0_folder_path)
		f0_dict_file = Path(f0_dict_path)
		if f0_dict_file.exists() or not f0_folder.exists():
			return

		f0_dict_file.parent.mkdir(parents=True, exist_ok=True)
		pitch_dict = {}
		for pitch_file in f0_folder.rglob("*.pth"):
			rel = pitch_file.relative_to(f0_folder).as_posix()
			audio_rel = rel[:-4] if rel.endswith(".pth") else rel
			basename = Path(audio_rel).name
			stem = Path(audio_rel).stem
			pitch_tensor = torch.load(pitch_file, map_location="cpu")

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
		"label_pattern": config.label_pattern,
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
	return DynBatchDataset(**filtered_kwargs)


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
	num_workers=config.num_workers,
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

criterion = FastPitchLoss(dur_loss_toofast_scale=1.0)
attention_kl_loss = AttentionBinarizationLoss()


critic = build_critic(device)

optimizer_d = torch.optim.AdamW(
	critic.parameters(),
	lr=config.d_lr,
	betas=(config.d_beta1, config.d_beta2),
	weight_decay=config.weight_decay,
)
chunk_len = 128

# resume from existing checkpoint
n_epoch, n_iter = 0, 0

if config.restore_model != "":
	state_dicts = torch.load(config.restore_model, map_location=device)
	model.load_state_dict(state_dicts["model"])
	if "model_d" in state_dicts:
		critic.load_state_dict(state_dicts["model_d"], strict=False)
	if "optim" in state_dicts:
		optimizer.load_state_dict(state_dicts["optim"])
	if "optim_d" in state_dicts:
		optimizer_d.load_state_dict(state_dicts["optim_d"])
	if "epoch" in state_dicts:
		n_epoch = state_dicts["epoch"]
	if "iter" in state_dicts:
		n_iter = state_dicts["iter"]
else:
	model_sd = torch.load("G:/models/fastpitch/nvidia_fastpitch_210824+cfg.pt")
	model.load_state_dict(
		{k.removeprefix("module."): v for k, v in model_sd["state_dict"].items()},
	)

writer = SummaryWriter(config.log_dir)


model.train()

for epoch in range(n_epoch, config.epochs):
	train_dataset.shuffle()
	for batch in train_loader:
		x, y, _ = batch_to_gpu(batch)

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

		# discriminator step
		d_org, fmaps_org = critic_forward(
			critic, chunks_org_.requires_grad_(True), cond_vecs  # noqa: FBT003
		)
		d_gen, _ = critic_forward(critic, chunks_gen_.detach(), cond_vecs)

		loss_d = 0.5 * (d_org - 1).square().mean() + 0.5 * d_gen.square().mean()

		critic.zero_grad()
		loss_d.backward()
		optimizer_d.step()

		# generator step
		loss, meta = criterion(y_pred, y)

		d_gen2, fmaps_gen = critic_forward(critic, chunks_gen_, cond_vecs)
		loss_score = (d_gen2 - 1).square().mean()
		loss_fmatch = calc_feature_match_loss(fmaps_gen, fmaps_org)

		loss += config.gan_loss_weight * loss_score
		loss += config.feat_loss_weight * loss_fmatch

		binarization_loss = attention_kl_loss(attn_hard, attn_soft)
		loss += 1.0 * binarization_loss

		optimizer.zero_grad()
		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
		optimizer.step()

		# LOGGING
		meta["loss_d"] = loss_d.detach()
		meta["score"] = loss_score.detach()
		meta["fmatch"] = loss_fmatch.detach()
		meta["kl_loss"] = binarization_loss.detach()


		print(f"loss: {meta['loss'].item()} gnorm: {grad_norm}")  # noqa: T201

		for k, v in meta.items():
			writer.add_scalar(f"train/{k}", v.item(), n_iter)

		if n_iter % config.n_save_states_iter == 0:
			save_states(
				"states.pth",
				model,
				critic,
				optimizer,
				optimizer_d,
				n_iter,
				epoch,
				net_config,
				config,
			)

		if n_iter % config.n_save_backup_iter == 0 and n_iter > 0:
			save_states(
				f"states_{n_iter}.pth",
				model,
				critic,
				optimizer,
				optimizer_d,
				n_iter,
				epoch,
				net_config,
				config,
			)

		n_iter += 1


save_states(
	"states.pth",
	model,
	critic,
	optimizer,
	optimizer_d,
	n_iter,
	epoch,
	net_config,
	config,
)


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
