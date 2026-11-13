"""Additional linear adapters over frozen embeddings, with optional learned temperature."""
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .config import INITIAL_TEMPERATURE, MAX_LOGIT_SCALE


class DualProjection(nn.Module):
    def __init__(self, input_dim, output_dim, *, initial_temperature=INITIAL_TEMPERATURE,
                 learnable_temperature=True, max_logit_scale=MAX_LOGIT_SCALE):
        super().__init__()
        if not math.isfinite(initial_temperature) or initial_temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        if not math.isfinite(max_logit_scale) or max_logit_scale < 1 / initial_temperature:
            raise ValueError('scale ceiling must include the initial inverse temperature')
        self.text = nn.Linear(input_dim, output_dim, bias=False)
        self.image = nn.Linear(input_dim, output_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / initial_temperature)),
                                        requires_grad=learnable_temperature)
        self.max_logit_scale = max_logit_scale

    def forward(self, text, image):
        return F.normalize(self.text(text), dim=-1), F.normalize(self.image(image), dim=-1)

    def scale(self):
        # Clamp BEFORE exp, so a very large log value cannot overflow first.
        return self.logit_scale.clamp(max=math.log(self.max_logit_scale)).exp()


def symmetric_infonce(text, image, temperature=INITIAL_TEMPERATURE, *, logit_scale=None):
    if text.ndim != 2 or text.shape != image.shape or len(text) < 2:
        raise ValueError('at least two aligned text/image rows are needed')
    if not torch.isfinite(text).all() or not torch.isfinite(image).all():
        raise ValueError('nonfinite features')
    if (text.norm(dim=-1) <= 1e-12).any() or (image.norm(dim=-1) <= 1e-12).any():
        raise ValueError('zero projected feature')
    if logit_scale is None:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        scale = 1 / temperature
    else:
        # This argument is exp(logit_scale), not the stored logarithm.
        scale = logit_scale
        if scale.ndim != 0 or not torch.isfinite(scale) or scale <= 0:
            raise ValueError('a finite positive scalar scale is required')
    logits = F.normalize(text, dim=-1) @ F.normalize(image, dim=-1).T * scale
    labels = torch.arange(len(text), device=text.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train_features(feature_path, out_dir, steps=60, lr=0.01, *, initial_temperature=INITIAL_TEMPERATURE,
                   learnable_temperature=True, max_logit_scale=MAX_LOGIT_SCALE, batch_size=32):
    if steps <= 0 or not math.isfinite(lr) or lr <= 0 or batch_size < 2:
        raise ValueError('positive steps, learning rate and batch_size >= 2 required')
    torch.manual_seed(7)
    data = json.loads(Path(feature_path).read_text(encoding='utf-8'))
    text = torch.tensor(data['text_features'], dtype=torch.float32)
    image = torch.tensor(data['image_features'], dtype=torch.float32)
    if text.ndim != 2 or text.shape != image.shape or len(text) < 2:
        raise ValueError('paired feature matrices must align and have at least two rows')
    if len(data['ids']) != len(text) or len(set(data['ids'])) != len(text):
        raise ValueError('unique feature IDs must align with rows')
    if not torch.isfinite(text).all() or not torch.isfinite(image).all():
        raise ValueError('nonfinite frozen features')
    model = DualProjection(text.shape[1], text.shape[1], initial_temperature=initial_temperature,
                           learnable_temperature=learnable_temperature, max_logit_scale=max_logit_scale)
    heads = list(model.text.parameters()) + list(model.image.parameters())
    groups = [{'params': heads, 'weight_decay': 0.01}]
    if learnable_temperature:
        groups.append({'params': [model.logit_scale], 'weight_decay': 0.0})
    optimizer = torch.optim.AdamW(groups, lr=lr)
    dataset = torch.utils.data.TensorDataset(text, image)
    actual_batch = min(batch_size, len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=actual_batch, shuffle=True, drop_last=False)
    losses, temperatures = [], []
    batches = iter(loader)
    # steps denotes optimizer updates, not epochs. Singleton remainders have
    # no in-batch negative, so skip them rather than reporting a zero loss.
    while len(losses) < steps:
        try:
            batch_text, batch_image = next(batches)
        except StopIteration:
            batches = iter(loader)
            continue
        if len(batch_text) < 2:
            continue
        optimizer.zero_grad(set_to_none=True)
        projected_text, projected_image = model(batch_text, batch_image)
        loss = symmetric_infonce(projected_text, projected_image, logit_scale=model.scale())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(max=math.log(max_logit_scale))
        losses.append(float(loss.detach()))
        temperatures.append(float(1 / model.scale().detach()))
    settings = {'initial_temperature': initial_temperature, 'learnable_temperature': learnable_temperature,
                'max_logit_scale': max_logit_scale}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'schema_version': 2, 'input_dim': text.shape[1], 'output_dim': text.shape[1],
                'base_encoder': data['encoder'], 'temperature_config': settings,
                'projection_kind': 'additional_linear_on_frozen_embeddings',
                'state_dict': model.state_dict()}, out_dir / 'projection.pt')
    mode = 'synthetic_feature_training' if data.get('synthetic', True) else 'frozen_feature_training'
    report = {'mode': mode, 'seed': 7, 'steps': steps, 'learning_rate': lr, 'batch_size': actual_batch,
              'losses': losses, 'temperatures': temperatures, 'temperature_config': settings,
              'final_logit_scale': float(model.logit_scale.detach()),
              'final_inverse_temperature': float(model.scale().detach()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'note': 'training-loss demonstration only; no held-out domain recall or ablation metric'}
    (out_dir / 'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
