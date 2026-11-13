"""Replace only this boundary when upstream embeddings become available."""
import json
from pathlib import Path
import numpy as np
from .vectors import unit_rows


class FixtureEncoder:
    """Lookup ideal upstream vectors. Unknown inputs fail instead of faking AI."""
    fingerprint = 'fixture-4d-v1'

    def __init__(self, fixture):
        self.data = json.loads(Path(fixture).read_text(encoding='utf-8'))
        self.lookup = self.data['queries']

    def encode(self, inputs, modality='text'):
        try:
            keys = [Path(text).stem if modality == 'image' else text for text in inputs]
            return unit_rows([self.lookup[modality + ':' + text] for text in keys])
        except KeyError as exc:
            raise ValueError('fixture only accepts its documented example inputs') from exc


class ChineseCLIPEncoder:
    """Real Chinese-CLIP ViT-B-16 API; weights download only on explicit use."""
    def __init__(self, model='ViT-B-16', device='cpu', download_root='models'):
        import torch
        import cn_clip.clip as clip
        self.torch, self.clip, self.device = torch, clip, device
        self.model, self.preprocess = clip.load_from_name(
            model, device=device, download_root=download_root)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.fingerprint = f'ChineseCLIP:{model}:base-v1'

    def encode(self, inputs, modality='text'):
        from PIL import Image
        if not inputs:
            raise ValueError('empty encoding batch')
        if modality not in ['text', 'image']:
            raise ValueError('modality must be text or image')
        with self.torch.inference_mode():
            if modality == 'text':
                tokens = self.clip.tokenize(inputs).to(self.device)
                result = self.model.encode_text(tokens)
            else:
                tensors = []
                for path in inputs:
                    with Image.open(path) as image:
                        tensors.append(self.preprocess(image.convert('RGB')))
                result = self.model.encode_image(self.torch.stack(tensors).to(self.device))
        return unit_rows(result.float().cpu().numpy())


class ProjectedEncoder:
    def __init__(self, base, checkpoint):
        import torch
        from .training import DualProjection
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['base_encoder'] != base.fingerprint:
            raise ValueError('checkpoint and base encoder differ')
        self.base, self.torch = base, torch
        self.projection = DualProjection(saved['input_dim'], saved['output_dim'],
                                          **saved.get('temperature_config', {}))
        state = dict(saved['state_dict'])
        # Version-1 checkpoints contain only the two heads. Their retrieval
        # space is unchanged: temperature is used only in the training loss.
        state.setdefault('logit_scale', self.projection.logit_scale.detach().clone())
        self.projection.load_state_dict(state)
        self.projection.eval()
        import hashlib
        digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:16]
        self.fingerprint = base.fingerprint + ':projection:' + digest

    def encode(self, inputs, modality='text'):
        features = self.base.encode(inputs, modality)
        with self.torch.inference_mode():
            layer = self.projection.text if modality == 'text' else self.projection.image
            result = layer(self.torch.as_tensor(features, dtype=self.torch.float32))
        return unit_rows(result.numpy())
