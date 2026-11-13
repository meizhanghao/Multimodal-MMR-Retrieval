"""Real upstream feature export: paired files -> aligned frozen tower features."""
import json
from pathlib import Path
from .data import load_pairs, torch_dataset
from .vectors import unit_rows


def export_features(pairs_path, encoder, output, batch_size=4):
    import torch
    from torch.utils.data import DataLoader
    source = json.loads(Path(pairs_path).read_text(encoding='utf-8'))
    pairs, rejected = load_pairs(pairs_path, check_images=True)
    if rejected or not pairs:
        raise ValueError('invalid source pairs: ' + json.dumps(rejected, ensure_ascii=False))
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    loader = DataLoader(torch_dataset(pairs, encoder.preprocess), batch_size=batch_size, shuffle=False, num_workers=0)
    ids, texts, images = [], [], []
    with torch.inference_mode():
        for batch in loader:
            text_tokens = encoder.clip.tokenize(batch['text']).to(encoder.device)
            text_vectors = encoder.model.encode_text(text_tokens).float().cpu().numpy()
            image_vectors = encoder.model.encode_image(batch['image'].to(encoder.device)).float().cpu().numpy()
            ids.extend(batch['id'])
            texts.extend(unit_rows(text_vectors).tolist())
            images.extend(unit_rows(image_vectors).tolist())
    payload = {'encoder': encoder.fingerprint, 'synthetic': source.get('synthetic', True), 'ids': ids,
               'text_features': texts, 'image_features': images}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'pairs': len(ids), 'dimension': len(texts[0]), 'encoder': encoder.fingerprint, 'output': str(output)}
