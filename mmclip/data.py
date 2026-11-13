"""Normalize paired metadata; invalid rows are observable, never silent."""
from pathlib import Path
import json
import re
import unicodedata


def clean_text(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def load_pairs(path, check_images=False):
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    valid, rejected, seen = [], [], set()
    for row_number, row in enumerate(payload['items']):
        try:
            item = dict(row)
            for key in ['id', 'doc_id', 'paragraph_id', 'text', 'image']:
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError('missing string: ' + key)
            if item['id'] in seen:
                raise ValueError('duplicate id')
            item['text'] = clean_text(item['text'])
            item['image_path'] = str((path.parent / item['image']).resolve())
            if check_images and not Path(item['image_path']).is_file():
                raise ValueError('missing image')
            seen.add(item['id'])
            valid.append(item)
        except (ValueError, TypeError) as exc:
            rejected.append({'row': row_number, 'error': str(exc)})
    return valid, rejected


def torch_dataset(pairs, preprocess):
    """Factory keeps PyTorch optional for the vector-only learning path."""
    from torch.utils.data import Dataset
    from PIL import Image

    class PairedDataset(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, i):
            item = pairs[i]
            with Image.open(item['image_path']) as image:
                tensor = preprocess(image.convert('RGB'))
            return {'id': item['id'], 'text': item['text'], 'image': tensor}

    return PairedDataset()
