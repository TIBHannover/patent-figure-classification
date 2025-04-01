import os
import json
import shutil

from pathlib import Path

from tqdm import tqdm

root_dir = '/nfs/data/vip_at_scale/'

def load_json(path):

    with open(Path(path), 'r') as rf:
        return json.load(rf)

def save_json(path, data):

    with open(Path(path), 'w') as wf:
        json.dump(data, wf, indent=4)

def copy_image(from_path, to_path):

    shutil.copy(
        src=Path(root_dir+from_path),
        dst=Path(to_path)
    )

for aspect in ['type']: #, 'uspc', 'projection', 'object']:

    for split in ['val', 'test', 'train_150']:
        data = load_json(Path(f'{root_dir}/epo_dataset2/outputs/PatFigCLS/{aspect}/{split}.json'))
        for id, sample in tqdm(data.items(), desc=f'{aspect}/{split}'):
            image_write_path = f'PatFigCLS/images/{sample["figure_path"]}'
            os.makedirs('/'.join(image_write_path.split('/')[:-1]), exist_ok=True)
            copy_image(sample['figure_path'], image_write_path)
            data[id][sample['figure_path']] = image_write_path
        save_json(Path(f'{root_dir}/epo_dataset2/outputs/PatFigCLS/{aspect}/{split}.json'), data)
            