import os
import io

from PIL import Image

import webdataset as wds

import asyncio
import aiofiles

from torchvision import transforms as T
from concurrent.futures import ProcessPoolExecutor

from utils import load_json, get_config

CONFIG = get_config()
ROOT_DIR = CONFIG['root_dir']
WRITE_DIR = f"{ROOT_DIR}/vqa"

BATCH_SIZE = 512

def get_batches(_list, batch_size=512):
    """Batch list into batches of size batch_size"""
    batches = [_list[i:i + batch_size] for i in range(0, len(_list), batch_size)]
    return batches

def expand2square(image):
    """Expand image to square."""
    width, height = image.size
    
    if width == height:
        return image
    elif width > height:
        result = Image.new(image.mode, (width, width), color="white")
        result.paste(image, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(image.mode, (height, height), color="white")
        result.paste(image, ((height - width) // 2, 0))
        return result
    
async def get_image(path):

    async with aiofiles.open(os.path.join(ROOT_DIR, path), mode="rb") as f:
        image_data = await f.read()

    image = Image.open(io.BytesIO(image_data))
    image_resized = T.Compose([
        expand2square,
        T.Grayscale(num_output_channels=3),
        T.Resize((324, 324), interpolation=T.InterpolationMode.BICUBIC)
    ])(image)

    img_byte_arr = io.BytesIO()
    image_resized.save(img_byte_arr, format="PNG")

    return img_byte_arr.getvalue()

async def write_to_shard(data):
    
    _batch, split, aspect = data
    idx, batch = _batch

    write_dir = os.path.join(WRITE_DIR, aspect, split, f"shard-{int(idx):06d}.tar")

    with wds.TarWriter(write_dir, encoder=False) as sink:
        
        for id, sample in batch:
            
            image = await get_image(sample['figure_path'])

            if aspect == "object_held_out": aspect = "object"
            
            sink.write({
                "__key__": f"{id}",
                "concept.txt": sample[aspect].encode("utf-8"),
                "task.txt": sample["task"].encode("utf-8"),
                "question.txt": sample["question"].encode("utf-8"),
                "answer.txt": sample["answer"].encode("utf-8"),
                "image.png": image
            })

    print(f"Created shard-{int(idx):06d}.tar")

def process_batch(data):
    asyncio.run(write_to_shard(data))

async def main():

    for aspect in ["type", "projection", "object", "uspc"]:

        for split in ["train_9", "train_18", "train_27", "train_54", "train_81", "train_150", "test", "val"]:
        
            print(f"Aspect: {aspect} Split: {split}")
                
            data = load_json(
                os.path.join("dataset", "vqa", aspect, "all_tasks", f"{split}.json")
            )

            os.makedirs(os.path.join(WRITE_DIR, aspect, split), exist_ok=True)

            batch_ids = get_batches(list(data.keys()), batch_size=BATCH_SIZE*10)
            batches = [(idx, [(id, data[id]) for id in batch]) for idx, batch in enumerate(batch_ids)]
            
            loop = asyncio.get_event_loop()

            with ProcessPoolExecutor(max_workers=16) as executor:
                tasks = [loop.run_in_executor(executor, process_batch, 
                                            (batch, split, aspect)) for batch in batches]
                await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())

    





        


