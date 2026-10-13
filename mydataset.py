import math
import random
from bisect import bisect_left

import torch
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset

IMG_H = 32
PAD_VAL = 0.0

BATCH_SIZE_BY_WIDTH = {
    384: 36,
    448: 32,
    512: 28,
    640: 22,
    768: 18,
}
CANDIDATE_W = sorted(BATCH_SIZE_BY_WIDTH.keys())
MAX_W = max(CANDIDATE_W)

class MyDataset(Dataset):
    def __init__(self, samples, normalize=True):
        self.samples = samples
        self.normalize = normalize
        self.canvas_widths = [self._read_canvas_width(image_path) for image_path, _ in samples]

    def __len__(self):
        return len(self.samples)
   
    def _resize_width(self,width, height):
        return min(max(round(width * IMG_H / height), 1), MAX_W)

    
    def _canvas_width(self,resized_width):
        index = bisect_left(CANDIDATE_W, resized_width)

        if index == len(CANDIDATE_W):
            raise ValueError(f"Resized width {resized_width} exceeds MAX_W={MAX_W}")

        return CANDIDATE_W[index]

    def _read_canvas_width(self, image_path):
        with Image.open(image_path) as image:
            resized_width = self._resize_width(*image.size)

        return self._canvas_width(resized_width)

    def _load_image(self, image_path, canvas_width):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            resized_width = self._resize_width(*image.size)
            image = image.resize((resized_width, IMG_H), Image.Resampling.BILINEAR)
            image_bytes = bytearray(image.tobytes())

        image = torch.frombuffer(image_bytes, dtype=torch.uint8)
        image = image.reshape(IMG_H, resized_width, 3).permute(2, 0, 1).float()
        image.div_(255.0)

        if self.normalize:
            image.sub_(0.5).div_(0.5)

        output = torch.full((3, IMG_H, canvas_width), PAD_VAL, dtype=torch.float32)
        output[:, :, :resized_width] = image
        return output, resized_width / canvas_width

    def __getitem__(self, index):
        image_path, label_text = self.samples[index]
        image, valid_ratio = self._load_image(image_path, self.canvas_widths[index])
        return image, label_text, valid_ratio


class WidthBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_sizes, shuffle=True, drop_last=True):
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.buckets = {width: [] for width in CANDIDATE_W}

        for index, width in enumerate(dataset.canvas_widths):
            self.buckets[width].append(index)

        missing_widths = set(CANDIDATE_W) - set(batch_sizes)

        if missing_widths:
            raise ValueError(f"Batch size is not defined for widths: {sorted(missing_widths)}")

    def __iter__(self):
        batches = []

        for width, indices in self.buckets.items():
            indices = indices.copy()
            batch_size = self.batch_sizes[width]

            if self.shuffle:
                random.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]

                if len(batch) == batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        total = 0

        for width, indices in self.buckets.items():
            batch_size = self.batch_sizes[width]

            if self.drop_last:
                total += len(indices) // batch_size
            else:
                total += math.ceil(len(indices) / batch_size)

        return total


class CollateFn:
    def __call__(self, batch):
        images = torch.stack([sample[0] for sample in batch])
        labels = [sample[1] for sample in batch]
        target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
        valid_ratios = torch.tensor([sample[2] for sample in batch], dtype=torch.float32)
        return images, labels, target_lengths, valid_ratios

def build_dataloader(samples, shuffle=True, drop_last=True, num_workers=4):
    dataset = MyDataset(samples, normalize=True)
    batch_sampler = WidthBatchSampler(
        dataset=dataset,
        batch_sizes=BATCH_SIZE_BY_WIDTH,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=CollateFn(),
    )


if __name__ == "__main__":
    samples = [
        ["image_001.jpg", "hello"],
        ["image_002.jpg", "world"],
    ]

    dataloader = build_dataloader(samples, num_workers=0)
    print("Batches:", len(dataloader))
