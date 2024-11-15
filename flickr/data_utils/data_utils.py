import torch
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import sentencepiece as spm

ds = load_dataset("nlphuji/flickr30k", split="test")


train_ds = ds.filter(lambda x: x["split"] == "train")
test_ds = ds.filter(lambda x: x["split"] == "test")
val_ds = ds.filter(lambda x: x["split"] == "val")


class FlickrDataset(Dataset):
    def __init__(
        self,
        spm_model_path,
        split: str = "train",
        split_size: int | float = None,
        width: int = 480,
    ):
        self.tokenizer = spm.SentencePieceProcessor(model_file=spm_model_path)
        # Padding token is outside of vocab
        self.pad_id = self.tokenizer.vocab_size()
        self.img_to_tensor = T.ToTensor()

        if isinstance(split_size, float):
            if split_size > 1 or split_size < 0:
                raise ValueError("split_size should be between 0 and 1, if float")
            ratio = split_size
        hg_ds = load_dataset("nlphuji/flickr30k", split="test").filter(
            lambda x: x["split"] == split
        )

        if isinstance(split_size, int):
            ratio = split_size / len(hg_ds)
        if split_size is not None:
            # Fake random sampling of subset
            hg_ds = hg_ds.train_test_split(test_size=ratio)["test"]
        self.id_to_img = {
            int(row["img_id"]): self.resize_to_width(row["image"], width)
            for row in tqdm(hg_ds, desc="Preprocessing images")
        }

        self.img_caption_pairs: list[tuple[int, str]] = []
        for row in hg_ds:
            pairs = [(int(row["img_id"]), caption) for caption in row["caption"]]
            self.img_caption_pairs.extend(pairs)

    def resize_to_width(self, img: Image.Image, width: int):
        wpercent = width / float(img.size[0])
        hsize = int((float(img.size[1]) * float(wpercent)))
        return img.resize((width, hsize), Image.Resampling.LANCZOS)

    def __len__(self):
        return len(self.img_caption_pairs)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img_id, caption = self.img_caption_pairs[idx]
        return (
            self.as_flat_tensors(self.id_to_img[img_id]),
            torch.tensor(
                self.tokenizer.encode(caption, add_bos=True), dtype=torch.int32
            ),
            torch.tensor(
                self.tokenizer.encode(caption, add_eos=True), dtype=torch.int32
            ),
        )

    def as_flat_tensors(self, img: Image.Image, box_width: int = 16):
        data: torch.Tensor = self.img_to_tensor(img)
        patches = torch.nn.functional.unfold(
            data.unsqueeze(0),
            kernel_size=(box_width, box_width),
            stride=(box_width, box_width),
        )
        return patches.squeeze(0).T

    def collate_fn(self, batch):
        imgs, in_caption, out_caption = zip(*batch)

        img_lens = torch.tensor([img.size(0) for img in imgs])
        img_tnsr = pad_sequence(imgs, batch_first=True)

        # No need to calculate twice
        caption_lens = [len(caption) for caption in in_caption]
        in_tnsr = pad_sequence(in_caption, batch_first=True, padding_value=self.pad_id)
        out_tnsr = torch.cat(out_caption, dim=0)

        return (img_tnsr, img_lens), (in_tnsr, caption_lens), out_tnsr


if __name__ == "__main__":
    dataset = FlickrDataset(
        spm_model_path="./tokeniser/tknz_20000.model", split="val", split_size=10
    )
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    for batch in dl:
        pass
