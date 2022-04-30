import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ArxivDataset(Dataset):
    def __init__(
        self, dataset, tokenizer, segment_size: int, segment_overlap: int
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.segment_size = segment_size
        self.segment_overlap = segment_overlap

    def _reshape_tensor(
        self, tensor: torch.Tensor, padding: int, segments: int, unique_size: int
    ) -> torch.Tensor:
        first_token = tensor[0].item()
        last_token = tensor[-1].item()

        content = tensor[1:-1]
        padded = F.pad(content, (0, padding))

        segments_list = []
        for i in range(segments):
            start = i * unique_size
            end = start + unique_size + self.segment_overlap

            segments_list.append(padded[start:end])

        segmented_tensor = torch.stack(segments_list)
        f_token_tensor = F.pad(segmented_tensor, (1, 0), value=first_token)
        fl_token_tensor = F.pad(f_token_tensor, (0, 1), value=last_token)
        fl_token_tensor[-1, -1] = 0
        fl_token_tensor[-1, -padding - 1] = last_token

        return fl_token_tensor

    def _segment_tokens(
        self, token_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = token_dict["input_ids"][0]

        unique_size = self.segment_size - 2 - self.segment_overlap
        segment_content = len(input_ids) - 2 - self.segment_overlap
        if segment_content < 0:
            raise ValueError("segment size is too large")

        segments = segment_content // unique_size
        left_content = segment_content % unique_size
        if left_content:
            padding_size = unique_size - left_content
            segments += 1
        else:
            padding_size = 0

        reshaped_input_ids = self._reshape_tensor(
            input_ids, padding_size, segments, unique_size
        )
        reshaped_attention_mask = torch.ones(
            (segments, self.segment_size), dtype=torch.int
        )
        if padding_size:
            reshaped_attention_mask[-1, -padding_size:] = 0

        return reshaped_input_ids, reshaped_attention_mask

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        txt = self.dataset[idx]["text"]
        tokens = self.tokenizer(txt, return_tensors="pt")
        segment_tokens = self._segment_tokens(tokens)

        label = self.dataset[idx]["label"]

        return *segment_tokens, label
