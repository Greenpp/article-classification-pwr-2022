import torch
from .config import TrainingConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

class BERTPreprocessor:
    def __init__(self, segment_size: int, segment_overlap: int, encoding_dim: int, device: str) -> None:
        self.segment_size = segment_size
        self.segment_overlap = segment_overlap
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.model)
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            TrainingConfig.model, num_labels=encoding_dim
            )
        self.encoder.trainable = False
        self.encoder.to(self.device)

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
        self, input_ids: torch.Tensor, att_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        reshaped_attention_mask = self._reshape_tensor(att_mask, padding_size, segments, unique_size)

        return reshaped_input_ids, reshaped_attention_mask

    def _split_batch(self, batch: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
        encodings = []
        encoding_end = 0
        for i in range(len(lengths)):
            encoding_start = encoding_end
            encoding_end = encoding_start + lengths[i]
            encodings.append(batch[encoding_start:encoding_end])

        return encodings

    def preprocess(self, data: list[dict]) -> list:
        texts, labels = []
        for d in data: 
            texts.append(d['text'])
            labels.append(d['label'])
        
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = tokens['input_ids']
        att_masks = tokens['attention_mask']

        processed_data = []
        for article_input_ids, attention_mask, label in zip(input_ids, att_masks, labels):
            # TODO process in batches if not enough memory
            segmented_input_ids, segmented_att_mask = self._segment_tokens(article_input_ids, attention_mask)
            encodings =  self.encoder(segmented_input_ids, attention_mask=segmented_att_mask)
            processed_data.append({
                'encodings': encodings,
                'label': label,
            })

        return processed_data
