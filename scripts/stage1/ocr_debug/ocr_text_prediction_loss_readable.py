#!/usr/bin/env python3
"""可读版 OCR text prediction loss。

这个文件用于解释 OCR teacher-forcing loss 的核心机制，不直接接入训练。
真实训练实现见：
    tokenizer/tokenizer_image/vq/vq_train.py
    class DeepSeekOCRTeacherForcingLoss
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def add_deepseek_ocr_image_format_tokens(image_features: torch.Tensor, ocr_core) -> torch.Tensor:
    """补齐 DeepSeek-OCR language model 需要的 image token 格式。

    DeepSeek-OCR 不只是把 image features 当作一串普通 token。
    它会在每一行视觉 token 后面加一个 image_newline，
    最后再加一个 view_separator。

    这部分是 DeepSeek-OCR 的内部格式要求，和 loss 思路本身无关。
    """
    batch_size, num_tokens, hidden_dim = image_features.shape
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"DeepSeek-OCR image token count must be square, got {num_tokens}")

    image_features = image_features.view(batch_size, side, side, hidden_dim)

    newline = ocr_core.image_newline.to(
        device=image_features.device,
        dtype=image_features.dtype,
    ).view(1, 1, 1, hidden_dim)
    newline = newline.expand(batch_size, side, 1, hidden_dim)

    image_features = torch.cat([image_features, newline], dim=2)
    image_features = image_features.reshape(batch_size, side * (side + 1), hidden_dim)

    view_separator = ocr_core.view_seperator.to(
        device=image_features.device,
        dtype=image_features.dtype,
    ).view(1, 1, hidden_dim)
    view_separator = view_separator.expand(batch_size, 1, hidden_dim)

    return torch.cat([image_features, view_separator], dim=1)


class OCRTextPredictionLoss(torch.nn.Module):
    """
    OCR loss 的目的：
    让 tokenizer 的 reconstruction 不只是像原图，
    还要尽量保留能被 OCR 读出来的文字信息。

    这里不直接比较 OCR 最终输出的字符串，
    因为“生成字符串 -> 和 GT text 比较”这个过程不能直接反向传播。

    所以使用冻结 OCR 模型内部的文本预测 loss：
    - 给 OCR 看 reconstruction；
    - 用当前图片的 GT text 作为目标文本；
    - 如果 reconstruction 里的文字更清楚，OCR 更容易预测 GT text，loss 更小；
    - OCR 模型参数冻结，梯度只回传到 tokenizer decoder。
    """

    def __init__(
        self,
        ocr_model_path: str,
        device: torch.device,
        image_size: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        prompt: str = "<image>\n<|grounding|>OCR this image.",
        max_text_tokens: int = 256,
    ):
        super().__init__()

        if "<image>" not in prompt:
            raise ValueError("prompt must contain <image>.")

        self.prompt = prompt
        self.image_size = image_size
        self.max_text_tokens = max_text_tokens
        self.dtype = dtype

        # 加载 OCR tokenizer，用来把 GT text 转成 token ids。
        self.ocr_tokenizer = AutoTokenizer.from_pretrained(
            ocr_model_path,
            trust_remote_code=True,
        )

        # 加载 DeepSeek-OCR。
        # 这里只把它当作 loss network 使用，不训练 OCR 本身。
        self.ocr_model = AutoModel.from_pretrained(
            ocr_model_path,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=dtype,
        ).eval().to(device)

        # 冻结 OCR 参数。
        # 注意：冻结参数不等于切断梯度。
        # loss 仍然可以通过 OCR 的 image features 回传到 reconstruction。
        for param in self.ocr_model.parameters():
            param.requires_grad_(False)

    def preprocess_reconstruction(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """
        tokenizer 输出的 reconstruction 范围是 [-1, 1]。
        DeepSeek-OCR 使用固定输入尺寸，所以这里 resize 到 image_size。
        """
        reconstruction = reconstruction.clamp(-1, 1)

        if reconstruction.shape[-2:] != (self.image_size, self.image_size):
            reconstruction = F.interpolate(
                reconstruction,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
            )

        return reconstruction.to(dtype=self.dtype)

    def encode_reconstruction_as_ocr_image_features(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """
        把 reconstruction 送进冻结 OCR 的视觉分支，
        得到 OCR language model 可以接收的 image features。

        这里不能 detach reconstruction，
        因为我们希望 OCR loss 的梯度能回传到 tokenizer decoder。
        """
        reconstruction = self.preprocess_reconstruction(reconstruction)

        ocr_core = self.ocr_model.get_model()

        # DeepSeek-OCR 的视觉分支。
        sam_features = ocr_core.sam_model(reconstruction)
        vision_features = ocr_core.vision_model(reconstruction, sam_features)

        # DeepSeek-OCR 会把不同来源的视觉特征拼起来，
        # 再投影到 language model 的 hidden dimension。
        image_features = torch.cat(
            [
                vision_features[:, 1:],
                sam_features.flatten(2).permute(0, 2, 1),
            ],
            dim=-1,
        )
        image_features = ocr_core.projector(image_features)

        # 补 DeepSeek-OCR 需要的 image_newline / view_separator。
        image_features = add_deepseek_ocr_image_format_tokens(
            image_features,
            ocr_core,
        )

        return image_features

    def build_text_prediction_inputs(
        self,
        reconstruction: torch.Tensor,
        gt_texts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构造 OCR language model 的输入。

        输入逻辑是：

            prompt_before + image_features + prompt_after + GT text

        loss 只算 GT text 部分。
        prompt 和 image 部分只是条件，不参与 loss。
        """
        image_features = self.encode_reconstruction_as_ocr_image_features(
            reconstruction
        )

        batch_size, num_image_tokens, _ = image_features.shape

        if batch_size != len(gt_texts):
            raise ValueError(f"batch size {batch_size} != number of texts {len(gt_texts)}")

        text_before, text_after = self.prompt.split("<image>", 1)

        before_token_ids = self.ocr_tokenizer.encode(
            text_before,
            add_special_tokens=False,
        )
        after_token_ids = self.ocr_tokenizer.encode(
            text_after,
            add_special_tokens=False,
        )

        bos_id = 0
        image_token_id = 128815
        pad_id = self.ocr_tokenizer.pad_token_id or 0

        input_id_rows = []
        label_rows = []
        image_token_masks = []

        for gt_text in gt_texts:
            target_token_ids = self.ocr_tokenizer.encode(
                str(gt_text),
                add_special_tokens=False,
            )

            # 限制目标文本长度，避免 OCR loss 占用过多显存。
            target_token_ids = target_token_ids[: self.max_text_tokens]

            # OCR LM 的输入 token。
            input_ids = (
                [bos_id]
                + before_token_ids
                + [image_token_id] * num_image_tokens
                + after_token_ids
                + target_token_ids
            )

            # 只监督 GT text。
            # -100 是 PyTorch cross entropy 的 ignore index。
            # 也就是 prompt 和 image token 不算 loss。
            num_condition_tokens = (
                1
                + len(before_token_ids)
                + num_image_tokens
                + len(after_token_ids)
            )
            labels = [-100] * num_condition_tokens + target_token_ids

            image_mask = [False] * len(input_ids)
            image_start = 1 + len(before_token_ids)
            image_end = image_start + num_image_tokens
            image_mask[image_start:image_end] = [True] * num_image_tokens

            input_id_rows.append(input_ids)
            label_rows.append(labels)
            image_token_masks.append(image_mask)

        # padding 到同一长度，方便 batch 计算。
        max_len = max(len(row) for row in input_id_rows)

        input_ids = torch.full(
            (batch_size, max_len),
            pad_id,
            dtype=torch.long,
            device=reconstruction.device,
        )
        labels = torch.full(
            (batch_size, max_len),
            -100,
            dtype=torch.long,
            device=reconstruction.device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=reconstruction.device,
        )
        image_token_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool,
            device=reconstruction.device,
        )

        for i in range(batch_size):
            row_len = len(input_id_rows[i])

            input_ids[i, :row_len] = torch.tensor(
                input_id_rows[i],
                device=reconstruction.device,
            )
            labels[i, :row_len] = torch.tensor(
                label_rows[i],
                device=reconstruction.device,
            )
            attention_mask[i, :row_len] = 1
            image_token_mask[i, :row_len] = torch.tensor(
                image_token_masks[i],
                device=reconstruction.device,
            )

        # 普通 text token 走 OCR LM 自己的 embedding table。
        input_embeddings = self.ocr_model.get_input_embeddings()(input_ids).clone()

        # 关键步骤：
        # 把 <image> token 的普通 embedding 替换成 reconstruction 提取出来的 image features。
        # 这样 OCR 文本预测 loss 才能对 reconstruction 反向传播。
        for i in range(batch_size):
            input_embeddings[i, image_token_mask[i], :] = image_features[i].to(
                dtype=input_embeddings.dtype
            )

        return input_embeddings, attention_mask, labels

    def forward(self, reconstruction: torch.Tensor, gt_texts: Sequence[str]) -> torch.Tensor:
        """
        返回 OCR 文本预测 loss。

        如果 reconstruction 能让 OCR 更容易预测 GT text，
        这个 loss 会变小。
        """
        input_embeddings, attention_mask, labels = self.build_text_prediction_inputs(
            reconstruction,
            gt_texts,
        )

        outputs = self.ocr_model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        ocr_text_prediction_loss = outputs.loss

        return ocr_text_prediction_loss
