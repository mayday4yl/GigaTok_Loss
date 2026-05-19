from pathlib import Path
import time

p = Path("/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation/tokenizer/tokenizer_image/vq/glyph_byt5.py")
stamp = time.strftime("%Y%m%d_%H%M%S")
backup = p.with_name(p.name + f".before_cache_position_patch_{stamp}")
backup.write_bytes(p.read_bytes())
s = p.read_text(encoding="utf-8")
old = """        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=False,
            output_attentions=output_attentions,
        )
"""
new = """        # transformers>=4.48 T5Attention expects cache_position even when use_cache=False.
        # Keep semantics identical by using the current sequence positions.
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=False,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
"""
if old not in s:
    raise SystemExit("target block not found")
p.write_text(s.replace(old, new), encoding="utf-8")
print(f"patched={p}")
print(f"backup={backup}")
