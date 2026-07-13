# --- vendor-sync:ignore-start ---
"""Qwen3-VL chat wrapper — vendored copy for the backend serving image.

Source of truth: modeling/chartqa/models/qwen_vl_chat.py. This copy is kept
self-contained so the backend Docker image builds without the modeling tree or
its training-only deps (e.g. `datasets`). The two files are kept in sync by
backend/tests/test_vendor_sync.py, which fails if the shared logic (everything
outside the vendor-sync:ignore markers) diverges — edit both, or the test fails.
"""
# --- vendor-sync:ignore-end ---
import importlib.util

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


def build_quantization_config(quantization: str | None) -> BitsAndBytesConfig | None:
    """Map a quantization mode to a `BitsAndBytesConfig` (None = full precision).

    Modes: "none"/None -> full precision (returns None); "4bit" -> NF4 with
    double quantization and bf16 compute dtype; "8bit" -> standard LLM.int8().

    Quantization is an explicit opt-in: if it is requested and `bitsandbytes`
    is not installed, fail loudly here with an actionable message instead of
    erroring later inside `from_pretrained` — this is not a fail-open gate.
    """
    mode = (quantization or "none").strip().lower()
    if mode == "none":
        return None
    if mode not in ("4bit", "8bit"):
        raise ValueError(
            f"Unknown quantization mode {quantization!r}; expected 'none', '8bit' or '4bit'."
        )
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            f"Quantization {mode!r} was requested but 'bitsandbytes' is not installed. "
            "Install it (pip install bitsandbytes) or use quantization 'none'."
        )
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


class QwenVLChat:
    """Wrapper around Qwen3-VL for single-image visual chat."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: str = "sdpa",
        adapter_path: str | None = None,
        quantization: str | None = None,
    ):
        # Opt-in 4-bit/8-bit loading; None/"none" keeps full precision.
        quantization_config = build_quantization_config(quantization)
        quant_kwargs = (
            {"quantization_config": quantization_config} if quantization_config else {}
        )
        # Load the model on the available device(s) (device_map="auto" uses the GPU).
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            **quant_kwargs,
        )
        # Optionally load a LoRA adapter checkpoint on top of the base model.
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            if quantization_config is None:
                # Fold the LoRA weights into the base model for faster inference.
                self.model = self.model.merge_and_unload()
            # With a quantized base the adapter must stay attached: merge_and_unload()
            # cannot fold LoRA deltas into 4-/8-bit weights, so inference runs
            # through the PeftModel wrapper (slightly slower, memory-cheap).
        # Prefer the checkpoint's processor (in case it added tokens) and fall
        # back to the base model's when running without an adapter.
        self.processor = AutoProcessor.from_pretrained(adapter_path or model_name)

    @staticmethod
    def build_messages(
        image=None,
        text: str = "",
        system_prompt: str | None = None,
        answer: str | None = None,
        history: list | None = None,
        images: list | None = None,
    ) -> list:
        """Build the chat `messages` structure for the Qwen processor.

        Shared by inference (`chat`) and training. Pass `answer` to append the
        assistant turn (used to build supervised training targets).

        Images: pass a single `image` (URL, path, or PIL image) for the one-chart case, or
        `images` (a list, oldest -> newest) for multi-chart conversations. With more than
        one image they're presented up front, NUMBERED ("Image 1", "Image 2", …) in a
        leading user turn, so a question can reference "image 1"; a short guidance line
        tells the model the newest image is the default when none is named. With exactly
        one image the output matches the original single-image prompt (no numbering), so
        existing evals/tests don't shift. Images ride only in that leading turn — Qwen
        re-reads them in context, so re-attaching per turn would waste vision tokens.

        Pass `history` (prior `{"role": "user"|"assistant", "text": str}` turns, oldest
        first) for multi-turn chat. When `history`/`images` are empty this is exactly the
        original single-turn, single-image behavior.
        """
        image_list = list(images) if images else ([image] if image is not None else [])

        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )

        # Present all images once, numbered, in a leading user turn. For a single image we
        # keep the classic "[image][question]" shape (no numbering) so nothing shifts.
        if len(image_list) > 1:
            lead = []
            for i, img in enumerate(image_list, start=1):
                lead.append({"type": "image", "image": img})
                lead.append({"type": "text", "text": f"Image {i}"})
            lead.append({"type": "text", "text": (
                f"There are {len(image_list)} chart images above, numbered 1 to "
                f"{len(image_list)}. Unless the question names a specific image, answer "
                f"about Image {len(image_list)} (the most recent)."
            )})
            messages.append({"role": "user", "content": lead})
            images_attached = True
        else:
            images_attached = False

        def _user_content(turn_text: str) -> list:
            nonlocal images_attached
            if not images_attached and image_list:
                images_attached = True
                return [{"type": "image", "image": image_list[0]},
                        {"type": "text", "text": turn_text}]
            return [{"type": "text", "text": turn_text}]

        for turn in history or []:
            role = turn.get("role")
            turn_text = turn.get("text", "")
            if role == "user":
                messages.append({"role": "user", "content": _user_content(turn_text)})
            elif role == "assistant":
                messages.append(
                    {"role": "assistant", "content": [{"type": "text", "text": turn_text}]}
                )

        messages.append({"role": "user", "content": _user_content(text)})
        if answer is not None:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            )
        return messages

    def chat(
        self,
        image=None,
        text: str = "",
        system_prompt: str | None = None,
        max_new_tokens: int = 128,
        history: list | None = None,
        images: list | None = None,
    ) -> str:
        """Send image(s) + text (with optional system prompt) and return the reply.

        Pass a single `image` (URL, path, or PIL image), or `images` (a list) for a
        multi-chart conversation. Pass `history` (prior `{"role", "text"}` turns) to
        continue a multi-turn conversation.
        """
        messages = self.build_messages(
            image, text, system_prompt, history=history, images=images
        )

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
