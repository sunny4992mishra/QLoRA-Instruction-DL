"""Prompt formatting shared by training and generation.

Training and generation must build prompts in exactly the same way,
otherwise the fine-tuned model sees a different format at test time.
"""

SYSTEM_PROMPT = "आप एक सहायक हैं। उपयोगकर्ता के निर्देश का पालन करें और हिंदी में स्पष्ट और सही उत्तर दें।"

# Used only when the tokenizer has no chat template (plain base models).
FALLBACK_PROMPT = "{system}\n\n### निर्देश:\n{instruction}\n\n### उत्तर:\n"


def _messages(instruction, system):
    msgs = [{"role": "user", "content": instruction}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    return msgs


def _apply_template(tokenizer, messages, add_generation_prompt):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except Exception:
        # Some templates (e.g. Gemma) reject the system role. Merge it into the user turn.
        if messages[0]["role"] != "system":
            raise
        system, user = messages[0]["content"], messages[1]["content"]
        merged = [{"role": "user", "content": f"{system}\n\n{user}"}] + messages[2:]
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def build_prompt(tokenizer, instruction, system=SYSTEM_PROMPT):
    """Text the model sees before it starts answering."""
    if tokenizer.chat_template:
        return _apply_template(tokenizer, _messages(instruction, system), add_generation_prompt=True)
    return (tokenizer.bos_token or "") + FALLBACK_PROMPT.format(
        system=system or "", instruction=instruction
    )


def build_example(tokenizer, instruction, response, system=SYSTEM_PROMPT):
    """Return (prompt_text, response_text). Loss is computed on response_text only.

    response_text includes the end-of-turn / EOS marker so the model learns to stop.
    """
    prompt = build_prompt(tokenizer, instruction, system)
    if not tokenizer.chat_template:
        return prompt, response + tokenizer.eos_token

    msgs = _messages(instruction, system) + [{"role": "assistant", "content": response}]
    full = _apply_template(tokenizer, msgs, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError(
            "Chat template output for the full conversation does not start with the "
            "generation prompt, so the response cannot be separated for loss masking. "
            "Check this model's chat template."
        )
    return prompt, full[len(prompt):]
