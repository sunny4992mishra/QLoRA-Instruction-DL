"""Prompt formatting shared by training and generation.

Training and generation must build prompts in exactly the same way,
otherwise the fine-tuned model sees a different format at test time.
"""

# Fixed Hindi system prompt, added to every example in training and generation.
# In English: "You are an assistant. Follow the user's instruction and give a clear and
# correct answer in Hindi." Together with the chat template it adds 93 tokens to every example.
# Why fixed: a different prompt per run would add a second variable to the rank and data-size
# sweeps.
SYSTEM_PROMPT = "आप एक सहायक हैं। उपयोगकर्ता के निर्देश का पालन करें और हिंदी में स्पष्ट और सही उत्तर दें।"

# Used only when the tokenizer has no chat template (plain base models).
# "निर्देश" means "instruction" and "उत्तर" means "answer". Qwen2.5-Instruct has a chat
# template, so our runs never use this.
FALLBACK_PROMPT = "{system}\n\n### निर्देश:\n{instruction}\n\n### उत्तर:\n"


def _messages(instruction, system):
    # Build the conversation in the role/content format that chat templates expect:
    # [system message (optional), user message].
    msgs = [{"role": "user", "content": instruction}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    return msgs


def _apply_template(tokenizer, messages, add_generation_prompt):
    # The chat template turns the messages into the exact text format the model was trained
    # on. For Qwen2.5 that is "<|im_start|>role\ncontent<|im_end|>" for each turn.
    # tokenize=False returns a string, so we can split prompt and response as text.
    # add_generation_prompt=True appends the opening of an assistant turn
    # ("<|im_start|>assistant\n"), so the model's next tokens are its answer.
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
    # Used by generate.py at test time, and by build_example below as the prompt part of a
    # training example. Using the same function in both places keeps the formats identical.
    if tokenizer.chat_template:
        return _apply_template(tokenizer, _messages(instruction, system), add_generation_prompt=True)
    return (tokenizer.bos_token or "") + FALLBACK_PROMPT.format(
        system=system or "", instruction=instruction
    )


def build_example(tokenizer, instruction, response, system=SYSTEM_PROMPT):
    """Return (prompt_text, response_text). Loss is computed on response_text only.

    response_text includes the end-of-turn / EOS marker so the model learns to stop.
    """
    # The prompt part: system + user turn + the opening of the assistant turn.
    prompt = build_prompt(tokenizer, instruction, system)
    if not tokenizer.chat_template:
        return prompt, response + tokenizer.eos_token

    # Render the full conversation, including the reference answer as the assistant turn.
    msgs = _messages(instruction, system) + [{"role": "assistant", "content": response}]
    full = _apply_template(tokenizer, msgs, add_generation_prompt=False)
    # The full text must begin with exactly the prompt text, so that everything after it is
    # the response. train.py gives the prompt tokens the label -100 (no loss) and trains on
    # the response tokens only.
    # What breaks without this check: the slice below could start at the wrong place and silently
    # train on the wrong tokens.
    if not full.startswith(prompt):
        raise ValueError(
            "Chat template output for the full conversation does not start with the "
            "generation prompt, so the response cannot be separated for loss masking. "
            "Check this model's chat template."
        )
    # Everything after the prompt: the answer plus the end-of-turn marker (<|im_end|> for
    # Qwen). Training on that marker is what teaches the model to stop. If a row is cut at the
    # 1,024-token limit, this marker is lost, so that row never shows the model where to stop.
    return prompt, full[len(prompt):]
