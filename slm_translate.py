from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto"
)

glossary = [
    "Password",
    "Firewall",
    "Malware",
    "Phishing",
    "Encryption",
    "Authentication",
    "Data breach",
    "Cyber attack"
]

for term in glossary:
    prompt = f"""Translate this cybersecurity glossary term from English to Tamil.
Keep the translation concise and technically meaningful.
Term: {term}
Tamil:"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    print(f"{term} -> {answer}")
