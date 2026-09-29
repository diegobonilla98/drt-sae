import base64
import json
import os
from pathlib import Path
from openai import OpenAI


def openai_client_from_env(api_key_env: str) -> OpenAI:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{api_key_env} is not set in this shell. Export it after activating the environment, "
            "or run with the environment variable prefixed on the command."
        )
    return OpenAI(api_key=api_key)


def parse_json_object(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        return json.loads(text[start:end + 1])
    return json.loads(text)


def llm_concept_messages(expert_id: int, examples: list[dict], random_examples: list[dict], stats: dict) -> list[dict]:
    evidence = []
    for item in examples:
        evidence.append({"activation": item.get("activation"), "token": item.get("token_text"), "context": item.get("context")})
    negatives = []
    for item in random_examples[:8]:
        negatives.append({"token": item.get("token_text"), "context": item.get("context")})
    prompt = {
        "task": "Name the human concept represented by one sparse autoencoder expert in an LLM residual stream.",
        "expert_id": expert_id,
        "activation_stats": stats,
        "top_activating_examples": evidence,
        "random_low_or_unknown_examples": negatives,
        "required_json_fields": ["expert_id", "short_label", "description", "positive_evidence", "negative_evidence", "likely_confounds", "confidence"]
    }
    return [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]


def cv_concept_messages(expert_id: int, contact_sheet_path: str, examples: list[dict], stats: dict) -> list[dict]:
    data = base64.b64encode(Path(contact_sheet_path).read_bytes()).decode("utf-8")
    evidence = [{"activation": item.get("activation"), "patch_index": item.get("patch_index"), "image_path": item.get("image_path")} for item in examples]
    text = json.dumps(
        {
            "task": "Name the visual concept represented by one sparse autoencoder expert in CLIP ViT patch activations.",
            "expert_id": expert_id,
            "activation_stats": stats,
            "top_activating_patch_examples": evidence,
            "required_json_fields": ["expert_id", "short_label", "description", "positive_evidence", "negative_evidence", "likely_confounds", "confidence"]
        },
        ensure_ascii=False,
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
            ],
        }
    ]


def chat_completion_body(model_name: str, messages: list[dict]) -> dict:
    return {
        "model": model_name,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }


def batch_chat_completion_request(custom_id: str, model_name: str, messages: list[dict]) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": chat_completion_body(model_name, messages),
    }


def describe_llm_concept(expert_id: int, examples: list[dict], random_examples: list[dict], stats: dict, model_name: str, api_key_env: str) -> dict:
    client = openai_client_from_env(api_key_env)
    response = client.chat.completions.create(
        **chat_completion_body(model_name, llm_concept_messages(expert_id, examples, random_examples, stats)),
    )
    row = parse_json_object(response.choices[0].message.content)
    row["expert_id"] = expert_id
    return row


def describe_cv_concept(expert_id: int, contact_sheet_path: str, examples: list[dict], stats: dict, model_name: str, api_key_env: str) -> dict:
    client = openai_client_from_env(api_key_env)
    response = client.chat.completions.create(
        **chat_completion_body(model_name, cv_concept_messages(expert_id, contact_sheet_path, examples, stats)),
    )
    row = parse_json_object(response.choices[0].message.content)
    row["expert_id"] = expert_id
    row["contact_sheet_path"] = contact_sheet_path
    return row


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def submit_chat_completion_batch(client: OpenAI, requests: list[dict], input_path: str | Path, metadata: dict | None = None):
    write_jsonl(input_path, requests)
    with Path(input_path).open("rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
    return client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata or {},
    )


def download_file_text(client: OpenAI, file_id: str) -> str:
    response = client.files.content(file_id)
    if hasattr(response, "read"):
        data = response.read()
    elif hasattr(response, "content"):
        data = response.content
    else:
        data = response
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return str(data)


def parse_batch_chat_completions(output_path: str | Path) -> dict[str, dict]:
    rows = {}
    for line in Path(output_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id")
        response = item.get("response") or {}
        body = response.get("body") or {}
        choices = body.get("choices") or []
        if not custom_id or not choices:
            continue
        content = choices[0].get("message", {}).get("content", "")
        rows[custom_id] = parse_json_object(content)
    return rows


def fallback_concept(expert_id: int, modality: str, examples: list[dict], stats: dict) -> dict:
    return {
        "expert_id": expert_id,
        "short_label": f"unlabeled_{modality}_expert_{expert_id}",
        "description": "No LLM labeling was run. Inspect the stored top examples for this expert.",
        "positive_evidence": examples[:8],
        "negative_evidence": [],
        "likely_confounds": [],
        "confidence": 0.0,
        "activation_stats": stats,
    }
