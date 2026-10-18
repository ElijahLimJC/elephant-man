import json
import os
import time
import random
from mistralai.client import Mistral

from dotenv import load_dotenv

load_dotenv()


api_key = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

SYSTEM_PROMPT = """You are an NLP dataset engineer specializing in dialogue fine-tuning.
Your task is to generate 2 distinct alternate user instruction prompts for a given character response.

Guidelines:
1. Target Character: John Merrick from Bernard Pomerance's play "The Elephant Man".
2. The character's response MUST remain identical and serve as a direct, natural answer to each new instruction.
3. Conversational query: A direct question asked by a visitor, physician, or curious companion.
4. Thematic query: An inquiry probing the philosophical, social, or emotional tension behind the statement.
5. Return ONLY a valid JSON object with the keys "conversational_query" and "thematic_query"."""

def call_mistral_with_backoff(user_prompt: str, model_name: str = "mistral-small-latest", max_retries: int = 10):
    """Executes an API call with exponential backoff and jitter for rate limit mitigation."""
    base_delay = 2.0  # Initial wait time in seconds

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.complete(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            return response

        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for rate limit or server capacity issues (HTTP 429 / 503)
            if "429" in error_msg or "rate limit" in error_msg or "capacity" in error_msg:
                # Calculate exponential delay with randomized jitter
                jitter = random.uniform(0.5, 1.5)
                wait_time = (base_delay * (2 ** (attempt - 1))) + jitter
                print(f"[Rate Limited] Attempt {attempt}/{max_retries}. Backing off for {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                # Re-raise non-transient errors (e.g., auth, bad requests)
                print(f"[Error] Non-rate-limit exception encountered: {e}")
                raise e

    raise RuntimeError(f"Failed after {max_retries} retries due to persistent rate limiting.")

def expand_with_mistral(
    input_file: str = "merrick_cleaned_filtered.jsonl",
    output_file: str = "merrick_expanded_99.jsonl",
    model_name: str = "mistral-small-latest",
    request_delay: float = 1.0  # Proactive throttle between successful requests
):
    with open(input_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(samples)} curated samples from {input_file}.")
    expanded_dataset = []

    for idx, item in enumerate(samples, 1):
        original_instruction = item["instruction"]
        response_text = item["output"]

        print(f"[{idx}/{len(samples)}] Generating instruction variants...")

        user_prompt = f"""Original Script Cue/Prompt:
"{original_instruction}"

Merrick's Fixed Output:
"{response_text}"

Generate two diverse alternative instruction prompts that naturally elicit this exact output.
Respond strictly in this JSON format:
{{
  "conversational_query": "<user prompt here>",
  "thematic_query": "<user prompt here>"
}}"""

        try:
            response = call_mistral_with_backoff(user_prompt, model_name=model_name)
            parsed = json.loads(response.choices[0].message.content)

            # 1. Retain original script pair
            expanded_dataset.append({
                "instruction": original_instruction,
                "output": response_text
            })

            # 2. Add conversational variant
            expanded_dataset.append({
                "instruction": parsed["conversational_query"],
                "output": response_text
            })

            # 3. Add thematic variant
            expanded_dataset.append({
                "instruction": parsed["thematic_query"],
                "output": response_text
            })

        except Exception as e:
            print(f"Failed to expand item {idx}: {e}. Retaining original sample only.")
            expanded_dataset.append({
                "instruction": original_instruction,
                "output": response_text
            })

        # Proactive pause between iterations to respect requests-per-second (RPS) limits
        time.sleep(request_delay)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in expanded_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nExpansion complete: Generated {len(expanded_dataset)} pairs.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    expand_with_mistral()