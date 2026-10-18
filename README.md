# Elephant Man

Code used to create **Elephant Man** — an LLM persona embodying John Merrick from Bernard Pomerance's stage play *The Elephant Man*, built on top of Llama 3.1 8B.

John Merrick speaks with gentle dignity, quiet vulnerability, and earnest curiosity — taking Victorian moral standards at face value to reveal their contradictions, never with malice or bitterness.

## Hugging Face Model

The quantised model weights and Ollama Modelfile are hosted on Hugging Face:
**[https://huggingface.co/Dankwiz/elephantman](https://huggingface.co/Dankwiz/elephantman)**

## Repository Contents

| File | Description |
|------|-------------|
| `parser.ipynb` | Parses source material into raw dialogue |
| `dataset_adder.py` | Builds and extends the training dataset |
| `elephantman_training.ipynb` | Fine-tuning notebook |
| `Modelfile` | Ollama Modelfile with system prompt, template, and inference parameters |

## Usage (Ollama)

Download the GGUF from Hugging Face, place it in the same directory as the `Modelfile`, then:

```bash
ollama create elephantman -f Modelfile
ollama run elephantman
```

## Character Details

- Speaks in soft-spoken, late-Victorian cadence — no modern slang, no archaic "thee/thou"
- Spends his days assembling a card model of St. Philip's Cathedral
- Sleeps sitting upright; believes in heaven and the redemption of suffering
- Never breaks character or acknowledges being an AI
