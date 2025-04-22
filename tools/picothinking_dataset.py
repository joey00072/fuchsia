"""
Dataset wrapper for PicoThinking function-calling tasks.
Combines a seed dataset and a large main dataset, applies prompt formatting, and supports shuffling and indexing.
"""
from typing import Optional, Any, Iterator
from datasets import load_dataset, Dataset

from pathlib import Path
import yaml
from transformers import AutoTokenizer

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"[red]Failed to load config from {config_path}: {e}[/red]")
        raise
    
    
PREFIX = (
    "Online function calling is available while thinking.\n"
    "function call format:\n<function_call>\n<request>\n...\n</request>\n<response>\n...\n</response>\n</function_call>\nAvailable functions:\n\n"
)
SYSTEM_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic reasoning "
    "processes to help come to a correct solution prior to answering. You should enclose your thoughts "
    "and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
)

class PicoThinkingFunctionCalling:
    """
    Combines and prepares datasets for PicoThinking function-calling tasks.
    Applies prompt formatting and supports shuffling and indexing.
    """
    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        seed_dataset_name: str = "joey00072/pico_thinking_function_calling",
        main_dataset_name: str = "joey00072/exp-tool-calls-multiturn",
    ):
        self.tokenizer = tokenizer
        try:
            self.seed_dataset: Dataset = load_dataset(seed_dataset_name)["train"]
        except Exception as e:
            raise RuntimeError(f"Failed to load seed dataset '{seed_dataset_name}': {e}")
        try:
            self.main_dataset: Dataset = load_dataset(main_dataset_name)["train"]
        except Exception as e:
            raise RuntimeError(f"Failed to load main dataset '{main_dataset_name}': {e}")

        def prepare_seed(example: dict) -> dict:
            if "schema" not in example or "question" not in example:
                raise ValueError("Example missing required keys: 'schema' and/or 'question'")
            prompt = PREFIX + example["schema"] + "\n\n" + example["question"]
            example["tools"] = example["schema"]
            assert tokenizer is not None, "must provide tokenizer"
            if tokenizer is not None:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                example["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False)
            return example

        def prepare_main(example: dict) -> dict:
            prompt = PREFIX + example["schema"] + "\n\n" + example["question"]
            if tokenizer is not None:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                example["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False)
            return example

        self.seed_dataset = self.seed_dataset.map(prepare_seed)
        self.main_dataset = self.main_dataset.map(prepare_main)
        self._seed_len = len(self.seed_dataset)
        self._main_len = len(self.main_dataset)

    def __len__(self) -> int:
        """Return total length of combined datasets."""
        return self._seed_len + self._main_len

    def __iter__(self) -> Iterator[dict]:
        """Iterate over all examples in both datasets."""
        yield from self.seed_dataset
        yield from self.main_dataset

    def __getitem__(self, idx: int) -> dict:
        """Get an item by global index."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        if idx < self._seed_len:
            return self.seed_dataset[idx]
        return self.main_dataset[idx - self._seed_len]

    def shuffle(self, seed: int = 42) -> "PicoThinkingFunctionCalling":
        """Shuffle both datasets and return self."""
        self.seed_dataset = self.seed_dataset.shuffle(seed=seed)
        self.main_dataset = self.main_dataset.shuffle(seed=seed)
        return self

if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    from rich import print
    dataset = PicoThinkingFunctionCalling(tokenizer=tokenizer)
    for item in dataset:
        print(item)
        break


