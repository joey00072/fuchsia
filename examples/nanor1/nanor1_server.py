from fuchsia.vllm_server import DataSamplerServer, ServerConfig, Rollout
from fuchsia.reward_utils import clean_completion
from datasets import load_dataset
# from rich import print
from typing import Optional, List
from datasets import Dataset
from transformers import AutoTokenizer
from pathlib import Path    
from tiny_equation_dataset import TinyEquationDataset

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SYSTEM_PROMPT = """Respond in following format:
<think>
...
</think>
<answer>
...
</answer>


Find anwer in thinking tag, and give only final response itn answer tag
"""

def equation_validation(equation: str) -> bool:
    """
    Validate that `equation` contains ONLY:
    - digits 0-9
    - operators: + - * /
    - parentheses: ( )
    - optional whitespace

    Anything else (letters, commas, decimal points, equals signs, etc.) is rejected.
    """
    if equation is None:
        return False
    if not isinstance(equation, str):
        return False

    allowed = set("0123456789+-*/() \t\r\n")
    if any(ch not in allowed for ch in equation):
        return False

    # Require at least one digit to avoid accepting empty / operator-only strings.
    return any(ch.isdigit() for ch in equation)

def response_format_reward(sample: dict, s: str, *args, **kwargs) -> float:
    
    # print(f"{sample=}")
    print("--------------------------------")
    print(s)
    # print(f"{('<think>' in s)=}")
    print("--------------------------------")
    
    """Improved reward function with better validation and scoring."""
    already_clean = kwargs.get("already_clean", False)
    if not already_clean:
        s = clean_completion(
            s,
            tokenizer=kwargs.get("tokenizer"),
            token_ids=kwargs.get("completion_ids"),
        )

    START_THINKING_TOKEN = "<think>"
    END_THINKING_TOKEN = "</think>"
    START_ANSWER_TOKEN = "<answer>"
    END_ANSWER_TOKEN = "</answer>"
    idx = kwargs["idx"]
    format_reward = 0
    content_reward = 0
    try:
        if not s:
            return 0
        
        if START_THINKING_TOKEN in s and s.count(START_THINKING_TOKEN) == 1:
            format_reward += 0.05
        if END_THINKING_TOKEN in s and s.count(END_THINKING_TOKEN) == 1:
            format_reward += 0.05
        if START_ANSWER_TOKEN in s and s.count(START_ANSWER_TOKEN) == 1:
            format_reward += 0.05
        if END_ANSWER_TOKEN in s and s.count(END_ANSWER_TOKEN) == 1:
            format_reward += 0.05
        
        if format_reward >= 0.2:
            think, answer = s.split(END_THINKING_TOKEN)
            if "24 * 10 + (6 - 4)" in think:
                format_reward -= 0.5
            if "Okay, user want equation with number 10, 24, 4, 6 is equal to 242," in think:
                format_reward -= 0.5
            if "24 * 10 + (6 - 4)" in answer:
                format_reward -= 0.5
            
            if (
                START_THINKING_TOKEN in think
                and START_ANSWER_TOKEN in answer
                and END_ANSWER_TOKEN in answer
            ):
                format_reward += 0.1
            
        if START_ANSWER_TOKEN in s and END_ANSWER_TOKEN in s and s.count(START_ANSWER_TOKEN) == 1 and s.count(END_ANSWER_TOKEN) == 1 and START_THINKING_TOKEN in s:
            answer = s.split(START_ANSWER_TOKEN)[1].split(END_ANSWER_TOKEN)[0]
            if "=" in answer:
                answer = answer.split("=")[0]
                format_reward -= 0.05
                
            is_eqn = False
            try:
                int(answer)
            except Exception as e:
                content_reward +=0.1
                is_eqn = True
            finally:
                pass
                
            if is_eqn:
                try:
                    output = eval(answer)
                    if int(output) == int(sample['answer']):
                        content_reward += 0.6
                    else:
                        content_reward += 0.2
                except Exception as e:
                    content_reward -= 0.1
            
        return format_reward + content_reward
    except Exception as e:
        print(f"Error in response_format_reward: {e}")
        return 0


idx = 0
def reward_function_1(rollouts: List[Rollout], *args, **kwargs):
    global idx
    idx += 1
    print(f"{idx=}")
    lst = []
    cleaned = kwargs.get("cleaned_completions")
    for i, rollout in enumerate(rollouts):
        completion = cleaned[i] if cleaned is not None else rollout.completion
        reward = response_format_reward(
            rollout.item,
            completion,
            idx=idx,
            already_clean=cleaned is not None,
            tokenizer=kwargs.get("tokenizer"),
            completion_ids=rollout.completion_ids,
        )
        lst.append(reward)
        
    return lst

def human_join(nums):
    nums = [str(x) for x in nums]
    if not nums:
        return ""
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return f"{nums[0]} and {nums[1]}"
    return ", ".join(nums[:-1]) + " and " + nums[-1]

# print(human_join([324, 232, 667, 8778]))
# 324, 232, 667 and 8778

def prepare_dataset(dataset: Dataset, tokenizer) -> Dataset:
    ("first think in <think></think> tag and then give final answer in <answer></answer>"
    "tag.can you " )
    
    x = """Respond in following format:

example:
what equation with number 10, 24, 4, 6 is equal to 242
<think>
Okay, user want equation with number 10, 24, 4, 6 is equal to 242,

242 is big number so there must be some multiplication in equation,
i'll try to make equation with multiplication first
lets try 24*10 = 240
wait that close to 242

but I still need to add 2
so I'll try 24*10 + 2 = 242

but 2 is not avalible in numbers
so I'll try 24*10 + 6 = 246

maybe, i should multiply 6 with 4
24*10 + 6 * 4 = 242

wait, I made a mistake
6 * 4 = 24
and 24 * 10 = 240
so 24*10 + 6 * 4 = 264 not 242

wait, 6 - 4 = 2 
aha I got it, 

24*10 + (6-4) = 242

I am confident that this is correct answer
24*10 = 240
6 - 4 = 2
240 + 2 = 242

Therefore final answer is
24*10 + (6-4) = 242

but user only want equation 
so I will give final answer only in answer tag
24*10 + (6-4)

</think>
<answer>
24 * 10 + (6 - 4)
</answer>

Now you try to solve this
"""
    
    def process_example(example):
        # n1,n2,n3,n4 = example['numbers']
        example['numbers'] = [str(n) for n in example['numbers']]
        example["text"] = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":  f"find equation with {human_join(example['numbers'])} and using operations +,- to get {example['answer']}, give such equation only in <answer> tag" },
            ],
            tokenize=False,
        )+" system\n"
        return example
    dataset = dataset.map(process_example)
    return dataset

def main():
    server_config = ServerConfig.from_yaml(Path(__file__).parent / "nanor1_config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(server_config.model)
    # dataset = load_dataset(server_config.dataset_name, server_config.dataset_split)["train"]
    dataset = TinyEquationDataset(n=4, min_n=4).build_dataset(size=1024*8)
    dataset = dataset.shuffle()
    dataset = prepare_dataset(dataset, tokenizer)
    
    server = DataSamplerServer(server_config, dataset, [reward_function_1])
    server.serve()


if __name__ == "__main__":
    main()
    
