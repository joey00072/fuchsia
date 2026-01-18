from fuchsia.vllm_server import DataSamplerServer, ServerConfig, Rollout
from datasets import load_dataset
from rich import print
from typing import Optional, List
from datasets import Dataset
from transformers import AutoTokenizer
from pathlib import Path    
from tiny_equation_dataset import TinyEquationDataset

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SYSTEM_PROMPT = """Respond in following format:
<think>
think deeply and solve the problem here
</think>
<answer>
give the final answer here
</answer>
"""

CURRICULUM_LEARNING = 0

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
    # print(s)
    # print(f"{('<think>' in s)=}")
    # print("--------------------------------")
    
    """Improved reward function with better validation and scoring."""
    START_OF_TEXT_TOKEN = "<|im_start|>"
    END_OF_TEXT_TOKEN = "<|eot_id|>"
    START_HEADER_TOKEN = "<|start_header_id|>"
    END_HEADER_TOKEN = "<|end_header_id|>"
    ASSISTANT_TOKEN = "assistant"
    USER_TOKEN = "user"

    START_THINKING_TOKEN = "<think>"
    END_THINKING_TOKEN = "</think>"
    START_ANSWER_TOKEN = "<answer>"
    END_ANSWER_TOKEN = "</answer>"
    idx = kwargs["idx"]
    format_reward = 0
    content_reward = 0
    try:
        
        if START_THINKING_TOKEN in s and s.count(START_THINKING_TOKEN) == 1:
            format_reward += 0.1
        if END_THINKING_TOKEN in s and s.count(END_THINKING_TOKEN) == 1:
            format_reward += 0.1
        if START_ANSWER_TOKEN in s and s.count(START_ANSWER_TOKEN) == 1:
            format_reward += 0.1
        if END_ANSWER_TOKEN in s and s.count(END_ANSWER_TOKEN) == 1:
            format_reward += 0.1
        
        if format_reward == 0.4:
            think, answer = s.split(END_THINKING_TOKEN)
            if (
                START_THINKING_TOKEN in think
                and START_ANSWER_TOKEN in answer
                and END_ANSWER_TOKEN in answer
            ):
                format_reward += 0.1
            
        if format_reward == 0.5:
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
                content_reward -= 0.1
                
            if is_eqn:
                try:
                    output = eval(answer)
                    if int(output) == int(sample['answer']):
                        content_reward += 0.4
                    else:
                        content_reward += 0.1
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
    for rollout in rollouts:
        reward = response_format_reward(rollout.item, rollout.completion , idx=idx)
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
<think>
think deeply and solve the problem here
</think>
<answer>
give the final answer here
</answer>
"""
    
    def process_example(example):
        # n1,n2,n3,n4 = example['numbers']
        example['numbers'] = [str(n) for n in example['numbers']]
        example["text"] = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x + f"find equation with {human_join(example['numbers'])} and basic arithmetic operations (+,-,*,/,'(',')') to get {example['answer']}, give such equation (only pure equation not any text or explanation not answer, only numbers and this characters (+,-,*,/,'(',')') in <answer> tag, make sure to confaim solution in thinking process)" },
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
    dataset = TinyEquationDataset(n=4, min_n=3).build_dataset(size=1024*8)
    dataset = dataset.shuffle()
    dataset = prepare_dataset(dataset, tokenizer)
    
    server = DataSamplerServer(server_config, dataset, [reward_function_1])
    server.serve()


if __name__ == "__main__":
    main()
    