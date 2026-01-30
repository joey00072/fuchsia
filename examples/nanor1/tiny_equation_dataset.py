import random
from dataclasses import dataclass
from rich import print
import json
from datasets import Dataset
from tqdm import tqdm
@dataclass
class Equation:
    eqn: str
    numbers: list
    ops: list
    answer: int
    
    def to_json(self):
        return {
            "eqn": self.eqn,
            "numbers": self.numbers,
            "ops": self.ops,
            "answer": self.answer
        }

class TinyEquationDataset:
    def __init__(self,n=4, min_n=None):
        self.n = n 
        self.ops = ["+","-"]#"+","+","-","*","/"]
        self.min_n = min_n if min_n is not None else n

    def build_dataset(self, size=1024):
        self._dict: dict[str, Equation] = {}

        pbar = tqdm(total=size, desc="Building dataset")
        attempts = 0
        max_attempts = max(size * 50, 10_000)
        while len(self._dict) < size and attempts < max_attempts:
            attempts += 1
            n = random.randint(self.min_n, self.n)
            numbers = [random.randint(10,100) for _ in range(n)]
            ops = [random.choice(self.ops) for _ in range(n-1)]
            total = numbers[0]
            eqn_str = "".join([str(total)] + [str(op) + str(num) for op, num in zip(ops, numbers[1:])])
            total = eval(eqn_str)     
            if int(total) != float(total):
                continue
            prev_len = len(self._dict)
            self._dict[eqn_str] = Equation(eqn_str, numbers, ops, int(total))
            if len(self._dict) > prev_len:
                pbar.update(1)
        pbar.close()
        if len(self._dict) < size:
            raise RuntimeError(
                f"Only generated {len(self._dict)} unique equations after {attempts} attempts. "
                f"Try reducing size or widening the sampling range."
            )
            

        # with open("./nanor1_dataset.jsonl", "w") as f:
        #     for eqn in self._dict.values():
        #         f.write(json.dumps(eqn.to_json()) + "\n")
        
        print(f"Dataset built with {len(self._dict)} equations")
        ds = Dataset.from_list([eqn.to_json() for eqn in self._dict.values()])
        return ds

if __name__ == "__main__":  
    n = 4
    min_n = 3
    size = 1024*8
    dataset = TinyEquationDataset(n, min_n)
    ds = dataset.build_dataset(size)
    print(ds[0])
