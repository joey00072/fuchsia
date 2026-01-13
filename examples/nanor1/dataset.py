import random
from dataclasses import dataclass
from rich import print
import json


@dataclass
class Equation:
    eqn: str
    numbers: list
    ops: list
    total: int
    answer: int
    
    def to_json(self):
        return {
            "eqn": self.eqn,
            "numbers": self.numbers,
            "ops": self.ops,
            "total": self.total,
            "answer": self.answer
        }

class Dataset:
    def __init__(self,n=4):
        self.n = n 
        self.ops = ["+","-","*","/"]


    def build_dataset(self, size=1024):
        self._dict:[int,Equation] = {}

        while len(self._dict) <size:
            numbers = [random.randint(10,100) for _ in range(self.n)]
            ops = [random.choice(self.ops) for _ in range(self.n-1)]
            total = numbers[0]
            eqn_str = "".join([str(total)] + [str(op) + str(num) for op, num in zip(ops, numbers[1:])])
            total = eval(eqn_str)     
            if int(total) != float(total):
                continue
            self._dict[int(total)] = Equation(eqn_str, numbers, ops, int(total), total)
            

        with open("./nanor1_dataset.jsonl", "w") as f:
            for eqn in self._dict.values():
                f.write(json.dumps(eqn.to_json()) + "\n")

    def get_dataset(self):
        return self._dict

if __name__ == "__main__":
    n = 4
    size = 1024*8
    dataset = Dataset(n)
    dataset.build_dataset(size)
