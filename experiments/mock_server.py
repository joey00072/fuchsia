import json
import fastapi
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import asyncio
from datasets import Dataset, load_dataset
import requests

app = FastAPI()


class MockServer:
    def __init__(self):
        file_name = "train_data_16k.jsonl"
        with open(file_name, "r") as f:
            self.data = [json.loads(line) for line in f]
        print(type(self.data))
        self.dataset_iter = iter(self.data)
        self.buffer_size = 4
        self.buffer = []
        self._is_filling = False
        self._epoch = 1
        self.buffer_fill()

    def buffer_fill(self):
        """Fill the buffer with samples from the dataset"""
        if self._is_filling:
            return

        self._is_filling = True
        try:
            while len(self.buffer) < self.buffer_size:
                try:
                    item = next(self.dataset_iter)
                    self.buffer.append(item)
                except StopIteration:
                    self.dataset_iter = iter(self.data)
                    self._epoch += 1
        finally:
            self._is_filling = False

        @app.get("/health/")
        async def health():
            """Health check endpoint"""
            return {"status": "ok"}

        @app.get("/get_sample/")
        async def get_sample(background_tasks: BackgroundTasks):
            """Get a sample from the buffer and trigger background buffer fill"""
            if len(self.buffer) == 0:
                await asyncio.sleep(5)
                return {"sample": None}

            sample = self.buffer.pop(0)
            if len(self.buffer) < self.buffer_size:
                background_tasks.add_task(self.buffer_fill)

            return {"sample": sample}

        @app.get("/buffer_status/")
        async def buffer_status():
            """Get the current status of the buffer"""
            return {
                "current_size": len(self.buffer),
                "max_size": self.buffer_size,
                "is_filling": self._is_filling,
                "epoch": self._epoch,
            }

    def serve(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the FastAPI server"""
        uvicorn.run(app, host=host, port=port)


class MockClient:
    def __init__(self):
        self.url = "http://localhost:8000/"

    def get_sample(self):
        url = self.url + "get_sample/"
        response = requests.post(url)
        return response.json()["sample"]


if __name__ == "__main__":
    server = MockServer()
    server.serve()
