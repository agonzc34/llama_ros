#!/usr/bin/env python3

# MIT License

# Copyright (c) 2023  Miguel Ángel González Santamarta

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import rclpy
from rclpy.node import Node
from llama_ros.langchain import LlamaROS

from typing import Optional

from pydantic import BaseModel, Field


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: int = Field(
        description="How funny the joke is, from 1 to 10"
    )


class LlamaDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("structured_output_llama_node")

        self.declare_parameter(
            "prompt",
            "",
        )
        self.prompt = self.get_parameter("prompt").get_parameter_value().string_value

        self.tokens = 0
        self.initial_time = -1
        self.eval_time = -1

        self._llama_client = LlamaROS(
            temp=0.2,
            penalty_last_n=8,
        )

    def send_prompt(self) -> None:
        structured_llm = self._llama_client.with_structured_output(Joke)

        result = structured_llm.invoke("Tell me a joke about drunk people")
        
        print(result)

def main():
    rclpy.init()
    node = LlamaDemoNode()
    node.send_prompt()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
