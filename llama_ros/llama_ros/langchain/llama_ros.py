# MIT License

# Copyright (c) 2024  Miguel Ángel González Santamarta

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


from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)
from operator import itemgetter
from pydantic import BaseModel

import json

from action_msgs.msg import GoalStatus
from llama_msgs.srv import Tokenize
from llama_ros.llama_client_node import LlamaClientNode
from llama_ros.langchain import LlamaROSCommon

from langchain_core.outputs import GenerationChunk
from langchain_core.language_models.llms import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage

from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.language_models import LanguageModelInput
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)

class LlamaROS(LLM, LlamaROSCommon):

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "llamaros"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        print(kwargs.get('tools'))

        goal = self._create_action_goal(prompt, stop, **kwargs)

        result, status = LlamaClientNode.get_instance().generate_response(goal)

        if status != GoalStatus.STATUS_SUCCEEDED:
            return ""
        return result.response.text


    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        print('with_structured_output')
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and is_basemodel_subclass(schema)
        if schema is None:
            raise ValueError(
                "schema must be specified when method is 'function_calling'. "
                "Received None."
            )
        
        tool_name = convert_to_openai_tool(schema)["function"]["name"]
        tool_choice = {"type": "function", "function": {"name": tool_name}}
        llm = self.bind_tools([schema], tool_choice=tool_choice)
        if is_pydantic_schema:
            print('is_pydantic_schema')
            output_parser = JsonOutputParser()

            
        print('end with_structured_output')

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, bool, str]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        print('bind_tools')
        """Bind tool-like objects to this chat model

        tool_choice: does not currently support "any", "auto" choices like OpenAI
            tool-calling API. should be a dict of the form to force this tool
            {"type": "function", "function": {"name": <<tool_name>>}}.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        tool_names = [ft["function"]["name"] for ft in formatted_tools]
        if tool_choice:
            if isinstance(tool_choice, dict):
                if not any(
                    tool_choice["function"]["name"] == name for name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice=} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            elif isinstance(tool_choice, str):
                chosen = [
                    f for f in formatted_tools if f["function"]["name"] == tool_choice
                ]
                if not chosen:
                    raise ValueError(
                        f"Tool choice {tool_choice=} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            elif isinstance(tool_choice, bool):
                if len(formatted_tools) > 1:
                    raise ValueError(
                        "tool_choice=True can only be specified when a single tool is "
                        f"passed in. Received {len(tools)} tools."
                    )
                tool_choice = formatted_tools[0]
            else:
                raise ValueError(
                    """Unrecognized tool_choice type. Expected dict having format like 
                    this {"type": "function", "function": {"name": <<tool_name>>}}"""
                    f"Received: {tool_choice}"
                )

        kwargs["tool_choice"] = tool_choice
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:

        goal = self._create_action_goal(prompt, stop, **kwargs)

        for pt in LlamaClientNode.get_instance().generate_response(goal, stream=True):

            if run_manager:
                run_manager.on_llm_new_token(
                    pt.text,
                    verbose=self.verbose,
                )

            yield GenerationChunk(text=pt.text)

    def get_num_tokens(self, text: str) -> int:
        req = Tokenize.Request()
        req.text = text
        tokens = self.llama_client.tokenize(req).tokens
        return len(tokens)
