# MIT License

# Copyright (c) 2024  Alejandro González Cantón
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
import base64
import cv2
import numpy as np
from pydantic import BaseModel

from llama_ros.langchain import LlamaROSCommon
from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.msg import Message
from llama_msgs.srv import FormatChatMessages
from action_msgs.msg import GoalStatus

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.language_models import LanguageModelInput
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)


import jinja2


class ChatLlamaROS(BaseChatModel, LlamaROSCommon):
    @property
    def _default_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "chatllamaros"

    def _messages_to_chat_messages(
        self, messages: List[BaseMessage]
    ) -> tuple[FormatChatMessages.Request, Optional[str], Optional[np.ndarray]]:

        chat_messages = FormatChatMessages.Request()
        image_url = None
        image = None

        for message in messages:
            role = "user" if message.type.lower() == "human" else message.type
            
            contents = message.content if isinstance(message.content, list) else [message.content]
            for single_content in contents:
                if isinstance(single_content, str):
                    chat_messages.messages.append(Message(role=role, content=single_content))
                elif single_content.get("type") == "text":
                    chat_messages.messages.append(Message(role=role, content=single_content["text"]))
                elif single_content.get("type") == "image_url":
                    image_text = single_content["image_url"]["url"]
                    if "data:image" in image_text:
                        image_data = image_text.split(",")[-1]
                        decoded_image = base64.b64decode(image_data)
                        np_image = np.frombuffer(decoded_image, np.uint8)
                        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
                    else:
                        image_url = image_text

        return chat_messages, image_url, image

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        llama_client = self.llama_client.get_instance()

        chat_messages, image_url, image = self._messages_to_chat_messages(messages)
        formatted_prompt = llama_client.format_chat_prompt(chat_messages).formatted_prompt

        goal_action = self._create_action_goal(
            formatted_prompt, stop, image_url, image, **kwargs
        )

        result, status = LlamaClientNode.get_instance().generate_response(goal_action)

        if status != GoalStatus.STATUS_SUCCEEDED:
            return ""

        generation = ChatGeneration(message=AIMessage(content=result.response.text))
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:

        llama_client = self.llama_client.get_instance()

        chat_messages, image_url, image = self._messages_to_chat_messages(messages)
        formatted_prompt = llama_client.format_chat_prompt(chat_messages).formatted_prompt

        goal_action = self._create_action_goal(
            formatted_prompt, stop, image_url, image, **kwargs
        )

        for pt in LlamaClientNode.get_instance().generate_response(
            goal_action, stream=True
        ):

            if run_manager:
                run_manager.on_llm_new_token(
                    pt.text,
                    verbose=self.verbose,
                )

            yield ChatGenerationChunk(message=AIMessageChunk(content=pt.text))
