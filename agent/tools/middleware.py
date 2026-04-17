from typing import Callable
from urllib import request

from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(    #工具执行监控
        #请求的数据封装
        request: ToolCallRequest,
        #执行函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
)->ToolMessage | Command:
    logger.info( f"[tool monitor]执行工具: {request.tool_call['name']} " )
    logger.info(f"[tool monitor]传入参数: {request.tool_call['args']} ")

    try:
        result = handler(request)

        logger.info( f"[tool monitor]工具: {request.tool_call['name']}调用成功")

        return result
    except Exception as e:
        logger.error( f"工具{request.tool_call['name']}调用失败,原因:{str(e)}")
        raise e

@before_model
def log_before_model(
        state: AgentState,  #整个agent智能体中的状态记录
        runtime: Runtime,   #记录了整个执行过程中上下文信息
):  #在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型,带有{len(state['messages'])}条消息")
    last_message = state["messages"][-1]
    message_type = type(last_message).__name__
    message_content = getattr(last_message, "content", "")
    if isinstance(message_content, str):
        message_content = message_content.strip()

    logger.debug(f"[log_before_model] {message_type} | {message_content}")
    return None


@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()