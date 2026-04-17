import os,sys
if __package__ is None or __package__ == "":
    # 兼容直接运行当前文件：把项目根目录加入模块搜索路径
    # 当前文件位于 agent/，因此需要回退两级到项目根目录
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
try:
    # LangChain v1 风格
    from langchain.agents import create_agent as lc_create_agent
    _AGENT_FACTORY = "langchain"
except Exception:
    # 兼容部分环境：使用 LangGraph 预构建 ReAct
    from langgraph.prebuilt import create_react_agent as lg_create_react_agent
    _AGENT_FACTORY = "langgraph"
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
if __package__:
    from .tools.agent_tools import (
        calc_distance_by_address,
        get_weather,
        rag_summarize,
        get_lng_lat,
        local_get_user_location,
    )
    from .tools.middleware import monitor_tool, log_before_model, report_prompt_switch
else:
    from agent.tools.agent_tools import (
        calc_distance_by_address,
        get_weather,
        rag_summarize,
        get_lng_lat,
        local_get_user_location,
    )
    from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        tools = [calc_distance_by_address, get_weather, rag_summarize, get_lng_lat, local_get_user_location]
        system_prompt = load_system_prompts()

        if _AGENT_FACTORY == "langchain":
            self.agent = lc_create_agent(
                model=chat_model,
                system_prompt=system_prompt,
                tools=tools,
                middleware=[monitor_tool, log_before_model, report_prompt_switch],
            )
        else:
            # LangGraph 分支没有与 LangChain v1 完全一致的 middleware 参数，
            # 先保证部署可运行，后续如需链路埋点可再迁移到 graph middleware。
            self.agent = lg_create_react_agent(
                model=chat_model,
                tools=tools,
                prompt=system_prompt,
            )

    def execute_stream(self, query: str, user_city: str | None = None):
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }

        if user_city and user_city.strip():
            # 给模型一个显式上下文，减少它调用 IP 定位工具的概率。
            input_dict["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": f"用户当前所在城市（用户手动提供）是：{user_city.strip()}。涉及位置相关问题时优先使用该信息。",
                },
            )

        for chunk in self.agent.stream(input_dict,stream_mode="values",context={"report":False}):
            latest_message=chunk["messages"][-1]
            if latest_message:
                yield latest_message.content.strip()+"\n"

if __name__ == "__main__":
    agent = ReactAgent()
    for chunk in agent.execute_stream("我想去自驾游有什么推荐的吗"):
        print(chunk,end="",flush=True)