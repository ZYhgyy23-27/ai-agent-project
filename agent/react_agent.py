import os,sys
if __package__ is None or __package__ == "":
    # 兼容直接运行当前文件：把项目根目录加入模块搜索路径
    # 当前文件位于 agent/，因此需要回退两级到项目根目录
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
from langchain.agents import create_agent
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
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[calc_distance_by_address, get_weather,rag_summarize,get_lng_lat,local_get_user_location],
            middleware=[ monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self,query:str):
        input_dict = {
            "messages": [
                {"role": "user", "content":query}
            ]
        }

        for chunk in self.agent.stream(input_dict,stream_mode="values",context={"report":False}):
            latest_message=chunk["messages"][-1]
            if latest_message:
                yield latest_message.content.strip()+"\n"

if __name__ == "__main__":
    agent = ReactAgent()
    for chunk in agent.execute_stream("我想去自驾游有什么推荐的吗"):
        print(chunk,end="",flush=True)