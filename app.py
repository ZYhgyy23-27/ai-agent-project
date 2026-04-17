import sys,os
import time

if __package__ is None or __package__ == "":
    # 兼容直接运行当前文件：把项目根目录加入模块搜索路径
    # 当前文件位于 agent/，因此需要回退两级到项目根目录
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from agent.react_agent import ReactAgent
import streamlit as st

#标题
st.title("小钟自驾出游助手")
st.divider()

#做一个保护防止ReactAgent多次创建消耗性能
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

#历史消息构建
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

#用户输入
prompt = st.chat_input()
#显示用户提问
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content":prompt})

    response_messages =[]
    with st.spinner("小钟思考中"):
       res_stream = st.session_state["agent"].execute_stream(prompt)

       def capture(generator,cache_list):
           for chunk in generator:
               cache_list.append(chunk)

               for char in chunk:
                   time.sleep(0.01)
                   yield char

    st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
    assistant_text = "".join(response_messages)
    st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
    st.rerun()