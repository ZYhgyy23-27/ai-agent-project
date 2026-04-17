import sys
import os.path

if __package__ is None or __package__ == "":
    # 兼容直接运行当前文件：把项目根目录加入模块搜索路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import  rag_conf


def _get_dashscope_api_key() -> str:
    """
    统一读取 DashScope API Key。
    Streamlit Cloud 需要在 Secrets 或环境变量中配置 DASHSCOPE_API_KEY。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "缺少 DASHSCOPE_API_KEY。请在部署环境中配置该环境变量（Streamlit Cloud: Manage app -> Settings -> Secrets）。"
        )
    return api_key


class BaseModelFactory(ABC):
    @abstractmethod  #@abstractmethod = 必须实现
        #生成模型
    def generator(self)->Optional[Embeddings|BaseChatModel]: #返回类型为Embeddings|BashChatModel实例
        pass

#聊天模型工厂
class ChatModelFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings|BaseChatModel]:
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            dashscope_api_key=_get_dashscope_api_key(),
        )

#嵌入模型工厂
class EmbeddingsFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings|BaseChatModel]:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=_get_dashscope_api_key(),
        )

chat_model = ChatModelFactory().generator()
embed_model=EmbeddingsFactory().generator()