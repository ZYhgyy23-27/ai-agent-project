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



class BaseModelFactory(ABC):
    @abstractmethod  #@abstractmethod = 必须实现
        #生成模型
    def generator(self)->Optional[Embeddings|BaseChatModel]: #返回类型为Embeddings|BashChatModel实例
        pass

#聊天模型工厂
class ChatModelFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings|BaseChatModel]:
        return ChatTongyi(model=rag_conf['chat_model_name'])

#嵌入模型工厂
class EmbeddingsFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings|BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf['embedding_model_name'])

chat_model = ChatModelFactory().generator()
embed_model=EmbeddingsFactory().generator()