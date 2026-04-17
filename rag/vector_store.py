from re import search
from xml.dom.minidom import Document

import os
import sys

if __package__ is None or __package__ == "":
    # 兼容直接运行当前文件：把项目根目录加入模块搜索路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import text_loader,listdir_with_allowed_type,get_file_md5_hex
from  utils.logger_handler import logger
from langchain_core.documents import Document

#向量存储服务
class VectorStoreService(object):
    def __init__(self):
        try:
            from langchain_chroma import Chroma
        except Exception as e:
            raise RuntimeError(
                "向量库依赖加载失败（langchain_chroma/chromadb/protobuf 版本可能冲突）。"
            ) from e

        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        ) #向量存储本身

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,

        ) #文档分割器

    #作用：从向量库里面，拿一个 “检索器” 出来。后面提问、查资料，全靠这个检索器。
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k":chroma_conf["k"]})

    def load_documents(self):
        """
        从数据文件夹读取数据文件，转为向量存入数据库
        要计算MD5做去重复
        :return: NONE
        """
        def check_md5_hex(md5_for_check:str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                #创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w",encoding="utf-8").close()
                return False #md5没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line=line.strip()
                    if line == md5_for_check:
                        return True #md5处理过
                return False  #md5没处理过

        def save_md5_hex(md5_for_check:str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check+"\n")

        def get_file_documents(read_path:str):
            if read_path.endswith(".txt"): #如果是以txt结尾
                return text_loader(read_path)

            return []

        allowed_files_path: list[str]= listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        loaded_count = 0
        skipped_count = 0
        failed_count = 0

        for path in allowed_files_path:
            #获取文件的MD5
            md5_hex=get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在于知识库内,跳过")
                skipped_count += 1
                continue

            try:
                documents:list[Document]=get_file_documents(path)

                if not documents:
                    logger.error(f"[加载知识库]{path}内没有有效文本内容,跳过")
                    failed_count += 1
                    continue

                split_document: list[Document]= self.spliter.split_documents(documents)

                if not split_document:
                    logger.error(f"[加载知识库]{path}分片后没有有效内容")
                    failed_count += 1
                    continue

                #将内容存入向量库
                self.vector_store.add_documents(split_document)

                #记录这个已经处理好的文件的md5，避免重复下载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path}内容加载成功")
                loaded_count += 1

            except Exception as e:
                #exc_info为True会记录详细的报错堆栈，如果为Flash仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{e}",exc_info=True)
                failed_count += 1
                continue

        logger.info(
            f"[加载知识库]完成：成功{loaded_count}个，跳过{skipped_count}个，失败{failed_count}个"
        )
        return {"loaded": loaded_count, "skipped": skipped_count, "failed": failed_count}

if __name__ == '__main__':
    #创建对象
    vs = VectorStoreService()
    #加载数据库
    load_stats = vs.load_documents()
    print(
        f"知识库加载结果：成功{load_stats['loaded']}，跳过{load_stats['skipped']}，失败{load_stats['failed']}"
    )
    #拿到检索器
    retriever = vs.get_retriever()
    #通过检索器查找关键字
    res = retriever.invoke("温州")
    print(f"检索命中条数：{len(res)}")
    if not res:
        print("未检索到相关内容，请确认向量库中是否已有数据，或尝试更换查询词。")
    for r in res:
        print(r.page_content)
        print("="*20)

