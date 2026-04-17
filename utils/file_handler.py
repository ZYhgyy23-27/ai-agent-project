import hashlib
import os
from xml.dom.minidom import Document

try:
    from utils.logger_handler import logger
except ModuleNotFoundError:
    from logger_handler import logger
from langchain_community.document_loaders import TextLoader

def get_file_md5_hex(filepath: str): #获取文件的md5值的十六进制字符串

    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]文件{filepath}不是文件")
        return

    md5_obj = hashlib.md5()

    chunk_size = 4096    #4KB的分片大小,避免文件过大爆内存
    try:
        with open(filepath, 'rb') as f:   #rb以二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)


        md5_hex = md5_obj.hexdigest() #hexdigest:把数据转成一串固定长度的 32 位 / 64 位十六进制字符串，用来当文件的唯一 “身份证”
        return md5_hex
    except Exception as e:
        logger.error(f"[计算文件{filepath}md5失败,{str(e)}]")
        return None



def listdir_with_allowed_type(path:str,allowed_types:tuple[str]):    #返回文件夹内的文件列表(允许的文件后缀)
    files = []

    if not os.path.isdir(path): #查看一个对象身上有哪些属性和方法的
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return tuple()

    for f in os.listdir(path): #列出文件夹里的东西
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)


# def pdf_loader(filepath:str,passwd=None):
#     return PyPDFLoader(filepath,passwd).load()



def text_loader(filepath:str,passwd=None)->list[Document]:
    # Windows 环境下默认编码可能是 gbk，优先按 utf-8 读取并启用自动编码探测
    return TextLoader(filepath, encoding="utf-8", autodetect_encoding=True).load()