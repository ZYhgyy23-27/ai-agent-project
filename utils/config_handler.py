"""
yaml:一种专门用来写配置文件的格式
k:v
"""
import yaml

try:
    from utils.path_tool import get_abs_path
except ModuleNotFoundError:
    from path_tool import get_abs_path


#下面四个函数是用来读取配置文件的
def load_rag_config(config_path:str=get_abs_path('config/rag.yml'),encoding='utf-8'):
    with open(config_path,'r',encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_chroma_config(config_path:str=get_abs_path('config/chroma.yml'),encoding='utf-8'):
    with open(config_path,'r',encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_prompts_config(config_path:str=get_abs_path('config/prompts.yml'),encoding='utf-8'):
    with open(config_path,'r',encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_agent_config(config_path:str=get_abs_path('config/agent.yml'),encoding='utf-8'):
    with open(config_path,'r',encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()

if __name__ == '__main__':
    print(rag_conf["chat_model_name"])