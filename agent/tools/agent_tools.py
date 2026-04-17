import os
import sys
if __package__ is None or __package__ == "":
    # 兼容直接运行当前文件：把项目根目录加入模块搜索路径
    # 当前文件位于 agent/tools/，因此需要回退三级到项目根目录
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
from langchain_core.tools import tool
import requests

GD_API_KEY = "9e8602aaec8b3e339fc5075bee76c7b1"

_rag_service = None


def _get_rag_service():
    global _rag_service
    if _rag_service is None:
        from rag.rag_service import RagSummarizeService
        _rag_service = RagSummarizeService()
    return _rag_service

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query:str)->str:
    return _get_rag_service().rag_summarize(query)

@tool(description="获取城市天气")
def get_weather(city: str) -> str:
    """
    调用 wttr.in 实时天气API，返回温度及天气状况
    参数:
    city：城市名称 如(杭州 / hangzhou)
    """
    #1.构建请求
    url = f"https://wttr.in/{city}"
    #2.配置参数
    params = {"format": "j1"}
    #3.发送请求
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    #4.解析响应
    data = r.json()
    current = (data.get("current_condition") or [{}])[0]
    temp_c = current.get("temp_C")
    desc = ((current.get("weatherDesc") or [{}])[0]).get("value")
    return f"{city}：{temp_c}°C，{desc}"

@tool("get_user_location", description="获取用户所在城市的名称，以字符串形式返回")
def local_get_user_location()->str:
    """
    优先读取用户手动设置的城市；若未设置，再通过公网 IP 定位接口获取当前城市。
    当主接口超时或缺少城市信息时，自动回退到其他接口；全部失败时返回“未知”。
    """
    manual_city = os.getenv("USER_CITY_OVERRIDE", "").strip()
    if manual_city:
        return manual_city

    providers = [
        (
            "https://ipwho.is/",
            None,
            lambda data: (
                data.get("city"),
                data.get("region"),
                data.get("country"),
            ) if data.get("success") else ("", "", ""),
        ),
        (
            "https://ipapi.co/json/",
            None,
            lambda data: (
                data.get("city"),
                data.get("region"),
                data.get("country_name"),
            ),
        ),
        (
            "https://ipinfo.io/json",
            None,
            lambda data: (
                data.get("city"),
                data.get("region"),
                data.get("country"),
            ),
        ),
        (
            "http://ip-api.com/json/",
            {
                "lang": "zh-CN",
                "fields": "status,message,country,regionName,city",
            },
            lambda data: (
                data.get("city"),
                data.get("regionName"),
                data.get("country"),
            ) if data.get("status") == "success" else ("", "", ""),
        ),
    ]

    for url, params, extractor in providers:
        try:
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            city, region, country = extractor(data)
            city = (city or "").strip()
            region = (region or "").strip()
            country = (country or "").strip()

            if city:
                return city
            if region:
                return region
            if country:
                return country
        except Exception:
            continue

    return "未知"



def _get_lng_lat_impl(address: str, city: str | None = None) -> tuple[float, float]:
    """
    将地址转换为经纬度坐标
    """
    url = 'https://restapi.amap.com/v3/geocode/geo'
    params = {
        'key': GD_API_KEY,
        'address': address,
        'city': city  # 可选，用于缩小搜索范围
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data['status'] == '1' and int(data['count']) > 0:
        location = data['geocodes'][0]['location']
        lng, lat = location.split(',')
        return float(lng), float(lat)
    else:
        raise Exception(f"地理编码失败：{data.get('info', '未知错误')}")

@tool(description="将地址将地址转换为经纬度坐标")
def get_lng_lat(address: str, city: str | None = None):
    return _get_lng_lat_impl(address, city)

@tool(description="输入两个地址，计算驾车导航距离")
def calc_distance_by_address(address1, address2, city1=None, city2=None):
    """
    输入两个地址，计算驾车导航距离（单位：米）
    """
    # 1. 获取起点和终点的经纬度
    origin_lng, origin_lat = _get_lng_lat_impl(address1, city1)
    dest_lng, dest_lat = _get_lng_lat_impl(address2, city2)

    # 2. 调用驾车路径规划API
    url = 'https://restapi.amap.com/v3/direction/driving'
    params = {
        'key': GD_API_KEY,
        'origin': f'{origin_lng},{origin_lat}',
        'destination': f'{dest_lng},{dest_lat}'
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data['status'] == '1' and int(data['count']) > 0:
        route = data['route']['paths'][0]
        return {
            'distance_meter': int(route['distance']),  # 导航距离（米）
            'duration_second': int(route['duration']),  # 预计耗时（秒）
            'distance_km': round(int(route['distance']) / 1000, 2)  # 转为公里
        }
    else:
        raise Exception(f"路径规划失败：{data.get('info', '未知错误')}")


# --- 使用示例 ---
if __name__ == '__main__':
    try:
        result = calc_distance_by_address('杭州市', '北京市')
        print(f"驾车导航距离：{result['distance_km']} 公里")
        print(f"预计耗时：{result['duration_second'] // 60} 分钟")
    except Exception as e:
        print(f"调用出错：{e}")