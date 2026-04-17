import os
import sys
import math
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
    try:
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
        if temp_c is None and not desc:
            return f"{city}：天气接口暂无有效数据，请稍后重试。"
        return f"{city}：{temp_c}°C，{desc}"
    except Exception as e:
        return f"{city}：天气查询失败（{e}）。请稍后重试，或直接告诉我你想去的景点，我可先按常规天气为你规划。"

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



def _haversine_km(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    """计算两点球面距离（公里）。"""
    r = 6371.0
    lng1, lat1, lng2, lat2 = map(math.radians, [lng1, lat1, lng2, lat2])
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return r * c


def _get_lng_lat_impl(
    address: str,
    city: str | None = None,
    origin_coord: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """
    将地址转换为经纬度坐标
    """
    url = "https://restapi.amap.com/v3/geocode/geo"

    def _request_geocode(target_address: str, target_city: str | None):
        params = {
            "key": GD_API_KEY,
            "address": target_address,
        }
        if target_city and target_city.strip():
            params["city"] = target_city.strip()
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

    def _pick_best_geocode(
        data: dict,
        prefer_city: str | None,
        ref_coord: tuple[float, float] | None,
    ):
        geocodes = data.get("geocodes") or []
        if not geocodes:
            return None

        # 1) 有城市偏好时优先匹配
        if prefer_city:
            prefer_city = prefer_city.strip()
            for item in geocodes:
                formatted = item.get("formatted_address", "")
                item_city = item.get("city")
                item_city_text = "".join(item_city) if isinstance(item_city, list) else str(item_city or "")
                if prefer_city in formatted or prefer_city in item_city_text:
                    return item

        # 2) 若提供起点坐标，重名地点按“离起点最近”消歧（适用于跨省周边游，如沪苏浙）
        if ref_coord and len(geocodes) > 1:
            ref_lng, ref_lat = ref_coord
            best_item = None
            best_distance = float("inf")
            for item in geocodes:
                location = item.get("location")
                if not location:
                    continue
                try:
                    lng_text, lat_text = location.split(",")
                    item_lng = float(lng_text)
                    item_lat = float(lat_text)
                    d = _haversine_km(ref_lng, ref_lat, item_lng, item_lat)
                    if d < best_distance:
                        best_distance = d
                        best_item = item
                except Exception:
                    continue
            if best_item:
                return best_item

        return geocodes[0]

    prefer_city = city or os.getenv("USER_CITY_OVERRIDE", "").strip() or None
    attempts = [(address, prefer_city), (address, None)]
    if prefer_city and prefer_city not in address:
        attempts.append((f"{prefer_city}{address}", None))

    last_info = "未知错误"
    for target_address, target_city in attempts:
        try:
            data = _request_geocode(target_address, target_city)
            if data.get("status") == "1" and int(data.get("count", 0)) > 0:
                best = _pick_best_geocode(data, prefer_city, origin_coord)
                if best and best.get("location"):
                    lng, lat = best["location"].split(",")
                    return float(lng), float(lat)
            last_info = data.get("info", last_info)
        except Exception as e:
            last_info = str(e)
            continue

    raise Exception(f"地理编码失败：{last_info}，address={address}，city={city}")

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
    dest_lng, dest_lat = _get_lng_lat_impl(address2, city2, origin_coord=(origin_lng, origin_lat))

    # 2. 调用驾车路径规划API
    url = 'https://restapi.amap.com/v3/direction/driving'
    params = {
        'key': GD_API_KEY,
        'origin': f'{origin_lng},{origin_lat}',
        'destination': f'{dest_lng},{dest_lat}'
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data['status'] == '1' and int(data['count']) > 0:
            route = data['route']['paths'][0]
            return {
                'distance_meter': int(route['distance']),  # 导航距离（米）
                'duration_second': int(route['duration']),  # 预计耗时（秒）
                'distance_km': round(int(route['distance']) / 1000, 2)  # 转为公里
            }
        return {
            "error": f"路径规划失败：{data.get('info', '未知错误')}",
            "distance_meter": None,
            "duration_second": None,
            "distance_km": None,
        }
    except Exception as e:
        return {
            "error": f"路径规划接口调用失败：{e}",
            "distance_meter": None,
            "duration_second": None,
            "distance_km": None,
        }


# --- 使用示例 ---
if __name__ == '__main__':
    try:
        result = calc_distance_by_address('杭州市', '北京市')
        print(f"驾车导航距离：{result['distance_km']} 公里")
        print(f"预计耗时：{result['duration_second'] // 60} 分钟")
    except Exception as e:
        print(f"调用出错：{e}")