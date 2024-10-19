import requests
import pandas as pd
import json
from io import StringIO

# 定义URL和查询参数
url = "https://iswa.gsfc.nasa.gov/IswaSystemWebApp/hapi/data"
params = {
    "id":"goesp_part_flux_P5M",
    "time.min":"2024-10-09T00:00:00.0Z",
    "time.max":"2024-10-10T08:00:00.0Z",
    "include":"header"
}

response = requests.get(url, params=params)

# 检查请求是否成功
if response.status_code == 200:
    # 将响应内容转换为字符串
    content = response.text
    
    # 分离出 JSON 头部和 CSV 数据
    lines = content.splitlines()
    
    # 找到第一个不以 '#' 开头的行索引
    for i, line in enumerate(lines):
        if not line.startswith('#'):
            break

    # 解析 JSON 头部
    json_header = "\n".join(line[1:] for line in lines[:i])  # 去掉每行的 '#'
    header_data = json.loads(json_header)

    parameters = header_data["parameters"]
    columns = []
    for j in range(len(parameters)):
        columns.append(parameters[j]["name"])


    # 打印 JSON 头部信息
    print("JSON Header:")
    print(json.dumps(header_data, indent=4))

    # 读取 CSV 数据
    csv_data = "\n".join(lines[i:])
    df = pd.read_csv(StringIO(csv_data), names=columns)
    
    # 打印数据框内容
    print("\nCSV Data:")
    print(df['P10'].max())
    print(df['P50'].max())
    print(df['P100'].max())
    print(df['P500'].max())
else:
    print(f"请求失败，状态码: {response.status_code}")
