import urllib.request
import os

def download_file(url, filename, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"正在下载 {filename}... (尝试 {attempt + 1}/{max_retries})")
            urllib.request.urlretrieve(url, filename)
            print(f"{filename} 下载完成")
            return True
        except Exception as e:
            print(f"下载失败: {str(e)}")
            if attempt < max_retries - 1:
                print("等待2秒后重试...")
                time.sleep(2)
            continue
    return False

# 创建模型目录
os.makedirs("models", exist_ok=True)

# 配置下载地址 - 使用多个备选镜像
models = [
    {
        "name": "haarcascade_frontalface_default.xml",
        "urls": [
            "https://gitee.com/mirrors/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml",
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        ]
    },
    {
        "name": "haarcascade_smile.xml",
        "urls": [
            "https://gitee.com/mirrors/opencv/raw/master/data/haarcascades/haarcascade_smile.xml",
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml"
        ]
    }
]

# 下载每个模型
for model in models:
    filename = os.path.join("models", model["name"])
    if os.path.exists(filename):
        print(f"{filename} 已存在，跳过下载")
        continue

    success = False
    for url in model["urls"]:
        if download_file(url, filename):
            success = True
            break
        print(f"尝试下一个镜像...")

    if not success:
        print(f"警告：{model['name']} 下载失败")
