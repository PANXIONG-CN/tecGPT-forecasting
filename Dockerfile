# 使用官方 Python 镜像作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量，防止 Python 生成 .pyc 文件和缓冲 stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# (可选) 创建一个专门用于存放预训练模型缓存的目录
# RUN mkdir -p /app/gpt2_cache_docker
# ENV TRANSFORMERS_CACHE=/app/gpt2_cache_docker
# ENV HF_HOME=/app/gpt2_cache_docker # Newer huggingface libs use this

# 更新 apt-get 并安装必要的包 (如果需要的话，例如 git for private repos)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#  && rm -rf /var/lib/apt/lists/*

# 复制需求文件并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目代码到工作目录
COPY . .

# (可选) 如果 HDF5 文件较小且希望包含在镜像中 (不推荐用于大数据)
# COPY data_preparation /app/data_preparation
# COPY processed_data /app/processed_data

# (可选) 指定默认的 HF cache 目录, gpt2_cache_docker 可以被挂载
# ENV TRANSFORMERS_CACHE /app/gpt2_cache_docker
# ENV HF_HOME /app/gpt2_cache_docker


# 暴露端口 (如果你的应用是网络服务)
# EXPOSE 8000

# 默认启动命令 (可以被 docker run 覆盖)
# CMD ["python", "src/train.py"]
# 或者如果你使用 Hydra，可能没有一个单一的CMD，而是通过 docker run 指定
# ENTRYPOINT ["python", "src/train.py"] 