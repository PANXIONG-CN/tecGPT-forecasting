#!/bin/bash

# 确保移除大文件
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch src/models/TEC-LLM/gpt2/pytorch_model.bin" --prune-empty --tag-name-filter cat -- --all

# 强制推送到GitHub
git push -f origin main

echo "如果上面命令因网络问题失败，可以尝试："
echo "1. 使用代理: git config --global http.proxy 'http://your-proxy:port'"
echo "2. 关闭代理: git config --global --unset http.proxy"
echo "3. 禁用SSL验证(不安全，仅用于测试): git config --global http.sslVerify false" 