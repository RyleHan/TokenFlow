#!/bin/bash
# 激活TokenFlow项目的虚拟环境

echo "激活 TokenFlow 项目环境..."
source tokenflow_env/bin/activate

echo "TokenFlow 环境已激活!"
echo "Python 路径: $(which python)"
echo "已安装的包:"
pip list --format=columns