@echo off
chcp 65001

:: 网络入侵检测系统启动脚本
:: 作者: AI Assistant
:: 日期: 2026-04-10

echo ========================================
echo 基于gcForest的网络入侵检测系统
echo ========================================

echo 正在激活虚拟环境...
call .\venv\Scripts\activate.bat

echo 正在启动Streamlit演示系统...
echo 请稍候，系统正在加载模型和数据...
echo 
echo 启动完成后，在浏览器中访问:
echo http://localhost:8501
echo 

:: 使用虚拟环境中的Python执行
.\venv\Scripts\python.exe -m streamlit run program\streamlit_demo.py

pause