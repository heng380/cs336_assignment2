#!/bin/bash

# 设置工作目录（根据你实际的路径修改）
cd /home/aiscuser/repos/assignment2-systems

# 日志文件路径
LOG_FILE="gpu_run_monitor.log"

# 记录日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查并重启进程函数
restart_if_not_running() {
    # 查找是否有匹配的进程
    PROCESS_COUNT=$(ps aux | grep "gpu_run.py" | grep -v "grep" | wc -l)

    if [ "$PROCESS_COUNT" -eq 0 ]; then
        log "检测到进程未运行，正在重启..."
        nohup python gpu_run.py > output.log 2>&1 &
        log "已重启任务，PID: $!"
    else
        log "任务正在运行，无需操作。"
    fi
}

# 主循环，每小时执行一次
while true; do
    restart_if_not_running
    sleep 3600  # 等待一小时
done