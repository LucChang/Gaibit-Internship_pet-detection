#!/bin/bash

# ================= 設定區 =================
NAS_USER="happyluc1114"
NAS_PASSWORD="Duposhiny0825"
NAS_IP="100.103.4.107"

# 遠端 NAS 錄影目錄基礎路徑 (每日會建立 yyyy-mm-dd 資料夾)
REMOTE_BASE_PATH="/volume1/docker/mediamtx/recordings/mixcat01"

# 本地儲存資料夾：video_test/{監視器名稱}
LOCAL_BASE_DIR="video_test"
CCTV_NAME="CCTV_1"
LOCAL_PATH="${LOCAL_BASE_DIR}/${CCTV_NAME}"

# 每個監視器保留的最大影片數量 (超過 3 部則自動刪除最早的影片)
MAX_VIDEOS=3

# 每隔幾秒執行一次同步 (預設 60 秒 = 1 分鐘)
SYNC_INTERVAL=60

# 本地日誌檔路徑
LOG_FILE="video_download.log"
# ==========================================

mkdir -p "$LOCAL_PATH"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

trap 'log "🛑 影片同步服務已停止。"; exit 0' SIGINT SIGTERM

cleanup_old_videos() {
    local target_dir="$1"
    local max_keep="$2"

    if [ ! -d "$target_dir" ]; then
        return
    fi

    local files=()
    while IFS= read -r file; do
        [ -f "$file" ] && files+=("$file")
    done < <(find "$target_dir" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.ts" -o -iname "*.avi" -o -iname "*.mov" \) -exec ls -1tr {} + 2>/dev/null)

    local total_count=${#files[@]}
    if [ "$total_count" -gt "$max_keep" ]; then
        local num_to_delete=$((total_count - max_keep))
        log "📁 [${target_dir}] 目前影片數量 (${total_count}) 超過上限 (${max_keep})，清理最早的 ${num_to_delete} 部影片..."
        
        for ((i = 0; i < num_to_delete; i++)); do
            local file_to_delete="${files[i]}"
            if [ -f "$file_to_delete" ]; then
                local filename
                filename=$(basename "$file_to_delete")
                rm -f "$file_to_delete"
                log "🗑️  已刪除最早影片: ${filename}"
            fi
        done
    else
        log "📁 [${target_dir}] 目前影片數量: ${total_count}/${max_keep}，無需刪除。"
    fi
}

sync_once() {
    local today
    today=$(date +%Y-%m-%d)
    local remote_dir="${REMOTE_BASE_PATH}/${today}"

    log "--------------------------------------------------------"
    log "🔍 檢查 NAS 當日目錄: ${remote_dir}"

    local ssh_cmd="ssh"
    if command -v sshpass >/dev/null 2>&1; then
        ssh_cmd="sshpass -p $NAS_PASSWORD ssh"
    fi

    # 取得遠端目錄下檔案數量
    local remote_count
    remote_count=$($ssh_cmd -o StrictHostKeyChecking=no "$NAS_USER@$NAS_IP" "ls -1 '$remote_dir' 2>/dev/null | wc -l" 2>/dev/null)
    if [ -n "$remote_count" ]; then
        log "📊 遠端目錄檔案統計: 共有 ${remote_count} 個檔案"
    fi

    # 取得遠端最新的一部影片名稱
    local latest_video
    latest_video=$($ssh_cmd -o StrictHostKeyChecking=no "$NAS_USER@$NAS_IP" "ls -1tr '$remote_dir'/*.mp4 '$remote_dir'/*.mkv '$remote_dir'/*.ts 2>/dev/null | tail -n 1" 2>/dev/null)

    if [ -n "$latest_video" ]; then
        log "🎬 遠端最新影片: $(basename "$latest_video")"
        log "⬇️  下載最新影片至 ${LOCAL_PATH}/ ..."
        
        if command -v sshpass >/dev/null 2>&1; then
            sshpass -p "$NAS_PASSWORD" scp -o StrictHostKeyChecking=no "$NAS_USER@$NAS_IP:$latest_video" "$LOCAL_PATH/" >> "$LOG_FILE" 2>&1
        else
            scp -o StrictHostKeyChecking=no "$NAS_USER@$NAS_IP:$latest_video" "$LOCAL_PATH/" >> "$LOG_FILE" 2>&1
        fi
    else
        log "ℹ️  遠端目錄目前尚無影片。"
    fi

    cleanup_old_videos "$LOCAL_PATH" "$MAX_VIDEOS"
}

log "========================================================"
log "🎥 監視器影片每分鐘自動同步與滾動清理服務已啟動"
log "🌐 NAS 位置: ${NAS_USER}@${NAS_IP}"
log "📂 本地儲存: ${LOCAL_PATH}"
log "⏱️  同步間隔: 每 ${SYNC_INTERVAL} 秒 (1 分鐘)"
log "📦 保留數量: 最新 ${MAX_VIDEOS} 部影片"
log "========================================================"

while true; do
    sync_once
    log "⏳ 等待 ${SYNC_INTERVAL} 秒後進行下一次檢查與同步..."
    sleep "$SYNC_INTERVAL"
done