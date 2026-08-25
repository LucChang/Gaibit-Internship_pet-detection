#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
監視器影片自動同步與滾動清理程式
- 連線至 NAS (保持 SSH 登入連線)
- 每次循環自動檢查並列出當日目錄所有檔案數量
- 下載遠端最新錄製影片至 video_test/CCTV_1
- 自動維護本地影片上限：超過 3 部自動刪除最舊版本
"""

import datetime
import os
import sys
import time
import paramiko
from scp import SCPClient

# 設定 Windows 控制台編碼防呆 (防止 CP950 編碼錯誤)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ================= 設定區 =================
NAS_USER = "happyluc1114"
NAS_IP = "100.103.4.107"
NAS_PASSWORD = "Duposhiny0825"

# 本地儲存基礎資料夾
LOCAL_BASE_DIR = "video_test"

# 多監視器頻道設定 (CCTV_1 & CCTV_2)
CCTV_CONFIGS = [
    {
        "name": "CCTV_1",
        "remote_base_path": "/volume1/docker/mediamtx/recordings/mixcat01",
        "local_path": os.path.join(LOCAL_BASE_DIR, "CCTV_1")
    },
    {
        "name": "CCTV_2",
        "remote_base_path": "/volume1/docker/mediamtx/recordings/mixcat02",
        "local_path": os.path.join(LOCAL_BASE_DIR, "CCTV_2")
    }
]

# 每個監視器保留的最大影片數量 (超過 3 部則自動刪除最早的影片)
MAX_VIDEOS = 3

# 每隔幾秒執行一次同步 (預設 60 秒 = 1 分鐘)
SYNC_INTERVAL = 60

# 本地日誌檔路徑
LOG_FILE = "video_download.log"
# ==========================================

# 支援的影片副檔名
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".ts", ".avi", ".mov", ".flv"}


def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    try:
        print(full_msg, flush=True)
    except UnicodeEncodeError:
        # CP950 相容 fallback
        safe_msg = full_msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
        print(safe_msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")
    except Exception:
        pass


class NASSSHClient:
    """保持 SSH 長連接的客戶端管理器"""

    def __init__(self, host, user, password):
        self.host = host
        self.user = user
        self.password = password
        self.ssh = None

    def connect(self):
        """建立 SSH 連線並啟用 Keep-Alive 保活"""
        try:
            if self.ssh is not None:
                self.close()

            log(f"[連線] 正在連接 NAS (SSH): {self.user}@{self.host} ...")
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=self.host,
                username=self.user,
                password=self.password,
                timeout=15,
                banner_timeout=30,
                look_for_keys=False,
                allow_agent=False
            )
            # 設定 30 秒心跳封包，保持連線持續存活
            transport = ssh.get_transport()
            if transport is not None:
                transport.set_keepalive(30)

            self.ssh = ssh
            log("[成功] SSH 連線成功，保持登入狀態中！")
            return True
        except Exception as e:
            log(f"[錯誤] 連接 NAS 失敗: {e}")
            self.close()
            return False

    def is_active(self):
        """檢查連線是否依然有效"""
        if self.ssh is None:
            return False
        transport = self.ssh.get_transport()
        return transport is not None and transport.is_active()

    def ensure_connected(self):
        """確保連線可用，若斷線則自動重連"""
        if not self.is_active():
            log("[重連] 偵測到連線中斷或尚未建立，重新建立連線中...")
            return self.connect()
        return True

    def exec_command(self, cmd):
        """執行遠端 Shell 指令並返回 stdout, stderr"""
        if not self.ensure_connected():
            return None, "Connection failed"
        try:
            stdin, stdout, stderr = self.ssh.exec_command(cmd, timeout=30)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            return out, err
        except Exception as e:
            log(f"[警告] 執行遠端指令異常: {e}")
            self.close()
            return None, str(e)

    def download_file(self, remote_file_path, local_target_path):
        """透過 SCP 下載遠端檔案"""
        if not self.ensure_connected():
            return False
        try:
            temp_file = local_target_path + ".tmp"
            with SCPClient(self.ssh.get_transport()) as scp:
                scp.get(remote_file_path, temp_file)
            if os.path.exists(local_target_path):
                os.remove(local_target_path)
            os.rename(temp_file, local_target_path)
            return True
        except Exception as e:
            log(f"[錯誤] SCP 下載檔案失敗 ({remote_file_path}): {e}")
            if os.path.exists(local_target_path + ".tmp"):
                try:
                    os.remove(local_target_path + ".tmp")
                except Exception:
                    pass
            return False

    def close(self):
        """安全關閉連線"""
        if self.ssh:
            try:
                self.ssh.close()
            except Exception:
                pass
            self.ssh = None


def cleanup_old_videos(target_dir: str, max_keep: int = 3):
    """檢查 target_dir 內的影片數量，若超過 max_keep 則刪除修改時間最早的影片"""
    if not os.path.exists(target_dir):
        return

    video_files = []
    for fname in os.listdir(target_dir):
        fpath = os.path.join(target_dir, fname)
        if os.path.isfile(fpath):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                video_files.append(fpath)

    # 依照最後修改時間由舊到新排序 (oldest first)
    video_files.sort(key=lambda p: os.path.getmtime(p))

    total_count = len(video_files)
    if total_count > max_keep:
        num_to_delete = total_count - max_keep
        log(f"[清理] 本地目錄 [{target_dir}] 目前影片數量 ({total_count}) 超過上限 ({max_keep})，清理最早的 {num_to_delete} 部影片...")
        for i in range(num_to_delete):
            file_to_del = video_files[i]
            try:
                os.remove(file_to_del)
                log(f"[刪除] 已刪除歷史舊影片: {os.path.basename(file_to_del)}")
            except Exception as e:
                log(f"[錯誤] 刪除檔案失敗 {os.path.basename(file_to_del)}: {e}")
    else:
        log(f"[統計] 本地目錄 [{target_dir}] 目前影片數量: {total_count}/{max_keep}，無需刪除。")


def sync_camera(client: NASSSHClient, cfg: dict):
    """針對單一監視器頻道的同步與下載邏輯"""
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    remote_base = cfg["remote_base_path"]
    local_path = cfg["local_path"]
    cam_name = cfg["name"]

    remote_dir = f"{remote_base}/{today_str}"
    os.makedirs(local_path, exist_ok=True)

    log("--------------------------------------------------------")
    log(f"[{cam_name}] 檢查 NAS 當日目錄: {remote_dir}")

    if not client.ensure_connected():
        log(f"[{cam_name}] 無法連線至 NAS，將於下個循環重試。")
        return

    # 1. 取得遠端當日目錄下檔案總數
    count_cmd = f"ls -1 '{remote_dir}' 2>/dev/null | wc -l"
    out_count, _ = client.exec_command(count_cmd)
    if out_count is not None:
        file_count = out_count.strip()
        log(f"[{cam_name}] 遠端目錄 [{remote_dir}] 目前共有 {file_count} 個檔案")
    else:
        log(f"[{cam_name}] 無法取得遠端檔案數量。")
        return

    # 2. 找出遠端最新的影片檔案 (依時間排序最後一個)
    latest_cmd = f"ls -1tr '{remote_dir}'/*.mp4 '{remote_dir}'/*.mkv '{remote_dir}'/*.ts 2>/dev/null | tail -n 1"
    out_latest, _ = client.exec_command(latest_cmd)
    
    if not out_latest or not out_latest.strip():
        log(f"[{cam_name}] 遠端資料夾目前尚無影片檔案。")
        return

    remote_latest_file = out_latest.strip()
    latest_filename = os.path.basename(remote_latest_file)
    log(f"[{cam_name}] 遠端最新影片: {latest_filename}")

    # 3. 檢查本地是否已經存在
    local_target_file = os.path.join(local_path, latest_filename)
    need_download = True

    if os.path.exists(local_target_file):
        size_cmd = f"wc -c < '{remote_latest_file}' 2>/dev/null"
        out_size, _ = client.exec_command(size_cmd)
        if out_size and out_size.strip().isdigit():
            remote_size = int(out_size.strip())
            local_size = os.path.getsize(local_target_file)
            if local_size == remote_size and local_size > 0:
                log(f"[{cam_name}] 最新影片 [{latest_filename}] 本地已存在且完整，無需重複下載。")
                need_download = False

    if need_download:
        log(f"[{cam_name}] 開始下載最新影片: {latest_filename} -> {local_path}/")
        success = client.download_file(remote_latest_file, local_target_file)
        if success:
            log(f"[{cam_name}] 下載完成: {latest_filename}")
        else:
            log(f"[{cam_name}] 下載失敗: {latest_filename}")

    # 4. 檢查本地影片數量，超過上限 (3 部) 則自動刪除最早的
    cleanup_old_videos(local_path, MAX_VIDEOS)


def sync_cycle(client: NASSSHClient):
    """每次循環檢查所有 CCTV 頻道"""
    for cfg in CCTV_CONFIGS:
        try:
            sync_camera(client, cfg)
        except Exception as e:
            log(f"[{cfg['name']}] 同步過程發生異常: {e}")


def main():
    for cfg in CCTV_CONFIGS:
        os.makedirs(cfg["local_path"], exist_ok=True)
        
    log("========================================================")
    log("多頻道監視器影片自動同步與滾動清理服務已啟動")
    log(f"NAS 位置: {NAS_USER}@{NAS_IP}")
    for cfg in CCTV_CONFIGS:
        log(f"  - 頻道 {cfg['name']}: {cfg['remote_base_path']} -> {cfg['local_path']}")
    log(f"同步頻率: 每 {SYNC_INTERVAL} 秒 (1 分鐘)")
    log(f"影片上限: 每個頻道最多保留最新 {MAX_VIDEOS} 部影片")
    log("========================================================")

    client = NASSSHClient(NAS_IP, NAS_USER, NAS_PASSWORD)
    client.connect()

    try:
        while True:
            sync_cycle(client)
            log(f"等待 {SYNC_INTERVAL} 秒後進行下一次檢查與同步...")
            time.sleep(SYNC_INTERVAL)
    except KeyboardInterrupt:
        log("收到中斷信號 (Ctrl+C)，關閉 NAS 連線並退出服務。")
    finally:
        client.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
