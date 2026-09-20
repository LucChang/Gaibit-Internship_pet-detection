# CCTV 智慧寵物定位與 ROI 接觸停留監控系統

基於 **YOLO 動態追蹤** 與 **Microsoft Florence-2 大型視覺語言模型** 的智慧寵物行為監控與 ROI（感興趣區域）接觸分析 Web 系統。支援即時多目標追蹤、自動靜態設施辨識、自訂 ROI 畫布標註、停留時間分析（> 3 秒有效過濾與去重）、事件影片自動剪輯（前後 10 秒）以及 NAS 遠端錄影自動同步。

---

## 系統特色

- **雙模型協同架構 (Dual-Model Architecture)**
  - **動態目標即時追蹤**：採用 **YOLO + StrongSORT / BoT-SORT** 搭配自適應平滑器（`TrackIDSmoother`），提供高 FPS、抗遮蔽、防止 ID 切換的穩定貓狗偵測與追蹤。
  - **靜態設施高精定位**：整合 **`microsoft/Florence-2-large`** 視覺語言模型，每 5 秒非同步偵測畫面中的貓砂盆（`litter_box`）與飼料/水碗（`bowl`），無縫結合動態與靜態物體分析。
- **精準停留判定與防重複機制**
  - **停留門檻過濾**：貓咪進入 ROI 或設施標記框需持續停留 **超過 3 秒** 才正式記錄事件，自動過濾路過或瞬時誤觸。
  - **連續停留去重**：貓咪持續停留在同一個標記框內時，系統僅保留單一事件紀錄並動態累加持續秒數，**絕不產生重複冗餘事件**。
  - **智慧片段剪輯**：貓咪離開標記框後，背景非同步截取該事件「**進入前 10 秒至離開後 10 秒**」的完整標註影片（儲存於 `static/clips/`），並提供線上即時播放與下載。
- **現代化互動 Web 介面 (Flask + Vanilla CSS / JS)**
  - **即時串流監控**：低延遲 MJPEG 串流，即時疊加追蹤框、Florence-2 設施框、接觸計時與 FPS 數據。
  - **互動式 ROI 標記**：直接在影片畫面上拖曳滑鼠繪製多邊形/矩形 ROI，支援自訂類別（水碗、飼料盆、貓砂盆、跳台等）與顏色，並支援按監視器頻道自動持久化記憶（`roi_config.json`）。
  - **事件管理與報表匯出**：即時事件日誌表格、搜尋與篩選、一鍵匯出完整 CSV 報表。
- **遠端 NAS 監視器錄影自動同步 (`video_download.py`)**
  - 透過 SSH / SCP 保持長連接，定時從遠端 Synology / QNAP NAS 錄影目錄同步最新錄影片段至本地 `video_test/`。
  - 滾動式儲存管理，自動刪除舊檔並維持本地保留最新 3 部影片。
  - 透過 `.env` 進行憑證與路徑安全配置。

---

## 專案程式架構

```
Objection_localization_roi_detect/
├── app.py                          # 主程式：Flask Web 伺服器、影像處理串流、YOLO+StrongSORT 追蹤與事件觸發
├── florence2_facility_detector.py   # Florence-2 大型模型設施偵測器 (GPU 加速、非同步執行序、日誌記錄)
├── video_download.py               # NAS 遠端監視器影片自動同步與滾動清理服務
├── .env.example                    # 環境變數設定範本 (NAS 帳密、CCTV 頻道路徑)
├── .gitignore                      # Git 版本控制忽略設定
│
├── static/                         # 前端靜態資源
│   ├── css/
│   │   └── style.css               # Web 介面排版與深色主題樣式
│   ├── js/
│   │   └── main.js                 # 畫布 ROI 拖曳拉框、API 輪詢與即時表格渲染邏輯
│   └── clips/                      # 自動生成的接觸事件 20 秒標註影片存放目錄 (git-ignored)
│
├── templates/
│   └── index.html                  # Web 監控儀表板主頁面
│
├── strongsort_tuned.yaml           # StrongSORT 追蹤器調校設定檔
├── bytetrack_tuned.yaml            # ByteTrack 追蹤器調校設定檔
└── weights/                        # YOLO 權重檔案放置目錄 (git-ignored)
```

---

## 系統環境需求與架設

### 1. 硬體與作業系統建議
- **作業系統**：Windows 10/11 或 Linux (Ubuntu 20.04+)
- **GPU**：建議具備 NVIDIA 獨立顯示卡 (支援 CUDA 11.8 或 12.x，VRAM $\ge 6\text{GB}$)
- **Python 版本**：Python 3.10 ~ 3.12 (或相容環境)

### 2. 安裝 PyTorch (GPU 加速版本)
請依據您的 CUDA 版本安裝對應之 PyTorch，例如（CUDA 12.6）：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3. 安裝相關依賴套件
```bash
pip install ultralytics transformers accelerate timm einops flask opencv-python numpy paramiko scp python-dotenv
```

### 4. 設定環境變數 (`.env`)
複製範本檔建立 `.env` 並填入您的 NAS 連線資訊與監視器頻道路徑：
```bash
cp .env.example .env
```
編輯 `.env` 內容：
```env
# NAS (SSH) 連線設定
NAS_USER=happyluc1114
NAS_IP=100.103.4.107
NAS_PORT=22
NAS_PASSWORD=YourPasswordHere

# 本地影片儲存基礎目錄
LOCAL_BASE_DIR=video_test

# 監視器頻道 1
CCTV_1_NAME=CCTV_1
CCTV_1_REMOTE_PATH=/volume1/docker/mediamtx/recordings/mixcat01

# 監視器頻道 2
CCTV_2_NAME=CCTV_2
CCTV_2_REMOTE_PATH=/volume1/docker/mediamtx/recordings/mixcat02

# 同步參數 (最多保留部數與檢查頻率秒數)
MAX_VIDEOS=3
SYNC_INTERVAL=60
LOG_FILE=video_download.log
```

---

## 啟動方式

### 方法一：一鍵啟動完整 Web 監控系統（推薦）
執行 `app.py`，系統將自動啟動 Web 服務並在背景帶起 NAS 影片同步服務：
```bash
python app.py
```
啟動成功後，終端機將顯示：
```text
============================================================
CCTV 物件定位與自訂 ROI 接觸事件紀錄 Web 系統 (StrongSORT 追蹤) 啟動中...
存取網址: http://127.0.0.1:5000
============================================================
```
開啟瀏覽器前往：`http://127.0.0.1:5000`

### 方法二：獨立執行 NAS 影片同步服務
若僅需測試或單獨運行 NAS 影片同步與滾動清理：
```bash
python video_download.py
```

---

## 使用操作指南

### 1. 選擇監視器頻道與影片
- 於畫面右上方的 **「監視器頻道」** 下拉選單切換頻道（如 `video_test/CCTV_1` 或 `video_test/CCTV_2`）。
- 串流畫面將自動載入該頻道下最新下載之錄影檔，並在影片結束時自動播放下一部。

### 2. 畫布自訂 ROI 標註
1. 在左側控制面板選擇或輸入 **ROI 類別名稱**（例如：`飼料盆 (Food Bowl)`、`貓砂盆 (Litter Box)` 或自訂類別）。
2. 選擇該標記框的顯示顏色。
3. 直接在中央 **即時影像畫面** 上按住滑鼠左鍵並拖曳拉框。
4. 放開滑鼠後即建立 ROI，右側「ROI 標記管理」面板會即時列出，可隨時微調座標、尺寸或刪除。
5. 系統會依據目前頻道自動將 ROI 記憶於 `roi_config.json`，重啟後不遺失。

### 3. Florence-2 自動設施標記
- 系統背景每 5 秒透過 `microsoft/Florence-2-large` 自動偵測環境中的貓砂盆與碗盤，並以專屬標籤 `[Florence-2]` 呈現在畫面上。

### 4. 停留事件紀錄與片段回放
- 當貓咪進入任一 ROI 或 Florence-2 標記框且停留 **超過 3 秒** 時，下方「觸發事件紀錄」表格將即時新增一筆紀錄。
- 貓咪離開後，系統自動生成 **前後 10 秒（共約 20 秒）** 的標註片段。
- 點擊表格中的 **「觀看 20s 片段」** 按鈕即可於跳出視窗中線上播放或另存影片檔。
- 點擊 **「匯出 CSV」** 按鈕可將所有偵測紀錄下載為試算表。

---

## 主要 API 介面一覽

| 請求路徑 | 方法 | 說明 |
| :--- | :--- | :--- |
| `/video_feed` | `GET` | MJPEG 即時影像串流 |
| `/api/folders` | `GET` | 取得現有監視器頻道資料夾列表 |
| `/api/select_folder` | `POST` | 切換至指定監視器頻道 |
| `/api/rois` | `GET / POST` | 取得或儲存目前頻道的 ROI 標記框設定 |
| `/api/rois/<roi_id>` | `DELETE` | 刪除指定的 ROI 標記框 |
| `/api/events` | `GET` | 取得即時觸發與停留事件列表 |
| `/api/clear_events` | `POST` | 清空所有歷史事件記錄 |
| `/api/export_csv` | `GET` | 匯出事件日誌為 CSV 檔案 |
| `/api/status` | `GET` | 取得系統狀態數據（FPS、接觸數、物件數等） |
