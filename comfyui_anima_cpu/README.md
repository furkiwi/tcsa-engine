# TCSA Anima CPU v2
針對能在本地使用 Anima Turbo GGUF 出圖的環境，實驗「畫師條件是否能跨提示詞重用」。
抽取與 adapter 投影在 CPU 執行，出圖沿用你的 CPU/GPU、GGUF Loader、sampler、步數、CFG。
輸出是自訂條件 profile，不是 LoRA。畫風有效性必須用真實圖片確認。

## 安裝（Windows，git clone 的 ComfyUI）
在 ComfyUI 根目錄，使用**啟動 ComfyUI 的同一個 Python**執行：

```powershell
git clone -b rework/anima-cpu-v2 https://github.com/furkiwi/tcsa-engine.git custom_nodes/tcsa-anima-cpu
python -m pip install -r custom_nodes/tcsa-anima-cpu/comfyui_anima_cpu/requirements.txt
```

重啟 ComfyUI。若使用 venv，先啟動原本的 venv。
ZIP 安裝則解壓 tcsa-anima-cpu 資料夾到 custom_nodes，安裝資料夾內的 requirements.txt。
Git 與 ZIP 擇一，避免同名節點重複載入。舊 anima_artist_demo 可移出 custom_nodes 避免混淆。
不要為本實驗另外更換 PyTorch/CUDA。依賴只有 numpy、safetensors、gguf（GGUF 插件通常已有後者）。
需要支援 Anima 的 ComfyUI，以及 PyTorch 的 load_state_dict(assign=True) 功能。

## 1. 設定抽取
保留目前能成功出圖的工作流。
- 使用你原本的 Anima CLIP Loader，設定 device=cpu。原生 CLIP Loader 可選 cpu。若目前 GGUF CLIP Loader 沒有 CPU 選項，可使用原版 qwen_3_06b_base.safetensors 搭配原生 loader，沿用能產生 Anima t5xxl_ids 的 type。抽取與對照必須使用同一個 CLIP。
- 新增 **TCSA Anima Adapter (CPU)**，選取與出圖 UNet Loader **完全相同的 Anima Turbo GGUF 檔案**。它只解量化 llm_adapter 張量為 float32，不載入其他去噪層。原始 GGUF 會以 mmap 開啟，adapter 本身仍佔用 RAM。
- 新增 **TCSA Extract Artist**，接上 adapter 與 CPU CLIP。
- artist 換成已知有效的 @繪師名；scenes 每行一個簡單英文場景，預設 3 個。限制 2–32 個不同場景。
- name 使用英數、底線或連字號。執行後存於 ComfyUI/output/tcsa_anima/<name>.safetensors。
- 同名預設拒絕覆寫，確定要重抽時才開啟 overwrite。

可載入 examples/conditioning_nodes.json 作為節點接線模板。這份模板只有條件處理節點，需接入你現有的 CPU CLIP 和 KSampler；沒有替你猜測模型檔名或取樣參數。

## 2. 接到已能出圖的工作流
新增 **TCSA Load Profile**，name 填剛才的名稱。抽取完成後，用它取代 Extract 的輸出，避免每次重新編碼抽取場景。
新增 **TCSA Compare Artist**，接上 adapter、同一個 CPU CLIP、profile。
scene 填**沒參與抽取**的新場景，不加 @繪師、不加括號加權。
positive 輸出接到原 KSampler 的 positive。negative、latent、VAE、sampler、steps、CFG 全部沿用。
固定 seed，將 control_after_generate 設成 fixed。每次只切換 mode。

| mode | 用途 |
|---|---|
| native_baseline | 原生「無畫師」條件，走正常模型 adapter |
| cpu_roundtrip | CPU adapter 預先投影的無畫師條件，確認轉換接線與精度 |
| native_artist | 原生「@畫師, 場景」，作為目標圖 |
| extracted | 抽取 profile 套用到新場景 |
| zero_strength | 強制強度 0，應與 cpu_roundtrip 條件逐值相等 |

先比較 native_baseline 與 cpu_roundtrip。兩者可能因 CPU float32 與原模型計算精度而有差異；若差異明顯，先停下檢查，不能把變化歸因於畫風。
確認 native_artist 有你要的風格，再看 extracted 是否保留主體且重現線條/配色。
zero_strength 與 cpu_roundtrip 的條件逐值相同；圖片若不同，檢查 seed、取樣非確定性及其他工作流節點。
strength=1 代表完整前綴移植。0–1 是補齊至 512 位置後的整段條件線性插值，不是原生 prompt 權重語意。建議先測 0 與 1。

profile 保存 adapter 權重 SHA-256、畫師、場景、token IDs 與文字編碼器 probe。
adapter 不一致、probe 不一致、測試場景出現在抽取清單、token 邊界不同都會報錯。
probe 是相容性檢查，不是整個文字編碼器的密碼學身分證。請自行確保出圖模型與 Adapter Loader 選相同檔案。
舊版 .npy 不含這些資料，請重新抽取。

## 3. 自動依序出 5 張對照（選用）
在現有工作流接好 Compare + Load Profile + 單一 KSampler/KSamplerAdvanced + SaveImage，先確定能出一張圖。
匯出 **API 格式 JSON**（不是一般工作流 JSON）。
在 ComfyUI 執行期間：

```powershell
python custom_nodes/tcsa-anima-cpu/comfyui_anima_cpu/compare_api.py workflow_api.json --seed 12345 --out compare_12345
```

ZIP 安裝的腳本位於 custom_nodes/tcsa-anima-cpu/compare_api.py。
腳本用同一個 seed，依序提交上述 5 種模式，保留原本取樣參數，一次只排一張，下載圖片並保存 manifest.json。
每張最長等待 2 小時，可用 --timeout 調整。逾時後工作可能仍在 ComfyUI 中，請檢查 queue 再重跑。
腳本要求單一 sampler，positive 直接連 Compare，且用 Load Profile；複雜多階段工作流請手動比較。
換另一個 seed 重複一次即可形成第一輪驗證。

## 方法與限制
在不同場景前面加入同一個「@artist, 」前綴，取得完整 adapter 輸出。
只有 T5 序列後綴與原場景逐 token 相同時才接受，以免錯取內容或 EOS。
保留每個前綴 token 的向量，跨場景平均。套用時將它們接到新場景已投影條件之前，再補齊到 512 個位置。
這是研究用假設。畫師造成的**內容 token 改變**並沒有被移植，新增前綴也可能本身影響構圖。
報告中的特徵 L2 距離不能判定畫風成功，不能作為 GO 門檻。
已投影條件會移除 T5 IDs 和 Qwen mask，避免二次 adapter 投影。
單一提示詞、未加權、最多 512 Qwen/T5 tokens；不支援已套 CLIP LoRA 或多段 conditioning。
CPU 抽取不需要全域 --cpu；出圖可以保留原本的 --lowvram 等啟動方式。
adapter 在 ComfyUI 節點快取中可能持續佔用 RAM。記憶體不足時可先減少抽取場景、關閉其他工作流，必要時重啟 ComfyUI。

## 驗證範圍
測試涵蓋真實 safetensors/GGUF 檔案讀寫、Q4_0 解量化、Q3_K 零值區塊、
精確零強度、前綴 token 邊界、覆寫與鎖、模型/編碼器不一致拒絕、節點 metadata、
以及批次比較保持 seed/取樣參數。
另以當時 ComfyUI 上游 LLMAdapter 程式及小型隨機權重測試 CPU forward / strict meta loading。
沒有使用你的完整 Turbo GGUF 或真實 Qwen 權重跑端到端，沒有宣稱已抽出有效畫風。

```powershell
python -m unittest discover -s tests -v
```

## 實作依據
- https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/anima/model.py
- https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_base.py
- https://github.com/city96/ComfyUI-GGUF/blob/main/loader.py
