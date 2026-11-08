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
- artist 可填單一 @繪師、多個逗號分隔的畫師，或原生加權提示。例如：

```text
(@sunapua:0.8), (@waonaolmao:0.9), (@oka \(bananashoe\):0.7)
```

畫師名稱的字面括號要寫成 `\(`、`\)`，外層權重括號保持不跳脫。
沿用 ComfyUI 原生 tokenizer；Qwen 路徑按 Anima 規則保持 token 權重 1，T5 權重於 adapter 輸出後套用。
native_artist 對照會使用保存的完整加權提示，extracted 的前綴已包含權重，不會再乘一次。
權重是抽取時固定的配方。更改各畫師權重後請重新抽取，建議另存 name（例如 kiwi_mix）。
Compare 的 strength 控制整份 profile 的套用強度，和個別畫師權重用途不同。
原有單一畫師 profile 可繼續載入；新的混合 profile 使用相同 v2 格式。

- scenes 每行一個簡單、未加權的英文場景，預設 3 個。限制 2–32 個不同場景。
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
在不同場景前面加入同一個畫師提示前綴（可以包含多位畫師與各自權重），取得完整 adapter 輸出。
只有 T5 序列後綴與原場景逐 token 相同時才接受，以免錯取內容或 EOS。
保留每個前綴 token 的向量，跨場景平均。套用時將它們接到新場景已投影條件之前，再補齊到 512 個位置。
這是研究用假設。畫師造成的**內容 token 改變**並沒有被移植，新增前綴也可能本身影響構圖。
報告中的特徵 L2 距離不能判定畫風成功，不能作為 GO 門檻。
已投影條件會移除 T5 IDs 和 Qwen mask，避免二次 adapter 投影。
scene 保持單一未加權提示詞；artist 支援多位畫師與原生括號權重。整段最多 512 Qwen/T5 tokens；不支援已套 CLIP LoRA 或多段 conditioning。
CPU 抽取不需要全域 --cpu；出圖可以保留原本的 --lowvram 等啟動方式。
adapter 在 ComfyUI 節點快取中可能持續佔用 RAM。記憶體不足時可先減少抽取場景、關閉其他工作流，必要時重啟 ComfyUI。

## 驗證範圍
測試涵蓋真實 safetensors/GGUF 檔案讀寫、Q4_0 解量化、Q3_K 零值區塊、
精確零強度、前綴 token 邊界、覆寫與鎖、模型/編碼器不一致拒絕、節點 metadata、
以及批次比較保持 seed/取樣參數。
新增多畫師加權的公開節點流程測試，包含抽取、存檔、重新載入、原生對照、避免重複加權、零強度與非有限權重拒絕。
另以當時 ComfyUI 上游 LLMAdapter 程式及小型隨機權重測試 CPU forward / strict meta loading。
沒有使用你的完整 Turbo GGUF 或真實 Qwen 權重跑端到端，沒有宣稱已抽出有效畫風。

```powershell
python -m unittest discover -s tests -v
```

## 實作依據
- https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/anima/model.py
- https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_base.py
- https://github.com/city96/ComfyUI-GGUF/blob/main/loader.py

## 4. 將成功的 profile 蒸餾為標準 LoRA（CPU 實驗版）
新增 **TCSA Train LoRA (CPU)**，接入三個輸入：
- Adapter Loader → adapter（與成功出圖相同的 GGUF）
- CPU CLIP Loader → clip
- Load Profile → profile（例如已確認有效的 kiwi_mix）

先使用 rank=32、steps=300、learning_rate=0.001、seed=42。
train_scenes 預設 8 個場景，可改成與用途相關的 4–64 行。validation_scenes 為 2–16 個不同場景，不能和訓練或原 profile 抽取場景重複。
節點使用已保存的 profile 及其權重配方作為教師，不必重新填 artist。
name 可填 kiwi_mix_lora。按執行後先快取各場景，再執行 CPU 訓練。主控台顯示 cache / train 進度，ComfyUI 可正常中止。

輸出位置：

```text
ComfyUI/output/tcsa_anima_lora/kiwi_mix_lora.safetensors
```

將檔案複製到 ComfyUI/models/loras/，重新整理模型清單。
使用 **Load LoRA (Model Only)** 或一般 **Load LoRA**：

| 連接/參數 | 設定 |
|---|---|
| GGUF UNet Loader 的 MODEL | 接 LoRA Loader 的 model |
| LoRA Loader 的 MODEL 輸出 | 接原有模型取樣設定節點/取樣器，維持原工作流順序 |
| strength_model | 先設 1.0 |
| strength_clip（若有） | 0.0 |
| 正向條件 | 一般 CLIP Text Encode，輸入新場景，不寫畫師 |
| 負向條件 | 沿用原本工作流 |

**LoRA 測試不要將 TCSA Compare 的已投影條件接到 positive。** 讓原生 Anima adapter 執行，LoRA 才會作用到這一層。
請使用與訓練相同的 GGUF，未驗證跨 checkpoint/量化版本可移植性。
先固定 seed 比較「無 LoRA」「原抽取 profile」「LoRA 1.0」，且不要同時套用 profile 和 LoRA。
若效果過強再比較 0.5；若缺乏風格，記下結果和 validation loss，不要只用提高強度判定成功。

### 訓練的實際範圍
只訓練 llm_adapter.out_proj 的低秩矩陣，Qwen、其餘 adapter、DiT 都不訓練。
先快取最後投影層的輸入/輸出，之後每步只需算低秩增量、RMSNorm 與條件空間的注意力回應。
教師是成功 profile + 新場景條件，包含完整畫師前綴。學生輸出維持無畫師提示詞的長度。
目標是在固定的風格/內容/隨機查詢向量下，學生的注意力回應接近教師，另加入 0.05 倍內容漂移限制。
這些查詢**不是從實際 DiT 抽出的 queries**，此 loss 只是 CPU 可行的替代目標。它不等同於去噪蒸餾或圖片品質指標。
保留空提示詞的回應作為訓練錨點，但沒有保證所有負向提示詞皆完全不變。
每 20 步在驗證場景評估並保留最低 loss 權重。若完全沒有改善則不匯出。
LoRA 無法增加原本不存在的畫師 token，profile 成功不保證本次 LoRA 蒸餾成功，必須重新出圖檢查。

輸出使用 ComfyUI 的 diffusion_model.llm_adapter.out_proj 鍵名，lora_up/down.weight 與 alpha=rank，float32。
rank32 約 65,536 個可訓練參數，權重約 256 KiB 加上 metadata。檔案小是因為只修改這一個投影層。
目前已驗證合成資料訓練、推論模式內啟用梯度、驗證選模、取消、匯出權重重建與 ComfyUI LoRA 解析。
真實 profile 的訓練時間、畫風保留程度和 K620/GGUF 端到端載入仍須在本地實測。

## 5. LoRA 幾乎無作用時的載入檢查
新增 **TCSA LoRA Check (CPU)**，接入原本的 CPU adapter 與 CPU CLIP，選取放在 models/loras 下的 TCSA LoRA。
scene 使用與出圖完全相同的無畫師提示，strength=1。此節點可單獨執行，主控台會顯示 JSON 報告。
它在 CPU adapter 的 out_proj 明確加入 LoRA 增量，執行後移除 hook，不更動原始模型權重。

報告的 conditioning_relative_l2 是投影與 RMSNorm 後的相對條件變化，conditioning_max_abs_change 是最大絕對變化。
weight_delta_relative_l2 比較增量與原權重大小。這些數字不是畫風成功門檻，也不能單獨證明 GGUF Loader 已套用。

要排除 GGUF LoRA 套用路徑，使用 **CFG=1** 的簡單對照：
1. KSampler 的 model 接回未套 LoRA 的 GGUF 模型。
2. positive 先接本節點 baseline，再接 lora_applied，各出一張，固定 seed 與其他設定。
3. 若明確套用後有目標畫風、正常 Loader 沒有，優先排查 GGUF runtime patching。
4. 若明確套用仍幾乎沒畫風，表示這份權重未能重現 profile 的視覺效果，應修改蒸餾方法。

這個診斷節點只處理傳入提示詞。CFG>1 時還有負向分支的 LoRA 效果，不能把只改正向的對照視為完整等價。
不要同時讓 sampler model 套用同一份 LoRA，以免混淆。
現行訓練的替代目標 loss 下降不足以證明畫風轉換成功；請先保留成功 profile 作為教師。


## 單步去噪蒸餾實驗（新增）

若 CPU 直接套用舊 LoRA 也沒有畫風，可改測真實 Anima 去噪輸出監督。
先看 [DENOISE_EXPERIMENT.md](DENOISE_EXPERIMENT.md)，依序執行老師快取、一次反向傳播探測、10 次短訓練。
新增節點 `TCSA Denoise Teacher Cache` / `TCSA Denoise Train (Experimental)`，
附有 `examples/denoise_teacher_workflow.json` 和 `examples/denoise_probe_workflow.json`。
這仍是未經完整 Anima/K620 驗證的實驗；probe 通過和 loss 下降都不等於畫風成功。
