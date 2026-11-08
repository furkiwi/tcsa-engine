# Anima 單步去噪蒸餾：先驗證能否反向傳播

這是新實驗，不是已證實成功的畫風 LoRA 訓練器。現有提取檔不必重做。
只更新 CPU 文字適配器的 `llm_adapter.out_proj` LoRA，透過凍結的 Anima 去噪模型反向傳遞梯度；
不再使用人工構造的注意力 queries。固定同一個 LoRA 位置，先檢查換訓練目標的效果。
若此位置容量不足，換目標也可能仍學不到風格。

## 更新

在 PowerShell 執行（沿用目前的實驗分支）：

```powershell
git -C C:\ComfyUI\custom_nodes\tcsa-anima-cpu branch --show-current
git -C C:\ComfyUI\custom_nodes\tcsa-anima-cpu pull --ff-only
```

第一行應顯示 `rework/anima-cpu-v2`。若不是，先切換到這個分支；若 Git 因本機修改拒絕更新，不要使用 reset --hard。
重新啟動 ComfyUI。不需要更新 PyTorch、CUDA 或重裝 GGUF。

## A. 先製作老師快取

拖入 `examples/denoise_teacher_workflow.json`。
這份範例已依目前環境填入 `anima-turbo-v1.1-Q4_K_M.gguf`、`qwen_3_06b_base.safetensors`、
`qwen_image_vae.safetensors` 和提取檔名稱 `kiwi_mix`，檔名不同時用下拉選單改掉。

接線：

- 原始 `UnetLoaderGGUF` MODEL → `TCSA Denoise Teacher Cache.model`。
- `TCSA Anima Adapter (CPU)` → `adapter`，選同一個 Anima checkpoint。
- CPU `CLIPLoader` → `clip`。
- `TCSA Load Profile` → `profile`，名稱填成功的提取檔名稱（不含副檔名）。
- `teacher_latent` → VAE Decode → Save Image。

初始設定：

| 欄位 | 值 |
|---|---|
| scenes | `a furry girl reading a book`，先只用一行 |
| width / height | 256 / 384 |
| seed | 42，fixed |
| name | `kiwi_denoise_cache_01` |

此節點固定 Euler、sgm_uniform、4 步、CFG 1、strength 1。
不接負面提示、不接現有 LoRA、不接其他 MODEL patches。

一個場景會快取兩條軌跡：seed 42 用於訓練，seed 43 用於驗證，共 8 筆時間步資料。
每個時間步都計算老師和基準輸出，兩者輸入的 latent 完全相同，因此總計 16 次去噪前向，
不是只生成兩張圖的 8 次。多行場景依比例增加；第一輪最多 4 行。

老師使用成功提取條件，基準使用普通場景文字。每一步保存當下的 latent、sigma、老師去噪結果、
基準去噪結果和 CPU adapter 中間特徵。每次 Euler 更新後才進入下一步；不會把純噪聲配給所有時間步。
比較目標統一為 ComfyUI `BaseModel.apply_model` 回傳的 denoised latent，不混用原始速度預測。

快取：`C:\ComfyUI\output\tcsa_anima_denoise\kiwi_denoise_cache_01.safetensors`。
旁邊 JSON 記錄模型指紋、噪聲排程、老師/基準差距與耗時。
預覽圖是**最後一行場景、seed 43** 的老師圖片。請先確認這張在低解析度仍有目標畫風。
若低解析度已失去畫風，先改用 512×768、另一個 cache 名稱重新建立；不要直接訓練無效老師。
已有快取不覆寫；重做請改 name。

## B. 先跑一次反向傳播

拖入 `examples/denoise_probe_workflow.json`，這個工作流不會重新生成老師。
原始 MODEL 和 CPU adapter 接到 `TCSA Denoise Train (Experimental)`。

| 欄位 | 值 |
|---|---|
| cache_name | `kiwi_denoise_cache_01` |
| mode | `probe_one_update` |
| updates | 10（probe 模式固定只做 1 次，不使用此數字） |
| rank | 32 |
| learning_rate | 0.0001 |
| seed | 42，fixed |
| name | `kiwi_denoise_probe_01` |

此模式會先做零 LoRA 的驗證，再執行一次真正的前向、反向傳播、AdamW 更新，最後再驗證。
所以它不只花一次前向的時間；一行場景會額外有更新前後共 8 次驗證前向。
**probe 不匯出 LoRA**，只確認此硬體與執行路徑能否訓練。

結果：`C:\ComfyUI\output\tcsa_anima_denoise\kiwi_denoise_probe_01.json`。
成功狀態 `probe_passed`，請回報整個 JSON，尤其是：

- `first_gradient_norm`：必須有限且大於 0；確認梯度穿過完整去噪模型到達 LoRA。
- `history` 中的 `update_seconds`：一次更新的時間，不包含後續驗證。
- `peak_cuda_allocated_mb`：PyTorch 記錄的 GPU 配置峰值，不是整張卡的總使用量，也不是 RAM 峰值。
- `initial_validation_loss` / `best_validation_loss`：相對老師/基準差距的誤差。
- `total_seconds`、`status`、`error`（如果失敗）。

若 OOM、梯度為零、圖被 detach、GGUF/attention 算子不支援 backward，會停止並寫失敗 JSON，
不會假裝成功匯出。不要增加 updates；先提供報告。
可按 ComfyUI 取消。中斷檢查在模型前向及每次更新邊界，單次 GPU 運算本身無法立即打斷。

## C. 通過後才做 10 次更新

沿用快取，把 mode 改為 `train_short`、updates 設 10、name 改為 `kiwi_denoise_train_01`。
每 5 次和最後一次檢查驗證誤差；只匯出驗證誤差有改善的最佳權重。
若沒有改善，狀態為 `no_validation_improvement`，沒有 LoRA 檔。

成功匯出時：

`C:\ComfyUI\output\tcsa_anima_denoise\kiwi_denoise_train_01.safetensors`

這個才是 LoRA；**不要把名字含 cache 的 safetensors 當 LoRA 載入**。
每次 train_short 從零 LoRA 開始，不是接續上次。每次測試請換新 name，保留報告。

匯出後複製至 `C:\ComfyUI\models\loras\`。先用既有 `TCSA LoRA Check (CPU)` 做直接套用測試：
MODEL 直接接未套 LoRA 的 GGUF，positive 比較 baseline / lora_applied，固定 seed、CFG 1，
且不重複套用提取條件。再用一般 LoRA Loader、原生 CLIP Text Encode 檢查載入一致性。
最後必須換未訓練的場景和 seed 才能評估泛化。

## 數值意義與限制

Loss = MSE(學生 denoised, 老師 denoised) / MSE(基準 denoised, 老師 denoised)。
零 LoRA 理論值約 1；小於 1 代表更接近老師，大於 1 代表更遠。它不是畫風相似度。
更新前若離 1 超過 5%，會拒絕訓練，避免混用不一致的快取/計算路徑。
老師與基準幾乎相同也會拒絕訓練。

- 第一輪驗證僅隔離 seed，**沒有隔離場景**，不能據此宣稱泛化成功。
- CFG 固定 1；沒有訓練負面/無條件分支，不保證 CFG > 1 的效果。
- 教師軌跡上的單步擬合不保證學生自己完整生成時仍維持相同軌跡。
- MODEL 指紋是每個張量的抽樣內容和模型/排程資訊，不是整個 checkpoint 的完整雜湊。
  CPU adapter 另比對提取時的 SHA256，且檢查 MODEL 內 out_proj / norm 是否一致。
- 透過逐 block checkpoint 和 saved-tensor CPU offload 減少顯存需求，代價是更多計算、RAM 和傳輸。
  GGUF 能推論不等於能反向傳播；**尚未在 K620 或完整 Anima 上實測**。
- 動態 VRAM、其他 MODEL patches/wrappers 暫不支援；不會自動降級成別種訓練方式。
- 這裡先使用原本的 adapter out_proj LoRA，仍可能需要改動生成模型內的其他層才有足夠表達能力。

開發驗證使用小型可微分模型，涵蓋配對軌跡、梯度到達、驗證改善、快取讀寫、取消、
失敗時還原模型及 checkpoint 梯度一致性；不把這些測試當成真實畫風成功。
