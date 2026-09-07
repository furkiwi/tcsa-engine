# TCSA Studio

CPU 上可跑的 **Text-Conditioning Style Absorption** 實作，對應整合論文：

*Text-Conditioning Style Absorption for Krea 2 RAW*

不跑 12B DiT。流水線對齊論文六段：

1. 配對 `P_c` / `P_{c+s}`（抽取集與測試集切開）
2. 官方文本編碼（Demo hash，或 Qwen3-VL **12 層 tap**：indices 2,5,…,35）
3. 四道門檻：cosine、PCA λ₁、風格 paraphrases vs 三組 null、soft-prompt `h+αμ`
4. Rank-1 解析初始化（`v = h̄ / ||h̄||²`）
5. **主方法** thin-SVD / 最小二乘 + ALS，寫入 `txt_in` / `text_fusion.projector`
6. 匯出 `.safetensors`，到 hosted Krea 2 Turbo 跑 A/B/C/D

`lora_alpha = rank`，slider 1.0 對應構造出的 ΔW。

## 啟動 GUI

```bash
cd style-lora-studio
python3 -m pip install -r requirements.txt
python3 app.py
```

瀏覽器打開 `http://127.0.0.1:8080`。

預設 **Demo**：無權重即可走完整條（熱圖、null、soft-prompt、下載 LoRA）。Demo 的 `W` 是隨機矩陣，格式正確但不能掛到 Krea。

## 真實模式

1. Qwen3-VL-4B + Krea 2 RAW `.safetensors`
2. GUI 選 Real，填路徑；instruction template 必須對齊 Krea 推理
3. `pip install torch transformers`
4. 編碼在 CPU 上跑短句；RAW **懶加載** 並 **對盤驗證** `txtmlp` / `txtfusion` 鍵名
5. 預設 rank **4**（論文第一次認真驗證）；rank 1 只當初始化，rank 8 才是升級

## 命令列

```bash
python3 cli.py go --mode demo --rank 4 --method svdl --out watercolor_tcsa.safetensors
python3 cli.py analyze --mode demo
python3 cli.py distill <job_id> --rank 4 --method svdl --format kohya
```

## 驗證

同一 seed 在 **Krea 2 Turbo** 上跑：

| 條件 | Prompt | LoRA |
|---|---|---|
| A | `a girl walking through a forest` | 關 |
| B | `a watercolor girl walking through a forest` | 關 |
| C | `a girl walking through a forest` | 開 |
| D | `a watercolor girl walking through a forest` | 開 |

成敗看 **A→C**。強度掃 0.25–1.25。

## 測試

```bash
python3 tests/test_pipeline.py
```
