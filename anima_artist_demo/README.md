# Anima Artist Demo（CPU）
解壓至 ComfyUI/custom_nodes/anima_artist_demo/，確認 __init__.py 位於這一層。
使用已有 Anima 工作流的 ComfyUI，以 python main.py --cpu --force-fp32 啟動。
Windows portable 請在 portable 根目錄執行 .\python_embeded\python.exe -s ComfyUI\main.py --cpu --force-fp32 --windows-standalone-build 。

1. 載入平常可使用的官方 Anima 工作流，使用原版、未套 LoRA 的 MODEL 及 Anima CLIP。兩者接到 AnimaArtistExtractCPU。輸入 @繪師、profile 英數檔名、每行一個英文場景。只排入抽取節點即可，不需要 VAE 或取樣。重用名稱會覆寫檔案。
2. 結果存於 ComfyUI/output/anima_artist_profiles/ 下的 .npy。每列是 artist 前綴（含逗號）在 T5 序列對應的一個 adapter 輸出向量，跨場景平均。保留 EOS 在內容側。若 token 邊界不一致會拒絕輸出。
3. 新增 AnimaArtistApplyCPU，接入相同 MODEL、CLIP，填寫同名 profile 與未用過的新場景；輸出接 KSampler 的 positive，negative 保持原設定。也可將 Extract 的 STRING 輸出接到 Apply 的 profile（轉成輸入），確保先抽再用。
4. 對同一新場景、固定 seed 與所有取樣參數比較：原生正向文字無標籤、原生正向文字加相同 @繪師 前綴、Apply 強度 0、Apply 強度 1。0 應等同原生無標籤；1 是移植候選。非零強度只縮放插入向量，並非線性插值。比較至少兩個 seed 的線條、上色及主體。若 0 就明顯不同，先排查工作流。
5. 檔案需配合同版本 Anima 與文字編碼器。此實驗只移植前綴，沒有學習繪師對內容 token 的改變。它不是標準 LoRA。ComfyUI 的 MODEL loader 仍需打開 checkpoint；抽取期間不跑去噪，出圖期間會跑 CPU 去噪。若記憶體不足先單獨抽取、關閉其他模型。
