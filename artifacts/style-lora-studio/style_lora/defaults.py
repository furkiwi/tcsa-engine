"""TCSA paper defaults: watercolor split, null controls, 12-layer tap."""

METHOD_NAME = "TCSA"
METHOD_FULL = "Text-Conditioning Style Absorption"

STYLE_NAME = "watercolor"

STYLE_PHRASES = [
    "watercolor",
    "watercolor painting",
    "traditional watercolor illustration",
    "hand-painted watercolor",
]

EXTRACTION_CONTENTS = [
    "a tiger walking through a jungle",
    "a woman standing in a city",
    "a spaceship above a planet",
    "a medieval castle on a mountain",
    "a bowl of fruit on a wooden table",
    "a futuristic motorcycle in a desert",
    "a harbor at dusk with fishing boats",
    "a jungle path with hanging vines",
    "an old library with tall shelves",
    "a red train crossing a steel bridge",
    "a ceramic teapot on a linen cloth",
    "a fox sitting in snow",
    "a greenhouse filled with ferns",
    "a stone bridge over a quiet canal",
    "a violin on an empty chair",
    "a lantern hanging from a cedar tree",
    "a fisherman on a misty lake",
    "a bicycle leaning against a bakery",
    "a cathedral interior with stained glass",
    "a windmill on a grassy hill",
]

EVALUATION_CONTENTS = [
    "a girl walking through a forest",
    "a dog sitting on a porch",
    "a portrait of an old man",
    "a robot in a workshop",
    "a night market with paper lanterns",
    "a bowl of noodles on a table",
    "a mountain landscape with a lake",
    "a steam locomotive in the rain",
]

KNOWN_STYLES = {
    "watercolor",
    "oil painting",
    "oil",
    "anime",
    "pencil",
    "pencil sketch",
    "cinematic",
    "vintage",
    "sketch",
    "concept art",
    "editorial illustration",
    "ink wash",
    "impressionist",
    "gouache",
    "charcoal",
    "ukiyo-e",
    "pixel art",
}

# Table 1 / §4.4 — three null families. Style paraphrases must beat these.
NULL_CONTROLS = {
    "adjective": {
        "id": "adjective",
        "label": "無關形容詞",
        "phrases": ["yellow", "ancient", "tiny", "glossy"],
        "note": "不是渲染器。方向不該跟風格一樣穩。",
    },
    "non_renderer": {
        "id": "non_renderer",
        "label": "非風格視覺屬性",
        "phrases": ["photographed from above", "wide angle", "brightly lit", "macro close-up"],
        "note": "鏡頭 / 光線屬性，不是畫風。",
    },
    "nonce": {
        "id": "nonce",
        "label": "無意義詞",
        "phrases": ["xyzzorp glorblesh", "fnorple pigment"],
        "note": "模型不該認識。門檻應失敗。",
    },
}

# Official Krea 2 Qwen3-VL tap: twelve layers, indices 2, 5, …, 35.
KREA_TAP_LAYERS = list(range(2, 36, 3))

STYLE_PRESETS = [
    {
        "id": "watercolor",
        "label": "水彩",
        "name": "watercolor",
        "phrases": [
            "watercolor",
            "watercolor painting",
            "traditional watercolor illustration",
            "hand-painted watercolor",
        ],
        "note": "論文預設目標。色漬、紙紋、邊緣滲色。",
    },
    {
        "id": "oil",
        "label": "油畫",
        "name": "oil_painting",
        "phrases": [
            "oil painting",
            "classical oil painting",
            "impasto oil painting",
            "old master oil painting",
        ],
        "note": "厚塗與筆觸。適合測 rank 是否只改色盤。",
    },
    {
        "id": "pencil",
        "label": "鉛筆素描",
        "name": "pencil_sketch",
        "phrases": [
            "pencil sketch",
            "graphite drawing",
            "detailed pencil drawing",
            "charcoal sketch",
        ],
        "note": "低彩、線條主導。負向對照時很乾淨。",
    },
    {
        "id": "anime",
        "label": "動畫風",
        "name": "anime",
        "phrases": [
            "anime",
            "anime still",
            "cel shaded anime",
            "japanese animation frame",
        ],
        "note": "模型幾乎一定有這個概念；方向應很穩。",
    },
    {
        "id": "ink",
        "label": "水墨",
        "name": "ink_wash",
        "phrases": [
            "ink wash",
            "sumi-e",
            "traditional ink wash painting",
            "chinese ink painting",
        ],
        "note": "留白與濃淡。和水彩容易混淆，適合做干擾實驗。",
    },
    {
        "id": "cinematic",
        "label": "電影感",
        "name": "cinematic",
        "phrases": [
            "cinematic",
            "cinematic photography",
            "anamorphic film still",
            "dramatic cinematic lighting",
        ],
        "note": "偏攝影而非繪製。用來測「風格」是否變成鏡頭。",
    },
    {
        "id": "unknown",
        "label": "假風格（對照）",
        "name": "xyzzorp",
        "phrases": [
            "xyzzorp glorblesh",
            "xyzzorp style",
            "glorblesh rendering",
            "fnorple pigment",
        ],
        "note": "模型不該認識。四道門檻應失敗，不該寫成品 LoRA。",
    },
]

# Recommended inclusion order: projector, then txt_in, then fusion blocks (Phase 4).
DEFAULT_LAYERS = [
    "text_fusion.projector",
    "txt_in.linear_1",
    "txt_in.linear_2",
]

DEMO_DIM = 384
DEMO_LAYERS = {
    "text_fusion.projector": (6144, 12),
    "txt_in.linear_1": (1024, 384),
    "txt_in.linear_2": (6144, 1024),
}

DIFFUSERS_TO_NATIVE = {
    "txt_in.linear_1": "txtmlp.1",
    "txt_in.linear_2": "txtmlp.3",
    "text_fusion.projector": "txtfusion.projector",
    "text_fusion.layerwise_blocks": "txtfusion.layerwise_blocks",
    "text_fusion.refiner_blocks": "txtfusion.refiner_blocks",
}

NATIVE_TO_DIFFUSERS = {v: k for k, v in DIFFUSERS_TO_NATIVE.items()}

LAYER_HELP = {
    "text_fusion.projector": "txtfusion.projector · 層混入 DiT（優先測）",
    "txt_in.linear_1": "txtmlp.1 · 第一段文本寬投影",
    "txt_in.linear_2": "txtmlp.3 · 第二段文本寬投影",
}

DEFAULT_RANK = 4
DEFAULT_METHOD = "svdl"
ALS_ITERS = 4
SOFT_PROMPT_GATE = 0.02
SPECIFICITY_MARGIN = 0.05
COSINE_GATE = 0.4
PCA_GATE = 0.35
