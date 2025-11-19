import streamlit as st
import json
import pandas as pd
import re
import time
import hashlib
import html  # 新增：用于 HTML 转义
from openai import OpenAI

# --- 缓存配置 ---
CACHE_SIZE_LIMIT = 10
CACHE_TTL_SECONDS = 30

# 初始化缓存
if 'llm_cache' not in st.session_state:
    st.session_state.llm_cache = {}

modelscope_key = st.secrets["modelscope"]["key"]
# 客户端配置（请按需配置）
modelscope_client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key=modelscope_key
)

# 模型配置
model_type = "modelscope"  # 或 "ollama"
# model_name = "gemma3n:latest"  # ollama
model_name = 'Qwen/Qwen3-235B-A22B-Instruct-2507'  # modelscope

# --- 模拟函数（仅用于演示）---
def get_mock_json_response(prompt_content):
    sentence_match = re.search(r'### \*\*待分析的英文长句\*\*\n(.*?)\n', prompt_content, re.DOTALL)
    sentence = sentence_match.group(1).strip() if sentence_match else ""

    if not sentence:
        return json.dumps({"error": "No sentence provided"})

    res_json_dict = {
        "Source Sentence": sentence,
        "Translation": "尽管该宣言承诺此后未经民选立法机构批准的法律不会生效，但法院似乎没有意识到这一承诺包含了一份宪法性文件。",
        "StructureAnalysis": [
            {
                "segment": "Although the manifesto pledged that henceforth no law would go into effect without the approval of a popularly elected legislature",
                "highlight": True,
                "role": "让步状语从句 (Adverbial Clause of Concession)",
                "explanation_cn": "由连词“Although”引导，说明了主句动作发生的背景或条件。"
            },
            {
                "segment": ", ",
                "highlight": False,
                "role": "",
                "explanation_cn": ""
            },
            {
                "segment": "the Court seemed unaware",
                "highlight": True,
                "role": "主句核心 (Main Clause Core)",
                "explanation_cn": "主语和主要谓语动词，表达了法院似乎没有意识到某个事实。"
            },
            {
                "segment": " that this pledge entailed a constitutional charter.",
                "highlight": True,
                "role": "宾语补足语从句 (Complement to 'unaware')",
                "explanation_cn": "由'that'引导，具体说明了法院没有意识到的内容。"
            }
        ],
        "Vocabulary": [
            {
                "word": "manifesto",
                "pos": "n.",
                "definition": "宣言，声明",
                "example": "The party issued a manifesto outlining its goals. (该党发布了一份阐述其目标的宣言。)"
            },
            {
                "word": "henceforth",
                "pos": "adv.",
                "definition": "从今以后，此后",
                "example": "Henceforth, all meetings will be held online. (从今以后，所有会议都将在网上举行。)"
            },
            {
                "word": "entailed",
                "pos": "v. (过去式)",
                "definition": "使……成为必需；牵涉，包含",
                "example": "The job entailed a lot of travelling. (这份工作需要经常出差。)"
            }
        ],
        "Decomposition": [
            {
                "id": 1,
                "function": "让步条件",
                "simplified_sentence_en": "The manifesto contained a promise about future laws."
            },
            {
                "id": 2,
                "function": "核心主干",
                "simplified_sentence_en": "The Court appeared to be unaware of a critical fact."
            },
            {
                "id": 3,
                "function": "未意识到的内容",
                "simplified_sentence_en": "That original promise itself essentially established a constitutional document."
            }
        ]
    }
    return json.dumps(res_json_dict)

# --- JSON 提取与校验 ---
def extract_json_from_llm_response(raw_text):
    # 尝试清理围栏
    json_text = re.sub(r'^\s*```(?:json)?\s*\n*', '', raw_text, flags=re.IGNORECASE | re.DOTALL)
    json_text = re.sub(r'\s*\n*```\s*$', '', json_text, flags=re.DOTALL)
    try:
        return json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return None

def validate_analysis_json(data):
    required_keys = ["Source Sentence", "Translation", "StructureAnalysis", "Vocabulary", "Decomposition"]
    if not isinstance(data, dict):
        return False
    for key in required_keys:
        if key not in data:
            return False
    sa = data.get("StructureAnalysis", [])
    if not isinstance(sa, list):
        return False
    for item in sa:
        if not isinstance(item, dict):
            return False
        if item.get("highlight", False):
            if not item.get("segment") or not item.get("role") or not item.get("explanation_cn"):
                return False
    return True

# --- LLM 调用（带缓存） ---
def llm_english_analyze_with_time(englist_sentence, llm_type): 
    cache_key = hashlib.md5(englist_sentence.encode('utf-8')).hexdigest()
    current_time = time.time()
    llm_cache = st.session_state.llm_cache

    # 缓存检查
    if cache_key in llm_cache:
        cached_entry = llm_cache[cache_key]
        if current_time - cached_entry['timestamp'] < CACHE_TTL_SECONDS:
            st.info("ℹ️ **缓存命中**: 30秒内避免重复调用 LLM。")
            return cached_entry['result'], 0.0

    # 调用 LLM
    start_time = time.time()
    
    prompt_text = """
你是一位专业的英语语言学专家和高级语法分析师。你的任务是对用户提供的任何英语长难句进行彻底的、结构化的语法解析和词汇解释。

---
### **任务指令**
请对用户提供的**唯一的英文长句**进行分析，并**严格**按照以下定义的 **JSON 结构**返回结果。

### **JSON 结构和内容要求**
你的输出必须是**纯粹的 JSON 格式**（不包含任何 Markdown 格式如 ````json`，也不包含任何解释性文字或开场白）。JSON 对象必须包含以下五个顶层键（Key）：

"Source Sentence"：对应处理的原始文本。
    * **Value 类型：** 字符串 (string)
"Translation"：对应中文翻译。
    * **Value 类型：** 字符串 (string)
    * **内容要求：** 提供一个准确、流畅的中文翻译。

"StructureAnalysis"：对应句子结构分析。
    * **Value 类型：** 数组 (array)
    * **内容要求：** 数组中的每个元素都是一个对象，包含以下四个键：
        * `segment`: 英文部分 (English Segment)。**注意：所有片段必须按顺序拼接起来，完整地重构原英文句。**
        * `highlight`: **布尔值** (boolean)。如果该片段是需要分析和高亮的重要结构（如主句、从句），则为 **true**；否则（如连接词、不重要的介词短语、标点、空格）则为 **false**。
        * `role`: 结构名称和功能 (Structure Name & Role)。**仅当 `highlight: true` 时填充。** 必须简要概括其结构类型和语法功能。
        * `explanation_cn`: 中文解释/备注。**仅当 `highlight: true` 时填充。** 解释该结构的作用。
    * 【验证要求】必须保证全部的segment的组合起来与原始句子是一致的。

"Vocabulary"：对应核心词汇。
    * **Value 类型：** 数组 (array)
    * **内容要求：** 数组中的每个元素都是一个对象，包含以下四个键：
        * `word`: 词汇 (Word)
        * `pos`: 词性 (Part of Speech)
        * `definition`: 定义 (Definition)
        * `example`: 示例 (包含英文例句及其中文翻译)

"Decomposition"：对应句子拆解。
    * **Value 类型：** 数组 (array)
    * **内容要求：** 数组中的每个元素都是一个对象，包含以下三个键：
        * `id`: 序号 (如 1, 2, 3...)
        * `function`: 功能 (Function)
        * `simplified_sentence_en`: 拆解后的简单**英文**句

---
### **严格约束**
1.  **输出必须是纯粹、完整的 JSON 对象，不可包含任何其他文本、Markdown 格式或说明。**
2.  必须严格遵循上述定义的 Key 名称和结构。

### **待分析的英文长句**
{englist_sentence}
"""
    prompt = prompt_text.format(englist_sentence=englist_sentence)

    llm_result = None
    try:
        match llm_type:
            case "modelscope":
                extra_body = {"enable_thinking": False, "thinking_budget": 1024}
                response = modelscope_client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=False,
                    extra_body=extra_body
                )
                llm_result = response.choices[0].message.content
            case _:
                llm_result = get_mock_json_response(prompt)
    except Exception as e:
        st.error(f"LLM 调用失败: {e}")
        return None, 0.0

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    json_result = extract_json_from_llm_response(llm_result)

    # 更新缓存
    if json_result and validate_analysis_json(json_result):
        llm_cache_keys = list(llm_cache.keys())
        if len(llm_cache_keys) >= CACHE_SIZE_LIMIT:
            oldest_key = min(llm_cache_keys, key=lambda k: llm_cache[k]['timestamp'])
            del llm_cache[oldest_key]
        llm_cache[cache_key] = {
            'result': json_result,
            'timestamp': current_time
        }

    return json_result, elapsed_time

# --- ✅ 修复核心：安全高亮函数 ---
def create_instant_hover_highlight(segment: str, tooltip_content: str, color: str) -> str:
    """
    安全生成带 Tooltip 的高亮 HTML，防止 XSS 和渲染失败。
    """
    # 🔒 核心：严格转义用户内容
    escaped_segment = html.escape(segment.strip())
    escaped_tooltip = html.escape(tooltip_content.strip(), quote=True)
    
    # 使用紧凑格式，避免多余空格/换行
    return (
        f'<span class="custom-tooltip" '
        f'data-tooltip="{escaped_tooltip}" '
        f'style="'
        f'background-color: {color}; '
        f'border-bottom: 2px solid #ffbf00; '
        f'padding: 2px 0; '
        f'margin-right: 2px;'
        f'">'
        f'{escaped_segment}'
        f'</span>'
    )

# --- CSS 样式 ---
CUSTOM_CSS = """
<style>
/* 高亮文本容器：保持 inline，但为 tooltip 定位提供参考 */
.custom-tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
    /* 避免行内元素间隙问题 */
    vertical-align: bottom;
}

/* Tooltip 主体 */
.custom-tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    z-index: 2000; /* 提高防止被其他元素遮挡 */

    /* ✅ 固定宽度 + 自适应高度 */
    width: 350px;              /* ← 固定宽度：可调整为 400px/500px */
    max-width: 90vw;           /* 响应式：不超过视口 90% */
    white-space: normal;       /* 允许换行 */
    word-wrap: break-word;     /* 强制止长词换行 */
    text-align: left;

    /* 样式美化 */
    background-color: #2e2e2e;
    color: #ffffff;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 14px;
    line-height: 1.5;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);

    /* 默认隐藏 */
    opacity: 0;
    visibility: hidden;
    pointer-events: none;      /* 鼠标穿透，避免干扰 hover */

    /* ✅ 定位：以高亮文本中心为基准，向上偏移 */
    top: auto;
    bottom: 100%;              /* 紧贴高亮文本上方 */
    left: 50%;
    transform: translateX(-50%) translateY(8px); /* 向上偏移 8px 留出间隙 */

    /* 动画：即时显示（0s），但退出时可加一点延迟避免闪跳 */
    transition: opacity 0.15s ease, visibility 0.15s, transform 0.15s;
}

/* 悬停显示 */
.custom-tooltip:hover::after {
    opacity: 1;
    visibility: visible;
    transform: translateX(-50%) translateY(0); /* 向上移动到最终位置 */
}

/* ✅ 可选：添加小箭头（指向高亮文本） */
.custom-tooltip::before {
    content: '';
    position: absolute;
    z-index: 2001;
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 6px solid #2e2e2e;
    top: auto;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%) translateY(2px); /* 箭头在 tooltip 下方 */
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.15s, visibility 0.15s;
}
.custom-tooltip:hover::before {
    opacity: 1;
    visibility: visible;
}

/* 深色模式适配（Streamlit 默认） */
[data-theme="dark"] .custom-tooltip::after {
    background-color: #3a3a3a;
}
[data-theme="dark"] .custom-tooltip::before {
    border-top-color: #3a3a3a;
}

/* 浅色模式适配（如有）*/
[data-theme="light"] .custom-tooltip::after {
    background-color: #333;
    color: #fff;
}
[data-theme="light"] .custom-tooltip::before {
    border-top-color: #333;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- 页面配置 ---
st.set_page_config(layout="wide", page_title="英语长难句分析工具")
st.title("📚 英语长难句深度分析器")
st.subheader(f"基于结构化 JSON 输出的解析和可视化 | LLM: **{model_type}** ***{model_name}***")

default_sentence = "Although the manifesto pledged that henceforth no law would go into effect without the approval of a popularly elected legislature, the Court seemed unaware that this pledge entailed a constitutional charter."
sentence_input = st.text_area("输入英文长难句:", default_sentence, height=100)

if st.button("开始分析", type="primary"):
    if not sentence_input.strip():
        st.error("请输入一个英文句子进行分析。")
        st.stop()

    with st.spinner("正在调用 LLM 或检查缓存..."):
        analysis_json, elapsed_time = llm_english_analyze_with_time(sentence_input.strip(), llm_type=model_type) 
    
    if analysis_json is None or not validate_analysis_json(analysis_json):
        st.error("❌ 分析失败：LLM 返回结果为空或 JSON 格式非法。")
        if analysis_json:
            st.json(analysis_json)
        st.stop()
    
    if elapsed_time > 0.0:
        st.info(f"⏱️ **LLM 分析耗时**: **{elapsed_time:.2f}** 秒")
    
    source_sentence_llm = analysis_json.get("Source Sentence", "").strip()
    if source_sentence_llm and source_sentence_llm != sentence_input.strip():
        st.warning(f"⚠️ LLM 处理的句子与输入略有不同：\n`{source_sentence_llm}`")

    st.success("✅ 分析完成！")
    st.divider()

    # --- 1. 翻译 ---
    st.header("1. 🔧 中文翻译 (Translation)")
    st.info(analysis_json.get("Translation", "N/A"))
    st.divider()

    # --- 2. 结构高亮（✅ 修复重点）---
    st.header("2. 🧐 句子结构分析 (Interactive Highlight)")
    st.caption("悬停高亮部分查看语法角色与解释")

    structure_data = analysis_json.get("StructureAnalysis", [])
    HIGHLIGHT_COLORS = ["#fce8a9", "#a9fce8", "#e8a9fc", "#fcd9a9"]
    color_index = 0
    highlighted_parts = []

    for item in structure_data:
        segment = item.get("segment", "")
        is_highlight = item.get("highlight", False)
        
        if is_highlight and segment.strip():
            color = HIGHLIGHT_COLORS[color_index % len(HIGHLIGHT_COLORS)]
            color_index += 1
            role = item.get("role", "结构")
            explanation = item.get("explanation_cn", "无解释")
            tooltip = f"【{role}】: {explanation}"
            highlighted_parts.append(create_instant_hover_highlight(segment, tooltip, color))
        else:
            # 非高亮部分也 escape，防尖括号破坏
            escaped_plain = html.escape(segment)
            highlighted_parts.append(escaped_plain)
    
    # 拼接为紧凑 HTML 字符串
    highlighted_sentence = "".join(highlighted_parts).strip()

    st.markdown("**原句:**")
    # ✅ 关键：使用 st.html() 渲染（Streamlit ≥1.34）
    try:
        st.html(highlighted_sentence)
    except AttributeError:
        # 兼容旧版 Streamlit（不推荐长期使用）
        st.warning("请升级 Streamlit 到 ≥1.34 以获得最佳 HTML 渲染效果！")
        st.markdown(highlighted_sentence, unsafe_allow_html=True)

    # --- 2b. 高亮片段卡片展示 ---
    st.header("2b. 🎴 高亮片段卡片展示")
    st.caption("每张卡片显示原句中的一个高亮部分及其语法角色与解释")

    for idx, item in enumerate(structure_data):
        segment = item.get("segment", "").strip()
        is_highlight = item.get("highlight", False)
        
        if is_highlight and segment:
            color = HIGHLIGHT_COLORS[idx % len(HIGHLIGHT_COLORS)]
            role = item.get("role", "结构")
            explanation = item.get("explanation_cn", "无解释")
            
            card_html = f"""
            <div style="
                background-color: {color};
                padding: 12px 16px;
                margin-bottom: 8px;
                border-radius: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            ">
                <strong>原句片段:</strong> {html.escape(segment)}<br>
                <strong>角色:</strong> {role}<br>
                <strong>解释:</strong> {explanation}
            </div>
            """
            try:
                st.html(card_html)
            except AttributeError:
                st.markdown(card_html, unsafe_allow_html=True)

    st.divider()

    # --- 3. 词汇 ---
    st.header("3. 🔑 核心词汇 (Key Vocabulary)")
    vocab_data = analysis_json.get("Vocabulary", [])
    if vocab_data:
        df_vocab = pd.DataFrame(vocab_data)
        df_vocab.columns = ["词汇 (Word)", "词性 (POS)", "定义 (Definition)", "示例 (Example)"]
        st.table(df_vocab)
    st.divider()

    # --- 4. 拆解 ---
    st.header("4. ✨ 句子拆解 (Sentence Decomposition)")
    decomp_data = analysis_json.get("Decomposition", [])
    if decomp_data:
        df_decomp = pd.DataFrame(decomp_data)
        df_decomp.columns = ["ID", "功能 (Function)", "拆解后的简单英文句 (Simplified English Sentence)"]
        st.table(df_decomp)