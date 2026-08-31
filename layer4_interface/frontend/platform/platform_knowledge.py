"""platform_knowledge — 平台功能帮助的单一事实来源拼装（Layer 4 平台）.

chat 信息工具（``get_platform_help``）与无 LLM 兜底（``fallback_intent``）
共用同一份按功能主题组织的帮助资料，避免各自漂移：每个主题给出
「是什么 / 怎么说一句话触发 / 完整入口在哪 / 注意事项」。主题关键词
（``PlatformTopic.keywords``）供字符串匹配把「具体功能怎么用」类提问
路由到正确的主题文档。

为什么需要它：旧 ``help`` 工具只返回一段泛泛的总览（``_HELP_TEXT``），
LLM 面对「怎么创建游戏 / 在线学习怎么用 / 评测中心在哪 / 教学对局是
什么」这类**具体功能**提问时拿不到权威资料，只能泛泛而谈或编造。
``get_platform_help(topic=...)`` 让模型像 ``describe_game`` 一样先取
权威说明再作答（零幻觉路径）；无 LLM 时同一份文档做确定性回答。

内容事实来源：平台路由（``server.py``）、前端页面（``platform-frontend``
导航/页面文案）、``docs/user/*.md`` 与 ``docs/design/online-learning.md``。
改造功能时同步更新对应主题文本（docs 更新即同步，与 ``game_knowledge``
同一原则）。依赖方向：只依赖标准库，``agent`` 与 ``platform`` 两侧均可
安全导入。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 单主题文档的字符上限（超出截断，fail-soft 宁缺毋滥）。
_HELP_TOPIC_MAX = 1000
#: 主题总览（index）的字符上限。
_INDEX_MAX = 1600


@dataclass(frozen=True)
class PlatformTopic:
    """一个平台功能主题的权威帮助文档。

    Attributes:
        key: 稳定主题 id（工具参数 enum 与 ``params.topic`` 用）。
        title: 展示标题（一句话）。
        summary: 主题总览里的一行简介。
        text: 完整回答（是什么 / 怎么说 / 入口在哪 / 注意点）。
        keywords: 用户提问关键词（子串匹配；同句多命中时最长匹配胜出）。
    """

    key: str
    title: str
    summary: str
    text: str
    keywords: tuple[str, ...]


_OVERVIEW_TEXT = (
    "Gavis 是自适应策略游戏平台，对话即操作：想开哪局直接说“玩月亮棋”/“来一局德州扑克”；"
    "对局中说动作（“下第2行第3列”“跟注”“打这张牌”）或直接点棋盘/牌面；"
    "“继续上一局”恢复对局，“这步怎么走”要提示，“看战绩”“复盘上一局”查记录。\n"
    "平台还有这些面板：创建游戏（自然语言写规则）、设置（难度/声音/主题）、"
    "评测中心（AI vs AI 求解器对比）、在线学习（人类决策→门禁发布）、"
    "教学对局（教练看你的牌）、LLM 配置（端点/模型/密钥）、视觉识别（截图/拍照识别）。\n"
    "完整平台界面 = 大厅 + 对局 + 战绩 + 复盘 + 创建 + 设置 + 评测 + 在线学习 + LLM 配置。"
)

PLATFORM_TOPICS: tuple[PlatformTopic, ...] = (
    PlatformTopic(
        key="overview",
        title="平台总览",
        summary="平台能做什么：对话即操作 + 各功能面板一览。",
        text=_OVERVIEW_TEXT,
        keywords=("有哪些功能", "功能介绍", "有什么功能", "什么功能", "功能列表", "怎么开始用", "怎么使用平台", "平台功能"),
    ),
    PlatformTopic(
        key="play",
        title="开局 / 对战",
        summary="说“玩X”或从大厅选游戏即可开局；平台支持棋盘/扑克/麻将/UNO/社交/自定义游戏。",
        text=(
            "说“玩月亮棋”“来一局德州扑克”（或在大厅直接点游戏）即可开局；没指明游戏时助手会追问。\n"
            "平台游戏：月亮棋、随机五子棋（棋盘）；德州扑克（扑克）；麻将六变种"
            "（广东/红中/血战/四川/长沙/台湾，默认 4 人）；UNO 六变体（经典/7-0/抢牌/叠加/"
            "摸到能打/严格+4，2-10 人）；谁是卧底（4-12 人发言桌游）；狼人杀（9 人社交推理——\n"
            "说「玩狼人杀」或在大厅点「狼人杀」即可开局）；以及其余自定义游戏。\n"
            "开局后直接进入对战；结束后可“复盘上一局”或“再来一局”。"
        ),
        keywords=(
            "怎么开局",
            "怎么开始游戏",
            "怎么开一局",
            "怎么开始一局",
            "开始新游戏",
            "开新对局",
            "新开一局",
            "怎么玩",
        ),
    ),
    PlatformTopic(
        key="resume",
        title="继续对局",
        summary="说“继续上一局”/“接着玩”恢复进行中的对局。",
        text=(
            "说“继续上一局”“接着玩”“恢复对局”即可回到当前进行中的对局；"
            "没有进行中对局时助手会说明并建议先开一局。\n"
            "完整平台界面里同样可以从大厅点进行中的对局继续。"
        ),
        keywords=("怎么继续", "继续对局", "恢复对局", "接着玩", "接着下", "接着打", "如何继续"),
    ),
    PlatformTopic(
        key="move",
        title="对局中操作",
        summary="对局里直接说动作或点击棋盘/牌面；合法动作以当前局面描述为准。",
        text=(
            "对局中直接说动作：棋盘游戏“下第2行第3列”“下第5格”“下中间”；"
            "扑克“跟注/弃牌/加注”；麻将“打这张”“碰/吃/杠”；UNO“出牌/摸牌”；"
            "发言桌游直接说台词（如“我是平民”）。也可以直接点击棋盘或牌面走快路径。\n"
            "含糊或不合法的动作助手会追问并给出当前合法动作清单（来自局面投影，绝不编造）；"
            "复杂局面（麻将/UNO/发言桌游）建议直接点击操作，最稳妥。"
        ),
        keywords=("怎么落子", "怎么下棋", "怎么出牌", "怎么打牌", "怎么发言", "怎么操作", "操作方式", "怎么走子"),
    ),
    PlatformTopic(
        key="hint",
        title="提示 / 指导",
        summary="说“这步怎么走”“提示我”要提示；分方向/具体/演示三级。",
        text=(
            "对局中说“这步怎么走”“提示我”“下一步怎么走”即可获得提示，分三级：\n"
            "· 方向（direction）：这一步大致该往哪个方向走；\n"
            "· 具体（specific）：给出具体推荐走法；\n"
            "· 演示（demo）：演示当前合法动作怎么执行。\n"
            "提示基于玩家自己可见的局面计算（不泄露 AI 信息），拿到后可以继续追问讲解。"
        ),
        keywords=(
            "怎么要提示",
            "怎么提示",
            "要提示",
            "提示功能",
            "这步怎么走",
            "下一步怎么走",
            "如何提示",
            "怎么走这步",
            "教我走",
        ),
    ),
    PlatformTopic(
        key="history",
        title="战绩 / 历史",
        summary="说“看战绩”查最近对局；完整表格在顶部「战绩」页。",
        text=(
            "说“看战绩”“查记录”“胜率”即可查看最近对局（游戏、难度、胜负、手数）。\n"
            "完整战绩表格在顶部导航「战绩」页（HistoryPage），支持按游戏/难度筛选；"
            "对局档案存在 data/matches/。"
        ),
        keywords=("看战绩", "查战绩", "战绩在哪", "怎么查战绩", "对局记录", "历史记录", "怎么查记录"),
    ),
    PlatformTopic(
        key="review",
        title="复盘",
        summary="说“复盘上一局”拉时间线+关键节点+改进建议；完整回放在「复盘」页。",
        text=(
            "说“复盘上一局”“回顾一下”即可：后端拉最近一局（或指定 match_id）的完整走子"
            "时间线 + 关键节点（转折点/胜着/昏招，含具体动作）+ 改进建议，由助手用中文讲解。\n"
            "完整逐手回放在顶部导航「复盘」页（ReviewPage）；没有已结束对局时助手会说明。"
        ),
        keywords=("怎么复盘", "如何复盘", "复盘功能", "复盘在哪", "回顾一下", "怎么回顾", "复盘一下"),
    ),
    PlatformTopic(
        key="create",
        title="创建自定义游戏",
        summary="说“创建一个新游戏”或顶部「创建游戏」；自然语言写规则或模板改变体。",
        text=(
            "说“创建一个新游戏”或进顶部导航「创建游戏」（/create）。两种模式：\n"
            "· 自然语言写规则：一段中文规则 → 翻译 → 校验 → 规则族识别（棋盘/扑克/麻将/社交）"
            "→ 直接可对弈；\n"
            "· 模板变体：改一款基础游戏（如给 UNO 加变体、麻将改番型）。\n"
            "创建结果展示校验结论、置信度、规则族与变更摘要；识别不了的规则会明确提示"
            "「暂不支持平台对弈」而不是静默失败。自定义游戏持久化在 data/custom_games/，"
            "大厅自定义卡片可删除；需要 LLM 翻译时可勾选（模型不可用自动回落确定性翻译）。"
        ),
        keywords=("怎么创建", "如何创建", "创建游戏", "新建游戏", "自定义游戏", "写规则", "规则翻译", "如何自定义"),
    ),
    PlatformTopic(
        key="settings",
        title="设置",
        summary="说“打开设置”/“怎么改难度”；可调 AI 难度、声音、主题等。",
        text=(
            "说“打开设置”“怎么改难度”或进顶部「设置」页：可调 AI 难度"
            "（简单/正常/困难；注：麻将当前为固定强度的启发式策略，三档暂无实际差异）、"
            "自适应难度、声音/主题、教练开关（教学对局）等偏好。\n"
            "LLM 的端点/模型/密钥配置在侧边栏「LLM 配置」页（见“LLM 配置”主题）。"
        ),
        keywords=(
            "怎么改难度",
            "如何改难度",
            "难度设置",
            "改变难度",
            "调难度",
            "难度",
            "声音",
            "主题设置",
            "怎么调设置",
        ),
    ),
    PlatformTopic(
        key="platform",
        title="完整平台界面",
        summary="说“打开平台界面”从对话切回完整平台（大厅/对局/战绩/复盘/…）。",
        text=(
            "说“打开平台界面”“回到大厅”即可从聊天切回完整平台界面："
            "大厅（选游戏/进行中对局）+ 对局页 + 战绩 + 复盘 + 创建游戏 + 设置 + "
            "评测中心 + 在线学习 + LLM 配置 + 我的画像。\n"
            "平台服务默认 8770 端口（开发模式前端 5173）。"
        ),
        keywords=("平台界面", "完整界面", "回平台", "平台首页", "回到大厅"),
    ),
    PlatformTopic(
        key="benchmark",
        title="评测中心",
        summary="说“评测中心”/“求解器对比”：AI vs AI 短赛对比各求解器，双方交替先手。",
        text=(
            "说“评测中心”“求解器对比”“模拟对局”或进顶部「评测中心」页："
            "发起 AI vs AI 短赛，双方交替先手消除先手优势，按注册表人数对局，"
            "对比各求解器（MCTS/CFR/PPO/PSRO/Hybrid/MAAC/QMix/HAPPO 等，"
            "视游戏装配）。页面可发起任务、看进行状态与结果明细；评测任务有数量上限保护。\n"
            "命令行等价入口：python -m demos.benchmark_all --game X --episodes N。"
        ),
        keywords=("评测中心", "求解器对比", "对比求解器", "模拟对局", "评测功能", "benchmark"),
    ),
    PlatformTopic(
        key="learning",
        title="在线学习",
        summary="说“在线学习”：收集人类决策→门禁短赛→不回归才发布给 AI 使用。",
        text=(
            "说“在线学习”“自动学习”或进顶部「在线学习」页：平台收集真实对局中人类的"
            "决策（按信息集聚合）→ 候选经验对手模型 → 与当前模型做门禁短赛"
            "（固定随机种子、双方换边、20 局）→ 不回归才发布（候选胜率不低于基线 −3%、"
            "样本 ≥30），失败保留旧版可回滚。\n"
            "德州扑克默认启用经验对手模型，发布后新开的对局 AI 自动使用；"
            "页面可手动 apply，服务端 --learning-interval N 可开后台自动发布。"
        ),
        keywords=("在线学习", "自动学习", "学习状态", "学习中心", "学习功能", "如何学习"),
    ),
    PlatformTopic(
        key="teaching",
        title="教学对局 / 教练",
        summary="开局开「教练」开关：教练能看你的牌并推理，走完对照参考动作点评。",
        text=(
            "开局时开启「教练」开关（或说“开一局教学对局”“教练模式”）：教练能看到"
            "**你自己的牌**并推理，像坐在你身后的教练——带你打、给你讲。\n"
            "· 开局教练开场说明教学局规则；每步走完对照“参考动作”点评（参考 = 求解器在"
            "**你的座位**上算的真实走法）；\n"
            "· 说“这步怎么走”升级为教练参考动作；“我现在该怎么听/打”可直接问教练；\n"
            "三条红线：教练看得不比你多（看不到 AI/对手的牌）、教练脑与对手脑互不相通、"
            "参考动作不污染在线学习样本。"
        ),
        keywords=("教学对局", "教练模式", "教学模式", "教练功能", "教练", "怎么教学"),
    ),
    PlatformTopic(
        key="llm",
        title="LLM 配置",
        summary="侧边栏「LLM 配置」填端点/模型/密钥（OpenAI 兼容）；保存即对聊天、翻译、社交 AI 生效。",
        text=(
            "侧边栏「LLM 配置」（/llm）：填端点 / 模型 / 密钥（OpenAI 兼容接口），"
            "默认本地 Ollama（http://127.0.0.1:11434，qwen3:8b）。\n"
            "保存写入 data/llm_config.json 并立即生效——平台聊天、规则翻译、社交类 AI 与"
            "陪伴对话同步切换；可先「测试连接」探测端点；密钥只写不回显（GET 只返回 "
            "has_api_key），清空即恢复环境变量/内置默认。\n"
            "等价环境变量：LLM_BASE_URL / LLM_MODEL / LLM_API_KEY。"
        ),
        keywords=("llm", "llm配置", "模型配置", "密钥", "大模型", "ai配置", "怎么配置模型"),
    ),
    PlatformTopic(
        key="vision",
        title="视觉识别",
        summary="独立视觉应用：截图/拍照识别棋盘或手牌观察（默认 8766 端口；P2 计划并入平台）。",
        text=(
            "视觉识别是独立的识别应用：上传截图/拍照 → AI 识别出棋盘布局或手牌观察 → "
            "接求解器给出可执行动作（识别结果不替代真实对局状态，仅作辅助）。\n"
            "启动：python -m layer4_interface.frontend.vision.server（默认 8766 端口）；"
            "走 DashScope 通义千问视觉模型（环境变量 DASHSCOPE_API_KEY / QWEN_BASE_URL / "
            "QWEN_MODEL=qwen-vl-plus）。P2 计划并入平台界面。"
        ),
        keywords=("视觉识别", "拍照识别", "图片识别", "截图识别", "摄像头识别", "识别功能"),
    ),
)

#: 稳定主题 key 顺序（工具参数 enum 与主题总览顺序）。
PLATFORM_TOPIC_KEYS: tuple[str, ...] = tuple(t.key for t in PLATFORM_TOPICS)

_KEY_TO_TOPIC: dict[str, PlatformTopic] = {t.key: t for t in PLATFORM_TOPICS}


def platform_help_text(topic: str) -> str:
    """按主题 key 返回权威帮助文档（未知/空 key → ``""``，调用方 fail-soft）。"""
    t = _KEY_TO_TOPIC.get(topic)
    if t is None:
        return ""
    text = f"{t.title}\n{t.text}"
    return text[:_HELP_TOPIC_MAX]


def platform_help_index() -> str:
    """主题总览：每个功能一行（标题 + 一句话简介），末尾提示怎么说触发。"""
    lines = ["平台功能总览（说“XX怎么用”可查某项的详细说明）:"]
    for t in PLATFORM_TOPICS:
        lines.append(f"· {t.title}：{t.summary}")
    body = "\n".join(lines)
    body = body[:_INDEX_MAX]
    return body


def match_platform_topic(text: str) -> str | None:
    """把一句话路由到最匹配的功能主题（最长关键词命中胜出）。

    大小写不敏感子串匹配，与 ``_find_game`` 的最长匹配策略一致：同一句
    命中多个主题时取关键词最长者（如「在线学习功能在哪」中 5 字的
    「在线学习功能」优先于 4 字的「在线学习」）。无命中返回 ``None``
    （调用方保持原有兜底，不猜）。
    """
    lowered = text.lower()
    best_key: str | None = None
    best_len = 0
    for t in PLATFORM_TOPICS:
        for kw in t.keywords:
            if kw in lowered and len(kw) > best_len:
                best_key = t.key
                best_len = len(kw)
    return best_key
