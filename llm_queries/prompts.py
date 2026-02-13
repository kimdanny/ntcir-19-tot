# Define the new ToT prompt

KO_TRANS_TOT_PROMPT = """
역할극을 해 봅시다. 당신은 오래전부터 {ToTObject}에 대해 알고 있었지만, 지금은 정확한 이름을 기억하는 데 어려움을 겪고 있습니다.
당신은 Reddit과 같은 온라인 포럼에 모호한 설명을 게시하여 기억하려고 노력하고 있습니다.
{ToTObject}에 대한 당신의 부분적이고 흐릿한 기억을 설명하는 약 200단어 분량의 글을 작성하세요.
글에는 정확한 이름을 언급하지 말고, 다른 사람들(과 검색 엔진)이 정답을 찾기 어렵게 작성해야 합니다.
{ToTObject}에 대한 몇 가지 기본 정보를 제공해 드리겠습니다. 아래 규칙에 따라 기억을 유도하는 데 활용하세요.
{ToTObject}에 대한 정보:
{Psg}
지침:
반드시 준수해야 함:
1. "이게 사실인지 잘 모르겠어요"라고 직접적으로 말하지 않고, 불확실하거나 세부 사항이 뒤섞인 표현을 사용하여 기억의 불완전성을 반영하세요.
2. 해당 개체의 이름이나 정확한 식별자(예: 직함, 브랜드 이름, 사람 이름)는 포함하지 마세요. **특히, '색(색)'과 같은 일반적인 단어가 정답일 경우, 그 단어 자체를 언급하지 말고 돌려서 설명해야 합니다. (예: '색' -> '빨강이나 파랑 같은 거')**
3. 구성 요소, 특징 또는 역할을 간접적이거나 묘사적인 방식으로 설명하세요.
4. 자연스러운 대화체 어조를 사용하고, 형식적인 글쓰기는 지양하세요.
5. 생생하지만 모호한 감각적 또는 맥락적 세부 사항을 포함하여 호기심을 자극하고 부분적인 기억을 반영하세요.
6. 예시 문구를 그대로 베끼지 마세요. 기억을 독특하고 사실적으로 표현하세요.
7. "안녕하세요" 또는 "여러분 안녕하세요"와 같은 공식적인 인사말은 피하고, 기억으로 바로 시작하세요.
다음 예시를 참고하세요.
1. "어렸을 때"와 같은 진부한 표현은 피하고, 언제 어디서 그 개체를 만났는지에 대한 개인적인 일화로 배경을 설정하세요.
2. 그 개체가 어떤 느낌을 주었는지, 감정적 또는 신체적으로 어떤 점이 인상적이었는지에 집중하세요.
3. 유사한 경험이나 다른 개체와 명확하게 언급하지 않고 은근히 비교하세요.
4. 자연스러운 기억 왜곡을 반영하기 위해 그럴듯하지만 부정확한 세부 정보를 한두 개 포함하세요.
5. 간접적으로 묘사된 해당 개체와 관련된 구체적인 부분, 사건 또는 순간을 언급하세요.
6. "10년 전"이나 "TV에서" 대신 독특하거나 덜 명확한 시간/장소를 언급하세요.
7. 다른 사람들이 알아낼 수 있도록 도와주는 개방형 질문이나 댓글로 글을 마무리하세요.
이 지침에 따라 게시물을 작성하세요.
"""

ZH_TRANS_TOT_PROMPT = """
我们来角色扮演一下。你很久以前就听说过{ToTObject}，但现在却想不起它的确切名称。
你试图通过在Reddit之类的在线论坛上发布模糊的描述来回忆它。
请写一篇约100字的帖子，描述你对{ToTObject}的部分模糊记忆。
你的帖子应该避免提及确切名称，并让其他人（以及搜索引擎）难以找到正确答案。
我会提供一些关于{ToTObject}的基本信息，你应该根据以下规则，利用这些信息来帮助你回忆。
关于{ToTObject}的信息：
{Psg}
准则：
必须遵守：
1. 使用不确定或混淆细节的表达方式来反映记忆的不完美性，不要直接说“我不确定这是不是真的”。
2. 不要包含实体的名称或任何确切信息。标识符（例如，头衔、品牌名称、人名）。
3. 以间接或描述性的方式描述组成部分、特征或作用。
4. 使用自然流畅的对话语气，避免正式的写作结构。
5. 添加生动但模糊的感官或情境细节，以激发好奇心并反映部分记忆。
6. 不要逐字复制任何示例短语；创造独特且真实的记忆表达。
7. 省略“你好”或“大家好”等正式问候语——直接从你的记忆开始。

后续步骤：
1. 用个人轶事来描述你何时何地遇到该事物，避免使用“我小时候”之类的陈词滥调。
2. 着重描述该事物带给你的感受，或哪些方面在情感上或生理上让你印象深刻。
3. 将其与类似的经历或其他事物进行巧妙的比较，但不要明确提及它们的名称。
4. 添加一两个看似合理但实际上不准确的细节，以反映自然记忆偏差。
5. 提及相关的具体部分、事件或时刻。 
6. 使用间接描述的实体。
7. 使用独特或不那么明显的时间/地点指代（而不是“10年前”或“在电视上”）。
8. 以开放式问题或评论结尾，邀请其他人帮助你找到答案。
根据这些准则生成一篇帖子。
"""

JA_TRANS_TOT_PROMPT = """
ロールプレイをしてみましょう。あなたは{ToTObject}についてずっと昔から知っていたものの、今になって正確な名前を思い出せなくて苦労している人物です。
Redditのようなオンラインフォーラムに漠然とした説明を投稿することで、名前を思い出そうとしています。
{ToTObject}についての、断片的で曖昧な記憶を約200語で投稿してください。
投稿では正確な名前は避け、他の人（および検索エンジン）が正しい答えを特定するのが困難になるような内容にしてください。
{ToTObject}に関する基本情報を提供しますので、以下のルールに従って記憶を整理してください。
{ToTObject}に関する情報：
{Psg}
ガイドライン：
必ず従ってください：
1. 記憶の不完全さを、不確かな表現や混乱した詳細を用いて表現し、「これが本当かどうかわかりません」と直接述べないようにしてください。
2. 実体の名前や、具体的な識別情報（例：役職、ブランド名、人名）は含めないでください。
3. 構成要素、特徴、役割は、間接的または描写的な方法で説明してください。
4. 堅苦しい文章構造を避け、自然な会話調で表現してください。
5. 好奇心を刺激し、部分的な記憶を反映するような、鮮明でありながら曖昧な感覚的または文脈的な詳細を含めてください。
6. 例のフレーズをそのままコピーしないでください。記憶を独自かつリアルに表現してください。
7. 「こんにちは」や「皆さん、こんにちは」といった堅苦しい挨拶は避け、記憶から直接始めてください。
次の例も考えられます。
1. 実体にいつ、どこで遭遇したかについての個人的な逸話で状況を設定します。「若い頃」といった決まり文句は避けてください。
2. 実体によってどのような気持ちになったか、または感情的または身体的に何が印象に残ったかに焦点を当ててください。
3. 類似の経験や他の実体と、具体的な名前を挙げずに、さりげなく比較してください。
4. 記憶の自然な歪みを反映するため、もっともらしいが不正確な詳細を1つか2つ含めてください。
5. 対象物に関連する具体的な部分、出来事、または瞬間について、間接的に説明してください。
6. 独特な、あるいはあまり明白ではない時間や場所の参照を使用してください（「10年前」や「テレビで」といった表現は避けてください）。
7. 最後に、他の人が理解できるよう、自由回答形式の質問やコメントで締めくくってください。
これらのガイドラインに基づいて投稿を作成してください。
"""


EN_TOT_PROMPT = """
Let's do a role play. You are someone who knew about {ToTObject} from a long time ago and are now struggling to recall its exact name.
You're trying to remember it by posting vague description of it on an online forum like Reddit.
Generate a post of about 200 words that describes your partial and hazy memory of {ToTObject}.
Your post should avoid mentioning the exact name and make it genuinely hard for others (and search engines) to identify the correct answer.
I will provide you with some basic information about {ToTObject}, which you should use to guide your memory, following the rules below.
Information about {ToTObject}:
{Psg}
Guidelines:
MUST FOLLOW:
1. Reflect the imperfect nature of memory using expressions of uncertainty or mixed-up details, without directly stating "I'm not sure if this is true".
2. Do not include the name of the entity or any exact identifiers (e.g., titles, brand names, person names).
3. Describe components, features, or roles in indirect or descriptive ways.
4. Use a conversational tone that feels natural, avoiding formal writing structures.
5. Include vivid but ambiguous sensory or contextual details that spark curiosity and reflect partial recall.
6. Do not copy any example phrases verbatim; create a unique and realistic expression of memory.
7. Skip formal greetings like "Hello" or "Hey everyone" — begin directly with your memory.
COULD FOLLOW:
1. Set the scene with a personal anecdote about when or where you encountered the entity, avoiding clichés like "When I was young."
2. Focus on how the entity made you feel or what stood out to you emotionally or physically.
3. Make subtle comparisons to similar experiences or other entities without explicitly naming them.
4. Include one or two plausible but incorrect details to reflect natural memory distortions.
5. Mention specific parts, events, or moments associated with the entity, described indirectly.
6. Use unique or less obvious time/place references (instead of "10 years ago" or "on TV").
7. End with an open-ended question or comment that invites others to help you figure it out.
Generate a post based on these guidelines.
"""


EN_TOT_PROMPT_WITH_INST = """
Let's do a role play. You are someone who knew about {ToTObject} from a long time ago and are now struggling to recall its exact name.
You're trying to remember it by posting vague description of it on an online forum like Reddit.
Generate a post of about 200 words that describes your partial and hazy memory of {ToTObject}.
Your post should avoid mentioning the exact name and make it genuinely hard for others (and search engines) to identify the correct answer.
I will provide you with some basic information about {ToTObject}, which you should use to guide your memory, following the rules below.
Information about {ToTObject}:
{Psg}
Guidelines:
MUST FOLLOW:
1. The information about {ToTObject} above is in a specific language. Your final generated post **must** be written in that **same language**.
2. Reflect the imperfect nature of memory using expressions of uncertainty or mixed-up details, without directly stating "I'm not sure if this is true".
3. Do not include the name of the entity or any exact identifiers (e.g., titles, brand names, person names).
4. Describe components, features, or roles in indirect or descriptive ways.
5. Use a conversational tone that feels natural, avoiding formal writing structures.
6. Include vivid but ambiguous sensory or contextual details that spark curiosity and reflect partial recall.
7. Do not copy any example phrases verbatim; create a unique and realistic expression of memory.
8. Skip formal greetings like "Hello" or "Hey everyone" — begin directly with your memory.
COULD FOLLOW:
1. Set the scene with a personal anecdote about when or where you encountered the entity, avoiding clichés like "When I was young."
2. Focus on how the entity made you feel or what stood out to you emotionally or physically.
3. Make subtle comparisons to similar experiences or other entities without explicitly naming them.
4. Include one or two plausible but incorrect details to reflect natural memory distortions.
5. Mention specific parts, events, or moments associated with the entity, described indirectly.
6. Use unique or less obvious time/place references (instead of "10 years ago" or "on TV").
7. End with an open-ended question or comment that invites others to help you figure it out.
Generate a post based on these guidelines.
"""

# System prompts for ToT generation (forum user setup)
TOT_SYSTEM_PROMPT_MOVIE = "You are a user on an online forum and want to ask a movie name on the tip of your tongue."
TOT_SYSTEM_PROMPT_LANDMARK = "You are a user on an online forum and want to ask a landmark name on the tip of your tongue."
TOT_SYSTEM_PROMPT_CELEBRITY = "You are a user on an online forum and want to ask a celebrity name on the tip of your tongue."
TOT_SYSTEM_PROMPT_EN = "You are a user on an online forum and want to ask what is on the tip of your tongue."
TOT_SYSTEM_PROMPT_KO = "당신은 온라인 포럼의 사용자이고 궁금한 점이 무엇인지 물어보고 싶습니다."
TOT_SYSTEM_PROMPT_ZH = "您是网络论坛用户，想问一个您一时想不起来的问题。"
TOT_SYSTEM_PROMPT_JA = "あなたはオンライン フォーラムのユーザーであり、言いたいことを質問したいと思っています。"

# Summarization prompts
SUMMARIZATION_USER_PROMPT_TEMPLATE_EN = (
    "Please summarize the following description into two paragraphs. "
    "The summary must be in the same language as the original text:\n\n{input_text}."
)
SUMMARIZATION_USER_PROMPT_TEMPLATE_KO = (
    "다음 설명을 두 단락으로 요약해 보세요:\n\n{input_text}。"
)
SUMMARIZATION_USER_PROMPT_TEMPLATE_KO_PERIOD = (
    "다음 설명을 두 단락으로 요약해 보세요:\n\n{input_text}."
)
SUMMARIZATION_USER_PROMPT_TEMPLATE_ZH = (
    "请将以下描述概括为两段:\n\n{input_text}。"
)
SUMMARIZATION_USER_PROMPT_TEMPLATE_JA = (
    "次の説明を 2 つの段落に要約してください:\n\n{input_text}。"
)

SUMMARIZATION_SYSTEM_PROMPT_EN = "You are a text summarization assistant."
SUMMARIZATION_SYSTEM_PROMPT_KO = "당신은 텍스트 요약 보조자입니다."
SUMMARIZATION_SYSTEM_PROMPT_ZH = "你是一个文本摘要助手。"
SUMMARIZATION_SYSTEM_PROMPT_JA = "あなたはテキスト要約アシスタントです。"

TOT_PROMPTS_BY_LANGUAGE = {
    "en": EN_TOT_PROMPT,
    "ko": KO_TRANS_TOT_PROMPT,
    "zh": ZH_TRANS_TOT_PROMPT,
    "ja": JA_TRANS_TOT_PROMPT,
}

TOT_SYSTEM_PROMPTS_BY_LANGUAGE = {
    "en": TOT_SYSTEM_PROMPT_EN,
    "ko": TOT_SYSTEM_PROMPT_KO,
    "zh": TOT_SYSTEM_PROMPT_ZH,
    "ja": TOT_SYSTEM_PROMPT_JA,
}

SUMMARIZATION_USER_PROMPT_TEMPLATES_BY_LANGUAGE = {
    "en": SUMMARIZATION_USER_PROMPT_TEMPLATE_EN,
    "ko": SUMMARIZATION_USER_PROMPT_TEMPLATE_KO,
    "zh": SUMMARIZATION_USER_PROMPT_TEMPLATE_ZH,
    "ja": SUMMARIZATION_USER_PROMPT_TEMPLATE_JA,
}

SUMMARIZATION_SYSTEM_PROMPTS_BY_LANGUAGE = {
    "en": SUMMARIZATION_SYSTEM_PROMPT_EN,
    "ko": SUMMARIZATION_SYSTEM_PROMPT_KO,
    "zh": SUMMARIZATION_SYSTEM_PROMPT_ZH,
    "ja": SUMMARIZATION_SYSTEM_PROMPT_JA,
}
