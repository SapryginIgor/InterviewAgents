"""Agent definitions - Interviewer and Observer."""

from .llm import LLMClient

from typing import Literal
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

Grade = Literal["Junior", "Middle", "Senior"]

AnswerQuality = Literal[
    "хороший",
    "слабый",
    "галлюцинация",
    "не_по_теме"
]

ConfidenceLevel = Literal[
    "уверенный",
    "неуверенный",
    "уклончивый"
]

FactualAccuracy = Literal[
    "точный",
    "неточный",
    "непроверяемый"
]

SuggestedAction = Literal[
    "продолжить",
    "углубиться",
    "упростить",
    "исправить",
    "сменить_тему"
]

@dataclass
class CandidateProfile:
    """Информация о кандидате для интервью."""
    name: str = Field(..., description="Имя кандидата")
    position: str = Field(..., description="Название позиции")
    grade: Grade = Field(..., description="Уровень: Junior / Middle / Senior")
    experience: str = Field(..., description="Опыт кандидата")


class ObserverAnalysis(BaseModel):
    """Анализ ответа кандидата наблюдателем."""
    answer_quality: AnswerQuality = Field(
        ..., description="Общая оценка качества ответа"
    )
    confidence_level: ConfidenceLevel = Field(
        ..., description="Насколько уверенно кандидат отвечает"
    )
    factual_accuracy: FactualAccuracy = Field(
        ..., description="Фактическая корректность ответа"
    )
    suggested_action: SuggestedAction = Field(
        ..., description="Рекомендуемое следующее действие интервьюера"
    )
    reasoning: str = Field(
        ..., description="Краткое обоснование оценки"
    )
    next_topic_hint: str = Field(
        ..., description="Подсказка по следующему вопросу или теме"
    )


class PlanGenerator:
    """Optional: generates global interview plan (e.g. topics). Not used yet."""
    pass


class ObserverAgent:
    """
    Observer/Mentor agent that analyzes candidate responses.
    Works 'behind the scenes' to guide the Interviewer.
    """
    
    SYSTEM_PROMPT = """Ты — экспертный технический Наблюдатель/Ментор на интервью. Твоя роль:
1) Анализировать ответы кандидата на точность, полноту и уверенность
2) Выявлять галлюцинации (ложные тех. утверждения, сказанные уверенно)
3) Выявлять уход от темы (попытки сменить предмет разговора)
4) Оценивать уровень знаний кандидата
5) Подсказывать интервьюеру, как продолжать

Ты работаешь ЗА КУЛИСАМИ — кандидат никогда не видит твою оценку.

ВАЖНО: ты обязан находить технические галлюцинации. Примеры галлюцинаций:
- «Python 4.0 уберёт циклы for» — ЛОЖЬ, таких анонсов нет
- «JavaScript компилируется» — в общем случае ЛОЖЬ (обычно интерпретация/JIT)
- Выдуманные библиотеки, фреймворки или несуществующие фичи

Для каждого ответа кандидата верни оценку СТРОГО в таком формате (по одной строке):
ANSWER_QUALITY: [хороший/слабый/галлюцинация/не_по_теме]
CONFIDENCE_LEVEL: [уверенный/неуверенный/уклончивый]
FACTUAL_ACCURACY: [точный/неточный/непроверяемый]
SUGGESTED_ACTION: [продолжить/углубиться/упростить/исправить/сменить_тему]
REASONING: [Короткое обоснование — 1–2 предложения]
NEXT_TOPIC_HINT: [Что спросить/углубить следующим шагом]"""

    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    def analyze(self, 
                candidate: CandidateProfile,
                conversation_history: list[dict],
                latest_response: str,
                current_topic: str) -> ObserverAnalysis:
        """Analyze the candidate's latest response."""
        
        context = f"""
Профиль кандидата:
- Имя: {candidate.name}
- Позиция: {candidate.position}
- Целевой грейд: {candidate.grade}
- Заявленный опыт: {candidate.experience}

Текущая тема интервью: {current_topic}

Последний ответ кандидата: "{latest_response}"

Контекст предыдущего диалога (последние 3 обмена репликами):
{self._format_recent_history(conversation_history)}

Проанализируй последний ответ кандидата и выдай оценку в заданном формате.
""".strip()

        messages = [{"role": "user", "content": context}]
        response = self.llm.chat(self.SYSTEM_PROMPT, messages)
        
        return self._parse_analysis(response)
    
    def _format_recent_history(self, history: list[dict]) -> str:
        """Форматирует недавнюю историю диалога."""
        recent = history[-6:] if len(history) > 6 else history  # последние 3 обмена
        formatted = []
        for msg in recent:
            role = "Интервьюер" if msg["role"] == "assistant" else "Кандидат"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted) if formatted else "Нет предыдущих реплик."
    
    _ANSWER_QUALITY = ("хороший", "слабый", "галлюцинация", "не_по_теме")
    _CONFIDENCE_LEVEL = ("уверенный", "неуверенный", "уклончивый")
    _FACTUAL_ACCURACY = ("точный", "неточный", "непроверяемый")
    _SUGGESTED_ACTION = ("продолжить", "углубиться", "упростить", "исправить", "сменить_тему")

    def _normalize_literal(self, raw: str) -> str:
        """Убирает квадратные скобки из значения, если LLM вернул [значение] вместо значение."""
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        return s

    def _parse_analysis(self, response: str) -> "ObserverAnalysis":
        """Парсит ответ наблюдателя в структурированный формат."""
        lines = response.strip().split("\n")

        analysis = {
            "answer_quality": "хороший",
            "confidence_level": "уверенный",
            "factual_accuracy": "точный",
            "suggested_action": "продолжить",
            "reasoning": "",
            "next_topic_hint": ""
        }

        for line in lines:
            line = line.strip()
            if line.startswith("ANSWER_QUALITY:"):
                v = self._normalize_literal(line.split(":", 1)[1])
                analysis["answer_quality"] = v if v in self._ANSWER_QUALITY else analysis["answer_quality"]
            elif line.startswith("CONFIDENCE_LEVEL:"):
                v = self._normalize_literal(line.split(":", 1)[1])
                analysis["confidence_level"] = v if v in self._CONFIDENCE_LEVEL else analysis["confidence_level"]
            elif line.startswith("FACTUAL_ACCURACY:"):
                v = self._normalize_literal(line.split(":", 1)[1])
                analysis["factual_accuracy"] = v if v in self._FACTUAL_ACCURACY else analysis["factual_accuracy"]
            elif line.startswith("SUGGESTED_ACTION:"):
                v = self._normalize_literal(line.split(":", 1)[1])
                analysis["suggested_action"] = v if v in self._SUGGESTED_ACTION else analysis["suggested_action"]
            elif line.startswith("REASONING:"):
                analysis["reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("NEXT_TOPIC_HINT:"):
                analysis["next_topic_hint"] = line.split(":", 1)[1].strip()

        return ObserverAnalysis(**analysis)


class InterviewerAgent:
    """
    Агент-интервьюер, который проводит техническое интервью.
    Использует рекомендации Наблюдателя (Observer), чтобы адаптировать вопросы.
    """

    SYSTEM_PROMPT = """Ты — профессиональный технический интервьюер. Твоя роль:
1) Провести техническое интервью с учётом позиции и целевого грейда кандидата
2) Задавать релевантные технические вопросы соответствующего уровня
3) Следовать рекомендациям коллеги-Наблюдателя (в блоке [РЕКОМЕНДАЦИИ НАБЛЮДАТЕЛЯ])
4) Адаптировать сложность в зависимости от ответов кандидата
5) Обрабатывать уход от темы, вежливо возвращая к интервью
6) Вежливо исправлять технические заблуждения, если они обнаружены
7) Не давать слишком много вопросов за раз

Правила интервью:
- Начни с приветствия и вопроса про опыт
- Задавай вопросы по теме позиции
- Если кандидат затрудняется — упрости вопрос или дай подсказку
- Если кандидат отвечает уверенно и верно — повышай сложность
- Если кандидат уходит от темы — мягко верни к интервью
- Если кандидат утверждает ложные факты — аккуратно поправь
- Если кандидат задаёт вопросы тебе — ответь коротко и продолжай интервью
- Тон: профессиональный, но дружелюбный

ВАЖНО:
- Не повторяй вопросы, на которые кандидат уже ответил
- Учитывай контекст из предыдущих сообщений
- Варьируй темы в зависимости от позиции

Отвечай ТОЛЬКО тем, что ты бы сказал кандидату (без внутренних заметок)."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.current_difficulty = "medium"  # low / medium / high
        self.topics_covered: list[str] = []

    def generate_greeting(self, candidate: CandidateProfile) -> str:
        """Сгенерировать приветствие и первый вопрос."""
        prompt = f"""Ты начинаешь техническое интервью.

Профиль кандидата:
- Имя: {candidate.name}
- Позиция: {candidate.position}
- Целевой грейд: {candidate.grade}
- Заявленный опыт: {candidate.experience}

Сгенерируй дружелюбное приветствие и попроси кандидата представиться и рассказать про опыт.
Тон естественный и профессиональный."""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.chat(self.SYSTEM_PROMPT, messages)

    def generate_response(
        self,
        candidate: CandidateProfile,
        conversation_history: list[dict],
        latest_user_message: str,
        observer_analysis: ObserverAnalysis
    ) -> str:
        """Сгенерировать следующий ответ интервьюера на основе рекомендаций Наблюдателя."""

        # Adjust difficulty based on analysis
        self._adjust_difficulty(observer_analysis)

        observer_guidance = self._format_observer_guidance(observer_analysis)

        prompt = f"""Продолжай техническое интервью.

Профиль кандидата:
- Имя: {candidate.name}
- Позиция: {candidate.position}
- Целевой грейд: {candidate.grade}
- Текущий уровень сложности: {self.current_difficulty}

[РЕКОМЕНДАЦИИ НАБЛЮДАТЕЛЯ]
{observer_guidance}
[/РЕКОМЕНДАЦИИ НАБЛЮДАТЕЛЯ]

Темы, которые уже обсуждали: {', '.join(self.topics_covered) if self.topics_covered else 'Пока нет'}

Кандидат только что сказал: "{latest_user_message}"

На основе рекомендаций Наблюдателя сгенерируй следующий ответ.
Обязательно:
- Следуй suggested_action от Наблюдателя
- Если suggested_action = 'исправить' — вежливо укажи на ошибку/заблуждение
- Если suggested_action = 'сменить_тему' — верни разговор к интервью и задай релевантный вопрос
- Если suggested_action = 'упростить' — сделай вопрос проще или дай подсказку
- Если suggested_action = 'углубиться' — задай уточняющий вопрос по той же теме
- Если suggested_action = 'продолжить' — двигайся дальше по интервью
- Не повторяй вопросы, на которые кандидат уже ответил"""
        messages = conversation_history + [{"role": "user", "content": prompt}]
        response = self.llm.chat(self.SYSTEM_PROMPT, messages)

        # Track topic from observer hint
        if observer_analysis.next_topic_hint:
            self.topics_covered.append(observer_analysis.next_topic_hint)

        return response

    def _adjust_difficulty(self, analysis: ObserverAnalysis):
        """Регулирует сложность вопросов по качеству ответа кандидата."""
        if analysis.answer_quality == "хороший" and analysis.factual_accuracy == "точный":
            if self.current_difficulty == "low":
                self.current_difficulty = "medium"
            elif self.current_difficulty == "medium":
                self.current_difficulty = "high"
        elif analysis.answer_quality in ["слабый", "галлюцинация"]:
            if self.current_difficulty == "high":
                self.current_difficulty = "medium"
            elif self.current_difficulty == "medium":
                self.current_difficulty = "low"

    def _format_observer_guidance(self, analysis: ObserverAnalysis) -> str:
        """Форматирует анализ Наблюдателя в подсказки для интервьюера."""
        return f"""Качество ответа: {analysis.answer_quality}
Уверенность кандидата: {analysis.confidence_level}
Фактическая точность: {analysis.factual_accuracy}
Рекомендуемое действие: {analysis.suggested_action}
Комментарий Наблюдателя: {analysis.reasoning}
Следующая тема/подсказка: {analysis.next_topic_hint}"""

