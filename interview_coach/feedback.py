"""Feedback generator - creates structured interview feedback."""

from typing import Literal
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from .llm import LLMClient
from .agents import CandidateProfile

Grade = Literal["Junior", "Middle", "Senior"]

HiringRecommendation = Literal["Точно нанимаем", "Нанимаем", "Не нанимаем"]

ClarityAssessment = Literal["ясно", "средне", "неясно"]
HonestyAssessment = Literal["честно", "иногда_уходит", "блефует"]
EngagementAssessment = Literal["вовлечён", "нейтрален", "пассивен"]


class KnowledgeGap(BaseModel):
    """Информация о пробеле в знаниях кандидата."""
    topic: str = Field(..., description="Тема, в которой обнаружен пробел")
    correct_answer: str = Field(..., description="Правильный ответ или объяснение")


@dataclass
class InterviewFeedback:
    """Структурированная обратная связь по интервью."""
    # Decision
    assessed_grade: Grade = Field(..., description="Оцененный уровень кандидата: Junior / Middle / Senior")
    hiring_recommendation: HiringRecommendation = Field(..., description="Рекомендация по найму")
    confidence_score: int = Field(..., description="Уровень уверенности в оценке (0-100)", ge=0, le=100)
    
    # Technical Review
    confirmed_skills: list[str] = Field(default_factory=list, description="Подтверждённые навыки и знания кандидата")
    knowledge_gaps: list[KnowledgeGap] = Field(default_factory=list, description="Пробелы в знаниях кандидата")
    
    # Soft Skills
    clarity_assessment: ClarityAssessment = Field(..., description="Оценка ясности коммуникации")
    honesty_assessment: HonestyAssessment = Field(..., description="Оценка честности кандидата")
    engagement_assessment: EngagementAssessment = Field(..., description="Оценка вовлечённости кандидата")
    
    # Roadmap
    study_recommendations: list[str] = Field(default_factory=list, description="Рекомендации по изучению")
    
    raw_feedback: str = Field(..., description="Полный текстовый вариант обратной связи")


class FeedbackGenerator:
    """Generates structured feedback from interview session."""
    
    SYSTEM_PROMPT = """Ты — эксперт по оценке технических интервью. На основе транскрипта интервью
сгенерируй подробный, структурированный отчёт с обратной связью.

Твой отчёт ОБЯЗАН включать ВСЕ секции ниже и использовать указанные значения.

## 1. РЕШЕНИЕ
- ASSESSED_GRADE: [Junior / Middle / Senior] — по фактически показанным знаниям, а не по заявленному грейду
- HIRING_RECOMMENDATION: [Точно нанимаем / Нанимаем / Не нанимаем]
- CONFIDENCE_SCORE: [0-100]% — насколько ты уверен в оценке

## 2. ТЕХНИЧЕСКИЙ РАЗБОР

### Подтверждённые навыки (где кандидат показал уверенное понимание):
- перечисли темы/навыки

### Пробелы в знаниях (где кандидат ошибался или путался):
- для каждого пункта укажи:
  - Topic: [тема]
  - Issue: [что было не так]
  - Correct Answer: [как правильно / что нужно было знать]

## 3. SOFT SKILLS
- CLARITY: [ясно / средне / неясно]
- HONESTY: [честно / иногда_уходит / блефует]
- ENGAGEMENT: [вовлечён / нейтрален / пассивен]

## 4. ПЕРСОНАЛЬНЫЙ ПЛАН РАЗВИТИЯ
Дай конкретные и применимые рекомендации по обучению на основе выявленных пробелов:
- пункт 1: [что именно изучить/повторить]
- пункт 2: [что именно изучить/повторить]
...

Будь конкретным и практичным: так, чтобы кандидат мог улучшиться."""


    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    def generate(self, 
                 candidate: CandidateProfile,
                 conversation_transcript: str,
                 internal_thoughts_summary: str) -> InterviewFeedback:
        """Generate comprehensive interview feedback."""
        
        prompt = f"""Generate a detailed interview feedback report.

CANDIDATE PROFILE:
- Name: {candidate.name}
- Position: {candidate.position}
- Claimed Grade: {candidate.grade}
- Stated Experience: {candidate.experience}

INTERVIEW TRANSCRIPT:
{conversation_transcript}

OBSERVER NOTES (internal analysis during interview):
{internal_thoughts_summary}

Generate the complete structured feedback report following the required format."""

        messages = [{"role": "user", "content": prompt}]
        raw_feedback = self.llm.chat(self.SYSTEM_PROMPT, messages)
        
        return self._parse_feedback(raw_feedback)
    
    def _parse_feedback(self, raw: str) -> InterviewFeedback:
        """Parse raw feedback into structured format."""
        # Default values
        feedback = InterviewFeedback(
            assessed_grade="Junior",
            hiring_recommendation="Не нанимаем",
            confidence_score=50,
            confirmed_skills=[],
            knowledge_gaps=[],
            clarity_assessment="средне",
            honesty_assessment="честно",
            engagement_assessment="нейтрален",
            study_recommendations=[],
            raw_feedback=raw
        )
        
        lines = raw.split("\n")
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Parse decision section
            if "assessed_grade:" in line_lower:
                grade = line_stripped.split(":", 1)[1].strip()
                if "senior" in grade.lower():
                    feedback.assessed_grade = "Senior"
                elif "middle" in grade.lower():
                    feedback.assessed_grade = "Middle"
                else:
                    feedback.assessed_grade = "Junior"
            
            elif "hiring_recommendation:" in line_lower:
                rec = line_stripped.split(":", 1)[1].strip().lower()
                if "точно нанимаем" in rec:
                    feedback.hiring_recommendation = "Точно нанимаем"
                elif "не нанимаем" in rec:
                    feedback.hiring_recommendation = "Не нанимаем"
                elif "нанимаем" in rec:
                    feedback.hiring_recommendation = "Нанимаем"
                else:
                    feedback.hiring_recommendation = "Не нанимаем"
            
            elif "confidence_score:" in line_lower:
                try:
                    score_str = line_stripped.split(":", 1)[1].strip()
                    score = int(''.join(filter(str.isdigit, score_str)))
                    feedback.confidence_score = min(100, max(0, score))
                except:
                    pass
            
            # Parse sections
            elif "confirmed skills" in line_lower:
                current_section = "confirmed"
            elif "knowledge gaps" in line_lower:
                current_section = "gaps"
            elif "personal roadmap" in line_lower or "study" in line_lower:
                current_section = "roadmap"
            
            # Parse soft skills
            elif "clarity:" in line_lower:
                feedback.clarity_assessment = line_stripped.split(":", 1)[1].strip()
            elif "honesty:" in line_lower:
                feedback.honesty_assessment = line_stripped.split(":", 1)[1].strip()
            elif "engagement:" in line_lower:
                feedback.engagement_assessment = line_stripped.split(":", 1)[1].strip()
            
            # Parse list items
            elif line_stripped.startswith("- ") or line_stripped.startswith("* "):
                item = line_stripped[2:].strip()
                if current_section == "confirmed" and item:
                    feedback.confirmed_skills.append(item)
                elif current_section == "roadmap" and item:
                    feedback.study_recommendations.append(item)
                elif current_section == "gaps" and item:
                    # Simple gap parsing
                    feedback.knowledge_gaps.append(
                        KnowledgeGap(
                            topic=item,
                            correct_answer="См. подробную обратную связь"
                        )
                    )
        
        return feedback
    
    def format_feedback_text(self, feedback: InterviewFeedback) -> str:
        sections = []

        sections.append("=" * 60)
        sections.append("📋 ОТЧЁТ ПО ИТОГАМ ИНТЕРВЬЮ")
        sections.append("=" * 60)
        sections.append("")

        sections.append("## 1) РЕШЕНИЕ")
        sections.append(f"• Оценённый грейд: {feedback.assessed_grade}")
        sections.append(f"• Рекомендация по найму: {feedback.hiring_recommendation}")
        sections.append(f"• Уверенность: {feedback.confidence_score}%")
        sections.append("")

        sections.append("## 2) ТЕХНИЧЕСКИЙ РАЗБОР")
        sections.append("")
        sections.append("✅ Подтверждённые навыки:")
        if feedback.confirmed_skills:
            for skill in feedback.confirmed_skills:
                sections.append(f"  • {skill}")
        else:
            sections.append("  • Не выделено")
        sections.append("")

        sections.append("❌ Пробелы в знаниях:")
        if feedback.knowledge_gaps:
            for gap in feedback.knowledge_gaps:
                sections.append(f"  • {gap.topic}")
        else:
            sections.append("  • Не выделено")
        sections.append("")

        sections.append("## 3) ГИБКИЕ НАВЫКИ")
        sections.append(f"• Ясность: {feedback.clarity_assessment}")
        sections.append(f"• Честность: {feedback.honesty_assessment}")
        sections.append(f"• Вовлечённость: {feedback.engagement_assessment}")
        sections.append("")

        sections.append("## 4) ПЕРСОНАЛЬНЫЙ ПЛАН РАЗВИТИЯ")
        if feedback.study_recommendations:
            for rec in feedback.study_recommendations:
                sections.append(f"  • {rec}")
        else:
            sections.append("  • Продолжать текущий план обучения")
        sections.append("")
        sections.append("=" * 60)

        return "\n".join(sections)

