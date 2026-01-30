"""LangGraph-based interview pipeline."""

from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator
from langgraph.graph import StateGraph, START, END

from .agents import (
    CandidateProfile,
    ObserverAnalysis,
    ObserverAgent,
    InterviewerAgent,
)
from .feedback import FeedbackGenerator
from .logger import InterviewLogger
from .llm import LLMClient


# ----- State schema (Pydantic) -----


def _get(state: Any, key: str, default: Any = None) -> Any:
    """Get attribute from state (dict or model)."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


class InterviewState(BaseModel):
    """State shared across all graph nodes."""

    candidate: Union[CandidateProfile, dict] = Field(
        ..., description="Профиль кандидата"
    )
    history: list[dict] = Field(
        default_factory=list,
        description="История диалога для LLM (role/content)",
    )
    current_topic: str = Field(
        default="Introduction",
        description="Текущая тема интервью",
    )
    internal_thoughts: list[str] = Field(
        default_factory=list,
        description="Внутренние заметки наблюдателя по ходу интервью",
    )
    user_message: str = Field(
        default="",
        description="Текущее сообщение пользователя",
    )
    observer_analysis: Optional[ObserverAnalysis] = Field(
        default=None,
        description="Анализ ответа кандидата наблюдателем",
    )
    interviewer_response: str = Field(
        default="",
        description="Последний ответ интервьюера пользователю",
    )
    final_feedback: Optional[str] = Field(
        default=None,
        description="Итоговая обратная связь по интервью",
    )
    is_stop: bool = Field(
        default=False,
        description="Пользователь запросил завершение",
    )
    first_turn: bool = Field(
        default=True,
        description="Первый вызов графа (приветствие)",
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("candidate", mode="before")
    @classmethod
    def _coerce_candidate(cls, v: Any) -> CandidateProfile:
        if isinstance(v, CandidateProfile):
            return v
        if isinstance(v, dict):
            return CandidateProfile(**v)
        raise ValueError("candidate must be CandidateProfile or dict")


# ----- Router node (no state change, used for conditional edges) -----


def _route_after_start(state: Any) -> str:
    """Route from entry: first turn -> greet, else -> process user message."""
    if _get(state, "first_turn", True):
        return "greet_and_get_candidates_description"
    return "route_user_message"


def _route_after_user_message(state: Any) -> str:
    """Route after route_user_message: stop -> summarize, else -> observer."""
    if _get(state, "is_stop", False):
        return "summarize"
    return "send_answer_to_observer"


# ----- Node implementations -----


def _make_greet_node(interviewer: InterviewerAgent, logger: InterviewLogger):
    def greet_and_get_candidates_description(state: InterviewState) -> dict:
        greeting = interviewer.generate_greeting(state.candidate)
        initial_thoughts = (
            f"[Observer]: New interview starting. Candidate: {state.candidate.name}, "
            f"Position: {state.candidate.position}, Target Grade: {state.candidate.grade}. "
            "[Interviewer]: Beginning with introduction and experience questions."
        )
        logger.log_agent_response(greeting, initial_thoughts)
        return {
            "interviewer_response": greeting,
            "first_turn": False,
            "internal_thoughts": state.internal_thoughts + [initial_thoughts],
        }

    return greet_and_get_candidates_description


def _make_route_user_message_node(logger: InterviewLogger):
    STOP_COMMANDS = [
        "стоп",
        "stop",
        "стоп интервью",
        "stop interview",
        "стоп игра",
        "давай фидбэк",
    ]

    def route_user_message(state: InterviewState) -> dict:
        message_lower = state.user_message.lower().strip()
        is_stop = any(cmd in message_lower for cmd in STOP_COMMANDS)
        logger.log_user_message(state.user_message)
        return {"is_stop": is_stop}

    return route_user_message


def _make_send_answer_to_observer_node(observer: ObserverAgent):
    def send_answer_to_observer(state: InterviewState) -> dict:
        analysis = observer.analyze(
            candidate=state.candidate,
            conversation_history=state.history,
            latest_response=state.user_message,
            current_topic=state.current_topic,
        )
        new_topic = (
            analysis.next_topic_hint
            if analysis.next_topic_hint
            else state.current_topic
        )
        thoughts = (
            f"[Observer]: Answer quality: {analysis.answer_quality}. "
            f"Confidence: {analysis.confidence_level}. "
            f"Factual accuracy: {analysis.factual_accuracy}. "
            f"Analysis: {analysis.reasoning} "
            f"[Interviewer]: Following suggestion to '{analysis.suggested_action}'. "
            f"Next topic: {analysis.next_topic_hint}"
        )
        return {
            "observer_analysis": analysis,
            "current_topic": new_topic,
            "internal_thoughts": state.internal_thoughts + [thoughts],
        }

    return send_answer_to_observer


def _make_send_hints_to_interviewer_node():
    """Pass-through: observer_analysis is already in state."""

    def send_hints_to_interviewer(state: InterviewState) -> dict:
        return {}

    return send_hints_to_interviewer


def _make_interviewer_response_node(
    interviewer: InterviewerAgent, logger: InterviewLogger
):
    def interviewer_response_or_question(state: InterviewState) -> dict:
        if state.observer_analysis is None:
            return {"interviewer_response": ""}
        response = interviewer.generate_response(
            candidate=state.candidate,
            conversation_history=state.history,
            latest_user_message=state.user_message,
            observer_analysis=state.observer_analysis,
        )
        logger.log_agent_response(
            response,
            state.internal_thoughts[-1] if state.internal_thoughts else "",
        )
        # Append this turn (assistant message that prompted user + user reply) so next invocation has full history
        new_history = state.history + [
            {"role": "assistant", "content": state.interviewer_response or ""},
            {"role": "user", "content": state.user_message},
        ]
        return {
            "interviewer_response": response,
            "history": new_history,
        }

    return interviewer_response_or_question


def _make_summarize_node(
    feedback_generator: FeedbackGenerator, logger: InterviewLogger
):
    def summarize(state: InterviewState) -> dict:
        # Use logger so transcript includes the final (stop) user message
        transcript = logger.get_turns_summary()
        thoughts_summary = "\n".join(state.internal_thoughts)
        feedback = feedback_generator.generate(
            candidate=state.candidate,
            conversation_transcript=transcript,
            internal_thoughts_summary=thoughts_summary,
        )
        feedback_text = feedback_generator.format_feedback_text(feedback)
        full_feedback = (
            feedback_text + "\n\n## ПОДРОБНЫЙ АНАЛИЗ\n\n" + feedback.raw_feedback
        )
        logger.set_final_feedback(full_feedback)
        return {"final_feedback": full_feedback}

    return summarize


# ----- Graph builder -----


def build_interview_graph(
    candidate: CandidateProfile,
    logger: InterviewLogger,
    *,
    llm: Optional[LLMClient] = None,
    observer: Optional[ObserverAgent] = None,
    interviewer: Optional[InterviewerAgent] = None,
    feedback_generator: Optional[FeedbackGenerator] = None,
):
    """
    Build and compile the interview LangGraph.

    Args:
        candidate: Candidate profile (stored in initial state).
        logger: Interview logger (used by nodes).
        llm: Optional LLM client; created if not provided.
        observer: Optional Observer agent; created if not provided.
        interviewer: Optional Interviewer agent; created if not provided.
        feedback_generator: Optional FeedbackGenerator; created if not provided.

    Returns:
        Compiled graph (invocable with initial state).
    """
    if llm is None:
        llm = LLMClient()
    if observer is None:
        observer = ObserverAgent(llm)
    if interviewer is None:
        interviewer = InterviewerAgent(llm)
    if feedback_generator is None:
        feedback_generator = FeedbackGenerator(llm)

    builder = StateGraph(InterviewState)

    # Nodes
    builder.add_node(
        "greet_and_get_candidates_description",
        _make_greet_node(interviewer, logger),
    )
    builder.add_node("route_user_message", _make_route_user_message_node(logger))
    builder.add_node(
        "send_answer_to_observer",
        _make_send_answer_to_observer_node(observer),
    )
    builder.add_node(
        "send_hints_to_interviewer",
        _make_send_hints_to_interviewer_node(),
    )
    builder.add_node(
        "interviewer_response_or_question",
        _make_interviewer_response_node(interviewer, logger),
    )
    builder.add_node(
        "summarize",
        _make_summarize_node(feedback_generator, logger),
    )

    # Entry: conditional routing
    builder.add_conditional_edges(START, _route_after_start)

    # Linear edges
    builder.add_edge("greet_and_get_candidates_description", END)
    builder.add_conditional_edges("route_user_message", _route_after_user_message)
    builder.add_edge("send_answer_to_observer", "send_hints_to_interviewer")
    builder.add_edge("send_hints_to_interviewer", "interviewer_response_or_question")
    builder.add_edge("interviewer_response_or_question", END)
    builder.add_edge("summarize", END)

    return builder.compile()
