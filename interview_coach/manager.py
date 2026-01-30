"""Interview Manager - orchestrates the multi-agent interview."""

from typing import Any, Optional, Callable

from .llm import LLMClient
from .agents import CandidateProfile, InterviewerAgent, ObserverAgent
from .logger import InterviewLogger
from .feedback import FeedbackGenerator
from .graph import build_interview_graph, InterviewState


def _state_dict(result: Any) -> dict[str, Any]:
    """Normalize graph result to a state dict (LangGraph may return model or dict)."""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return dict(result) if result else {}


def _graph_state_copy(state: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Return a copy of graph state for next invocation (avoid mutating stored state)."""
    if state is None:
        return {}
    return {k: v for k, v in state.items()}


class InterviewManager:
    """
    Orchestrates the multi-agent interview process via LangGraph.
    Coordinates between Interviewer and Observer agents.
    """

    STOP_COMMANDS = [
        "стоп",
        "stop",
        "стоп интервью",
        "stop interview",
        "стоп игра",
        "давай фидбэк",
    ]

    def __init__(
        self,
        candidate_name: str,
        position: str,
        grade: str,
        experience: str,
        verbose: bool = True,
        log_participant_name: Optional[str] = None,
    ):
        """Initialize the interview manager.
        candidate_name: used in the interview (interviewer addresses the candidate by this).
        log_participant_name: name saved in the log file (e.g. your real name for competition); if None, uses candidate_name.
        """
        self.llm = LLMClient()
        self.candidate = CandidateProfile(
            name=candidate_name,
            position=position,
            grade=grade,
            experience=experience,
        )
        self.verbose = verbose
        participant_for_log = log_participant_name if log_participant_name else candidate_name

        self.interviewer = InterviewerAgent(self.llm)
        self.observer = ObserverAgent(self.llm)
        self.feedback_generator = FeedbackGenerator(self.llm)

        self.logger = InterviewLogger(
            participant_name=participant_for_log,
            position=position,
            grade=grade,
            experience=experience,
        )

        self._graph = build_interview_graph(
            self.candidate,
            self.logger,
            llm=self.llm,
            observer=self.observer,
            interviewer=self.interviewer,
            feedback_generator=self.feedback_generator,
        )
        self._graph_state: Optional[dict[str, Any]] = None
        self.is_active = False

    def start(self) -> str:
        """Start the interview and return the initial greeting."""
        self.is_active = True
        initial = InterviewState(
            candidate=self.candidate,
            first_turn=True,
        )
        result = self._graph.invoke(initial.model_dump())
        self._graph_state = _state_dict(result)
        greeting = result.get("interviewer_response", "")
        if self.verbose and result.get("internal_thoughts"):
            self._print_internal_thoughts(result["internal_thoughts"][-1])
        return greeting

    def process_response(self, user_message: str) -> tuple[str, bool]:
        """
        Process user response and generate next interviewer message.
        Returns (response, is_finished).
        """
        if self._graph_state is None:
            self._graph_state = InterviewState(
                candidate=self.candidate,
                first_turn=False,
            ).model_dump()
        next_state = {
            **_graph_state_copy(self._graph_state),
            "user_message": user_message,
            "first_turn": False,
        }
        result = self._graph.invoke(next_state)
        self._graph_state = _state_dict(result)

        if result.get("final_feedback"):
            self.is_active = False
            return result["final_feedback"], True

        if self.verbose and result.get("internal_thoughts"):
            self._print_internal_thoughts(result["internal_thoughts"][-1])
        return result.get("interviewer_response", ""), False

    def _print_internal_thoughts(self, thoughts: str):
        """Print internal thoughts to console (for debugging/demo)."""
        print("\n" + "-" * 50)
        print("🧠 ВНУТРЕННИЕ ЗАМЕТКИ (скрыты от кандидата):")
        print(thoughts)
        print("-" * 50 + "\n")

    def _read_multiline_input(self) -> str:
        """Read lines until EOF (Ctrl+D). User types as much as they want, then Ctrl+D to send."""
        lines = []
        try:
            while True:
                prompt = "👤 Вы: " if not lines else "    "
                line = input(prompt)
                lines.append(line)
        except EOFError:
            pass
        return "\n".join(lines).strip()

    def _generate_final_feedback(self) -> str:
        """Generate and return final feedback (used on KeyboardInterrupt)."""
        self.is_active = False
        transcript = self.logger.get_turns_summary()
        thoughts_summary = "\n".join(
            self._graph_state.get("internal_thoughts", [])
            if self._graph_state
            else []
        )
        feedback = self.feedback_generator.generate(
            candidate=self.candidate,
            conversation_transcript=transcript,
            internal_thoughts_summary=thoughts_summary,
        )
        feedback_text = self.feedback_generator.format_feedback_text(feedback)
        full_feedback = (
            feedback_text + "\n\n## ПОДРОБНЫЙ АНАЛИЗ\n\n" + feedback.raw_feedback
        )
        self.logger.set_final_feedback(full_feedback)
        return full_feedback
    
    def save_log(self, filepath: Optional[str] = None) -> str:
        """Save the interview log to JSON file."""
        return self.logger.save(filepath)
    
    def run_interactive(self, output_callback: Optional[Callable[[str], None]] = None):
        """
        Run an interactive interview session in the terminal.
        """
        print("\n" + "=" * 60)
        print("🎯 ТРЕНЕР ПО ТЕХНИЧЕСКИМ ИНТЕРВЬЮ")
        print("=" * 60)
        print(f"Кандидат: {self.candidate.name}")
        print(f"Позиция: {self.candidate.position}")
        print(f"Целевой грейд: {self.candidate.grade}")
        print(f"Опыт: {self.candidate.experience}")
        print("=" * 60)
        print("Введите «стоп» или «stop», чтобы завершить интервью и получить обратную связь.")
        print("Для ответа: вводите текст (Enter — новая строка), затем Ctrl+D — отправить ответ.")
        print("=" * 60 + "\n")
        
        # Start interview
        greeting = self.start()
        print(f"🎤 Интервьюер: {greeting}\n")
        
        # Interview loop
        while self.is_active:
            try:
                user_input = self._read_multiline_input()
                if not user_input:
                    continue
                print("-" * 50)
                
                response, is_finished = self.process_response(user_input)
                
                if is_finished:
                    print("\n" + response)
                    break
                else:
                    print(f"\n🎤 Интервьюер: {response}\n")
                    
            except KeyboardInterrupt:
                print("\n\nИнтервью прервано. Формирую обратную связь...")
                response = self._generate_final_feedback()
                print("\n" + response)
                break
        
        # Save log
        log_path = self.save_log()
        print(f"\n📁 Лог интервью сохранён в: {log_path}")
        
        return log_path
