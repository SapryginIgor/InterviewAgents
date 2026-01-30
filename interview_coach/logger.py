"""Interview session logger - saves to JSON format as per specification."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


class Turn(BaseModel):
    """Single turn in the interview dialogue."""
    turn_id: int
    agent_visible_message: str
    user_message: str
    internal_thoughts: str


class InterviewLog(BaseModel):
    """Complete interview session log."""
    participant_name: str
    position: str
    target_grade: str
    experience: str
    timestamp: str
    turns: list[Turn]
    final_feedback: Optional[str] = None


class InterviewLogger:
    """Handles logging of interview sessions to JSON."""
    
    def __init__(self, participant_name: str, position: str, grade: str, experience: str):
        self.log = InterviewLog(
            participant_name=participant_name,
            position=position,
            target_grade=grade,
            experience=experience,
            timestamp=datetime.now().isoformat(),
            turns=[]
        )
        self._current_turn_id = 0
        self._pending_agent_message: Optional[str] = None
        self._pending_thoughts: Optional[str] = None
    
    def log_agent_response(self, visible_message: str, internal_thoughts: str):
        """Log agent's response and internal thoughts (before user responds)."""
        self._pending_agent_message = visible_message
        self._pending_thoughts = internal_thoughts
    
    def log_user_message(self, user_message: str):
        """Log user message and complete the turn."""
        self._current_turn_id += 1
        turn = Turn(
            turn_id=self._current_turn_id,
            agent_visible_message=self._pending_agent_message or "",
            user_message=user_message,
            internal_thoughts=self._pending_thoughts or ""
        )
        self.log.turns.append(turn)
        self._pending_agent_message = None
        self._pending_thoughts = None
    
    def set_final_feedback(self, feedback: str):
        """Set the final feedback summary."""
        self.log.final_feedback = feedback
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save the log to a JSON file with multi-line text as line arrays for readability."""
        if filepath is None:
            safe_name = self.log.participant_name.replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/interview_log_{safe_name}_{timestamp}.json"
        
        data = self.log.model_dump()
        # Format long text fields as arrays of lines so the file is readable (not one-liners)
        for turn in data.get("turns", []):
            for key in ("agent_visible_message", "user_message", "internal_thoughts"):
                if key in turn and isinstance(turn[key], str):
                    turn[key] = (turn[key].split("\n") if turn[key] else [])
        if data.get("final_feedback") and isinstance(data["final_feedback"], str):
            data["final_feedback"] = data["final_feedback"].split("\n")

        path = Path(filepath)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return str(path)
    
    def get_conversation_history(self) -> list[dict]:
        """Get conversation history for context awareness."""
        history = []
        for turn in self.log.turns:
            history.append({"role": "assistant", "content": turn.agent_visible_message})
            history.append({"role": "user", "content": turn.user_message})
        return history
    
    def get_turns_summary(self) -> str:
        """Get a text summary of all turns for feedback generation."""
        summary = []
        for turn in self.log.turns:
            summary.append(f"Interviewer: {turn.agent_visible_message}")
            summary.append(f"Candidate: {turn.user_message}")
        return "\n".join(summary)
