#!/usr/bin/env python3
"""
Multi-Agent Interview Coach - Main Entry Point

Usage:
    python main.py
    You will be asked for your name, position, grade, and experience at the start.
"""

import argparse
import os
import sys
from interview_coach import InterviewManager


def get_input_with_default(prompt: str, default: str = "") -> str:
    """Get input from user with optional default value."""
    if default:
        result = input(f"{prompt} [{default}]: ").strip()
        return result if result else default
    return input(f"{prompt}: ").strip()


def main():
    parser = argparse.ArgumentParser(
        description="Тренер по техническим интервью — практика с AI-агентами"
    )
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Скрыть внутренние заметки агентов")
    parser.add_argument("--output", "-o", help="Путь к файлу лога")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🎯 ТРЕНЕР ПО ТЕХНИЧЕСКИМ ИНТЕРВЬЮ")
    print("    Система практики с AI-агентами")
    print("=" * 60 + "\n")
    
    # Ask user for candidate info at the start
    name = get_input_with_default("Введите ваше имя", "Кандидат")
    position = get_input_with_default("Введите целевую позицию", "Backend-разработчик")
    grade = get_input_with_default("Введите целевой грейд (Junior/Middle/Senior)", "Junior")
    experience = get_input_with_default(
        "Кратко опишите ваш опыт",
        "Изучаю программирование, пет-проекты"
    )
    
    # Name in log file (e.g. your real name for competition); if unset, same as name above
    log_name = os.environ.get("LOG_PARTICIPANT_NAME", "").strip() or name

    # Initialize interview manager
    try:
        manager = InterviewManager(
            candidate_name=name,
            position=position,
            grade=grade,
            experience=experience,
            verbose=not args.quiet,
            log_participant_name=log_name if log_name != name else None,
        )
    except Exception as e:
        print(f"\n❌ Ошибка при запуске интервью: {e}")
        print("\nУбедитесь, что:")
        print("1. Создан файл .env с API-ключом (см. .env.example)")
        print("2. Установлены зависимости: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run interview
    try:
        log_path = manager.run_interactive()
        
        # Override output path if specified
        if args.output:
            import shutil
            shutil.move(log_path, args.output)
            log_path = args.output
            print(f"📁 Лог сохранён в: {log_path}")
            
    except Exception as e:
        print(f"\n❌ Ошибка во время интервью: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ Интервью завершено! Обратная связь выше.")
    print(f"📁 Полный лог сессии сохранён в: {log_path}")
    print("\nУдачи на реальных интервью! 🚀\n")


if __name__ == "__main__":
    main()
