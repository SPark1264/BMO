from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Literal

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

#Environment Config
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TRANSCRIPTION_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "5"))


#Audio Record
def record_audio_wav(output_path: Path, duration_seconds: int) -> None:
    """Record a mono 16kHz wav file using arecord (Linux / Raspberry Pi)."""
    if shutil.which("arecord") is None:
        raise RuntimeError(
            "`arecord` not found. Install with: sudo apt install alsa-utils"
        )

    subprocess.run(
        [
            "arecord",
            "-d", str(duration_seconds),
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            str(output_path),
        ],
        check=True,
    )


#assistant
class BMOAssistant:
    def __init__(self, client: OpenAI) -> None:
        self.client = client

    def ask_chatbot(self, prompt: str) -> str:
        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": (
                "You are BMO, a friendly Raspberry Pi assistant. "
                "Keep answers concise and practical."
            ),
        }

        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": prompt,
        }

        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[system_message, user_message],
        )

        return (response.choices[0].message.content or "").strip()

    def transcribe_microphone(self, duration_seconds: int = RECORD_SECONDS) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = Path(temp_wav.name)

        try:
            record_audio_wav(temp_path, duration_seconds)

            with temp_path.open("rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=TRANSCRIPTION_MODEL,
                    file=audio_file,
                )

            return transcript.text.strip()

        finally:
            temp_path.unlink(missing_ok=True)


#OPENAI check
def require_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)
    raise RuntimeError("Set OPENAI_API_KEY first.")


#cli
def run_cli(assistant: BMOAssistant) -> None:
    print("=== BMO OpenAI Assistant (CLI) ===")
    print("1) Text chatbot")
    print("2) Speech-to-text + chatbot")

    choice = input("Select mode (1/2): ").strip()

    if choice == "1":
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                return
            if not user_input:
                continue
            print(f"BMO: {assistant.ask_chatbot(user_input)}\n")

    elif choice == "2":
        print(f"Recording for {RECORD_SECONDS} seconds...")
        spoken_text = assistant.transcribe_microphone()
        print(f"You (speech): {spoken_text}")
        print(f"BMO: {assistant.ask_chatbot(spoken_text)}")

    else:
        print("Invalid choice.")


#gui
def run_gui(assistant: BMOAssistant) -> None:
    import tkinter as tk
    from tkinter import scrolledtext
    from typing import Callable

    root = tk.Tk()
    root.title("BMO Assistant")
    root.geometry("900x520")

    output = scrolledtext.ScrolledText(root, wrap="word", font=("Arial", 14))
    output.pack(fill="both", expand=True, padx=10, pady=10)

    input_frame = tk.Frame(root)
    input_frame.pack(fill="x", padx=10, pady=(0, 10))

    user_entry = tk.Entry(input_frame, font=("Arial", 14))
    user_entry.pack(side="left", fill="x", expand=True)

    def ui(fn: Callable[[], None]) -> None:
        # Keeps tkinter updates on the main thread and avoids PyCharm's after() typing noise
        root.after(0, lambda: fn())

    def append_line(text: str) -> None:
        output.insert("end", text + "\n")
        output.see("end")

    def set_enabled(enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        user_entry.configure(state=state)
        send_button.configure(state=state)
        speak_button.configure(state=state)

    def send_text() -> None:
        text = user_entry.get().strip()
        if not text:
            return

        append_line(f"You: {text}")
        user_entry.delete(0, "end")
        set_enabled(False)

        def worker() -> None:
            try:
                reply = assistant.ask_chatbot(text)
                ui(lambda: append_line(f"BMO: {reply}"))
            except Exception as exc:
                ui(lambda: append_line(f"Error: {exc}"))
            finally:
                ui(lambda: set_enabled(True))

        threading.Thread(target=worker, daemon=True).start()

    def speak() -> None:
        set_enabled(False)
        append_line("BMO: Recording...")

        def worker() -> None:
            try:
                spoken = assistant.transcribe_microphone()
                ui(lambda: append_line(f"You (speech): {spoken}"))
                reply = assistant.ask_chatbot(spoken)
                ui(lambda: append_line(f"BMO: {reply}"))
            except Exception as exc:
                ui(lambda: append_line(f"Error: {exc}"))
            finally:
                ui(lambda: set_enabled(True))

        threading.Thread(target=worker, daemon=True).start()

    send_button = tk.Button(input_frame, text="Send", command=send_text)
    send_button.pack(side="left", padx=(8, 0))

    speak_button = tk.Button(input_frame, text="Speak", command=speak)
    speak_button.pack(side="left", padx=(8, 0))

    user_entry.bind("<Return>", lambda _event: send_text())
    append_line("BMO: Ready")
    root.mainloop()
#main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cli", "gui"], default="cli")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assistant = BMOAssistant(require_openai_client())

    if args.mode == "gui":
        run_gui(assistant)
    else:
        run_cli(assistant)


if __name__ == "__main__":
    main()
