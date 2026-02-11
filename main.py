"""BMO Raspberry Pi assistant with OpenAI chat + speech-to-text."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TRANSCRIPTION_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "5"))


def record_audio_wav(output_path: Path, duration_seconds: int) -> None:
    """Record a mono 16kHz wav file from the default ALSA input device."""
    if shutil.which("arecord") is None:
        raise RuntimeError(
            "`arecord` not found. Install with: sudo apt install alsa-utils"
        )

    subprocess.run(
        [
            "arecord",
            "-d",
            str(duration_seconds),
            "-f",
            "S16_LE",
            "-r",
            "16000",
            "-c",
            "1",
            str(output_path),
        ],
        check=True,
    )


class BMOAssistant:
    def __init__(self, client: OpenAI):
        self.client = client

    def ask_chatbot(self, prompt: str) -> str:
        """Type-safe OpenAI chat call (fixes PyCharm type warning)."""
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

        content = response.choices[0].message.content or ""
        return content.strip()

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


def require_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY first.")
    return OpenAI(api_key=api_key)


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
        print("Invalid choice. Please run again and pick 1 or 2.")


def run_gui(assistant: BMOAssistant) -> None:
    import tkinter as tk
    from tkinter import scrolledtext

    root = tk.Tk()
    root.title("BMO Assistant")
    root.geometry("900x520")

    output = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 14))
    output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    input_frame = tk.Frame(root)
    input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

    user_entry = tk.Entry(input_frame, font=("Arial", 14))
    user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def append_line(text: str) -> None:
        output.insert(tk.END, text + "\n")
        output.see(tk.END)

    def set_enabled(enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        user_entry.configure(state=state)
        send_button.configure(state=state)
        speak_button.configure(state=state)

    def send_text() -> None:
        text = user_entry.get().strip()
        if not text:
            return
        append_line(f"You: {text}")
        user_entry.delete(0, tk.END)
        set_enabled(False)

        def worker() -> None:
            try:
                reply = assistant.ask_chatbot(text)
                root.after(0, lambda: append_line(f"BMO: {reply}\n"))
            except Exception as exc:
                root.after(0, lambda: append_line(f"Error: {exc}\n"))
            finally:
                root.after(0, lambda: set_enabled(True))

        threading.Thread(target=worker, daemon=True).start()

    def speak() -> None:
        set_enabled(False)
        append_line("BMO: Recording...")

        def worker() -> None:
            try:
                spoken = assistant.transcribe_microphone()
                root.after(0, lambda: append_line(f"You (speech): {spoken}"))
                reply = assistant.ask_chatbot(spoken)
                root.after(0, lambda: append_line(f"BMO: {reply}\n"))
            except Exception as exc:
                root.after(0, lambda: append_line(f"Error: {exc}\n"))
            finally:
                root.after(0, lambda: set_enabled(True))

        threading.Thread(target=worker, daemon=True).start()

    send_button = tk.Button(input_frame, text="Send", command=send_text)
    send_button.pack(side=tk.LEFT, padx=(8, 0))

    speak_button = tk.Button(input_frame, text="Speak", command=speak)
    speak_button.pack(side=tk.LEFT, padx=(8, 0))

    user_entry.bind("<Return>", lambda _event: send_text())
    append_line("BMO: Ready")
    root.mainloop()


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
