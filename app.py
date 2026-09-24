"""Whisper -> SRT: desktop transcription tool for Adobe Premiere Pro."""

from __future__ import annotations

import queue
import os
import re
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


MEDIA_TYPES = (
    "Nagrania audio i wideo (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.mp4 *.mov *.mkv *.avi *.webm);;"
    "Wszystkie pliki (*.*)"
)
MODELS = ("tiny", "base", "small", "medium", "large-v3")
CUDA_DLL_DIRECTORIES = []
LONG_WORD_PAUSE_SECONDS = 0.4
SHORT_TILE_MAX_DURATION_EXTENSION = 0.25
ALL_TILES_MAX_DURATION_EXTENSION = 0.5
NUMBER_PATTERN = re.compile(r"^[+-]?\d+(?:[.,]\d+)?(?:%|°)?[.)]?$")
UNIT_WORDS = frozenset(
    {
        "g", "kg", "mg", "t", "l", "ml", "dl", "m", "cm", "mm", "km",
        "m2", "m3", "w", "kw", "mw", "v", "a", "mah", "ah", "wh", "kwh",
        "hz", "mb", "gb", "tb", "ms", "s", "min", "h", "zł", "pln",
        "eur", "usd", "procent", "procentów", "stopni", "lat", "roku",
        "lata", "dni", "dzień", "godzin",
    }
)
CONNECTOR_WORDS = frozenset(
    {
        "a", "aby", "ale", "ani", "bez", "bo", "by", "być", "czy", "dla",
        "do", "gdy", "i", "jak", "jeśli", "lecz", "lub", "na", "nad", "nie",
        "o", "od", "oraz", "po", "pod", "ponad", "przez", "u", "w", "we",
        "więc", "z", "za", "ze", "że",
    }
)


def enable_cuda_libraries() -> None:
    """Make NVIDIA DLLs installed by pip discoverable on Windows."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    nvidia_directory = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_directory.is_dir():
        return

    for package_directory in nvidia_directory.iterdir():
        bin_directory = package_directory / "bin"
        if bin_directory.is_dir():
            CUDA_DLL_DIRECTORIES.append(os.add_dll_directory(str(bin_directory)))


def srt_time(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""
    total_ms = max(0, round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, milliseconds = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def word_ends_with_comma(word: str) -> bool:
    """Return whether a word finishes with a comma, before closing quotes."""
    return bool(re.search(r",(?=[\"'”’»)\]]*$)", word))


def remove_final_comma(word: str) -> str:
    """Hide the comma which has been used as a subtitle boundary."""
    return re.sub(r",(?=[\"'”’»)\]]*$)", "", word)


def is_connector(word: str) -> bool:
    """Whether a word should stay with the word which follows it."""
    return re.sub(r"[^a-ząćęłńóśźż]+$", "", word.lower()) in CONNECTOR_WORDS


def is_number(word: str) -> bool:
    """Recognize numbers emitted by the transcription."""
    return bool(NUMBER_PATTERN.fullmatch(word.strip()))


def is_unit(word: str) -> bool:
    """Recognize a common unit that must remain with its number."""
    normalized = re.sub(r"[^a-zA-Ząćęłńóśźż]+$", "", word.lower())
    return normalized in UNIT_WORDS


def group_character_count(words) -> int:
    return len(" ".join(clean_text(word.word) for word in words))


def may_exceed_limit_for_number_and_unit(words, start: int, end: int) -> bool:
    return end - start >= 2 and is_number(clean_text(words[end - 2].word)) and is_unit(clean_text(words[end - 1].word))


def split_balanced_group(words, target_words: int, max_words: int, max_chars: int) -> list[list]:
    """Balance a phrase around a soft target while respecting hard limits."""
    if len(words) == 1:
        return [words]

    minimum_group_count = (len(words) + max_words - 1) // max_words
    best_partition: tuple[float, list[list]] | None = None

    for group_count in range(minimum_group_count, len(words) + 1):
        # Prefer the fewest tiles possible.  Extra tiles are introduced only
        # when a hard word/character limit requires them; speech pauses have
        # already been handled before this balancing step.
        if best_partition is not None:
            break
        ideal_size = len(words) / group_count
        costs: dict[tuple[int, int], tuple[float, list[int]]] = {(0, 0): (0.0, [])}

        for group_index in range(group_count):
            for start in range(len(words)):
                state = costs.get((group_index, start))
                if state is None:
                    continue
                accumulated_cost, breaks = state
                remaining_groups = group_count - group_index - 1
                for end in range(start + 1, len(words) - remaining_groups + 1):
                    size = end - start
                    candidate_words = words[start:end]
                    number_and_unit = may_exceed_limit_for_number_and_unit(words, start, end)
                    if size > max_words and not number_and_unit:
                        continue
                    if group_character_count(candidate_words) > max_chars and size > 1 and not number_and_unit:
                        continue
                    cost = accumulated_cost + (size - ideal_size) ** 2 + 0.05 * (size - target_words) ** 2
                    if end < len(words) and is_connector(clean_text(words[end - 1].word)):
                        cost += 1000
                    if end < len(words) and is_number(clean_text(words[end - 1].word)) and is_unit(clean_text(words[end].word)):
                        cost += 1000
                    if size == 1 and len(words) > 1:
                        cost += 25
                    previous = costs.get((group_index + 1, end))
                    if previous is None or cost < previous[0]:
                        costs[(group_index + 1, end)] = (cost, breaks + [end])

        final_state = costs.get((group_count, len(words)))
        if final_state is None:
            continue
        break_positions = final_state[1][:-1]
        groups: list[list] = []
        start = 0
        for end in break_positions + [len(words)]:
            groups.append(words[start:end])
            start = end
        candidate = (final_state[0], groups)
        if best_partition is None or candidate[0] < best_partition[0]:
            best_partition = candidate

    return best_partition[1] if best_partition else [words]


def split_caption_word_groups(
    words, target_words: int, max_words: int, max_chars: int, pause_seconds: float
) -> list[list]:
    """Split on commas, numbers and long pauses, then balance phrases."""
    groups: list[list] = []
    phrase: list = []
    for index, word in enumerate(words):
        word_text = clean_text(getattr(word, "word", ""))
        if is_number(word_text) and phrase:
            groups.extend(split_balanced_group(phrase, target_words, max_words, max_chars))
            phrase = []
        phrase.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        pause_after_word = (
            next_word is not None
            and next_word.start - word.end >= pause_seconds
        )
        if word_ends_with_comma(word_text) or pause_after_word:
            groups.extend(split_balanced_group(phrase, target_words, max_words, max_chars))
            phrase = []
    if phrase:
        groups.extend(split_balanced_group(phrase, target_words, max_words, max_chars))
    return groups


def extend_short_tile_durations(entries, max_chars: int, pause_seconds: float) -> None:
    """Give short text a little more reading time when a real pause permits it."""
    short_tile_limit = max(4, max_chars // 3)
    for index, (start, end, text) in enumerate(entries):
        if len(text) > short_tile_limit:
            continue
        if index + 1 < len(entries) and entries[index + 1][0] - end >= pause_seconds:
            continue
        next_start = entries[index + 1][0] if index + 1 < len(entries) else end + SHORT_TILE_MAX_DURATION_EXTENSION
        available_end = next_start - 0.01
        extended_end = min(end + SHORT_TILE_MAX_DURATION_EXTENSION, available_end)
        if extended_end > end:
            entries[index] = (start, extended_end, text)


def extend_all_tile_durations(entries, extension_seconds: float, pause_seconds: float) -> None:
    """Extend every tile into the available gap without moving its start."""
    for index, (start, end, text) in enumerate(entries):
        if index + 1 < len(entries) and entries[index + 1][0] - end >= pause_seconds:
            continue
        next_start = entries[index + 1][0] if index + 1 < len(entries) else end + extension_seconds
        extended_end = min(end + extension_seconds, next_start - 0.01)
        if extended_end > end:
            entries[index] = (start, extended_end, text)


def write_srt(
    segments,
    destination: Path,
    target_words: int = 7,
    max_words: int = 10,
    max_chars: int = 42,
    pause_seconds: float = LONG_WORD_PAUSE_SECONDS,
    tile_extension_seconds: float = ALL_TILES_MAX_DURATION_EXTENSION,
) -> int:
    """Create a UTF-8 SRT file from faster-whisper segments."""
    entries: list[tuple[float, float, str]] = []
    for segment in segments:
        word_timestamps = list(getattr(segment, "words", None) or [])
        if not word_timestamps:
            continue
        for group in split_caption_word_groups(
            word_timestamps, target_words, max_words, max_chars, pause_seconds
        ):
            text_words = [clean_text(word.word) for word in group]
            text_words[-1] = remove_final_comma(text_words[-1])
            text = " ".join(word for word in text_words if word)
            if text:
                entries.append((group[0].start, group[-1].end, text))

    extend_short_tile_durations(entries, max_chars, pause_seconds)
    extend_all_tile_durations(entries, tile_extension_seconds, pause_seconds)
    with destination.open("w", encoding="utf-8-sig", newline="\n") as file:
        for index, (start, end, text) in enumerate(entries, start=1):
            file.write(f"{index}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n\n")
    return len(entries)


class WhisperSrtApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Napisy do Premiere — Whisper → SRT")
        self.geometry("720x680")
        self.minsize(640, 640)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_name = tk.StringVar(value="small")
        self.language = tk.StringVar(value="pl")
        self.target_words = tk.IntVar(value=7)
        self.max_words = tk.IntVar(value=10)
        self.max_chars = tk.IntVar(value=42)
        self.pause_seconds = tk.DoubleVar(value=LONG_WORD_PAUSE_SECONDS)
        self.tile_extension_seconds = tk.DoubleVar(value=ALL_TILES_MAX_DURATION_EXTENSION)
        # CPU works out of the box. CUDA requires matching NVIDIA CUDA libraries.
        self.device = tk.StringVar(value="cpu")
        self.status = tk.StringVar(value="Wybierz nagranie, aby rozpocząć.")
        self._build_ui()
        self.after(100, self._read_events)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self, padding=22)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Napisy do Premiere", font=("Segoe UI", 18, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(
            frame,
            text="Lokalna transkrypcja Whisper. Wynikiem jest plik SRT gotowy do zaimportowania do Premiere Pro.",
            wraplength=630,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 20))

        self._path_row(frame, 2, "Nagranie:", self.input_path, self.choose_input, "Wybierz plik")
        self._path_row(frame, 3, "Plik SRT:", self.output_path, self.choose_output, "Zapisz jako")

        ttk.Label(frame, text="Model:").grid(row=4, column=0, sticky="w", pady=(18, 4))
        ttk.Combobox(frame, textvariable=self.model_name, values=MODELS, state="readonly", width=16).grid(
            row=4, column=1, sticky="w", pady=(18, 4)
        )
        ttk.Label(frame, text="small: dobry balans szybkości i jakości").grid(
            row=4, column=2, sticky="w", padx=(12, 0), pady=(18, 4)
        )

        ttk.Label(frame, text="Język:").grid(row=5, column=0, sticky="w", pady=4)
        language_box = ttk.Combobox(frame, textvariable=self.language, state="readonly", width=16)
        language_box["values"] = ("pl", "auto", "en", "de", "es", "fr", "uk")
        language_box.grid(row=5, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="pl = polski, auto = automatyczne wykrywanie").grid(
            row=5, column=2, sticky="w", padx=(12, 0), pady=4
        )

        ttk.Label(frame, text="Sprzęt:").grid(row=6, column=0, sticky="w", pady=4)
        device_box = ttk.Combobox(frame, textvariable=self.device, state="readonly", width=16)
        device_box["values"] = ("auto", "cpu", "cuda")
        device_box.grid(row=6, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="CPU działa bez CUDA; CUDA wybierz tylko przy skonfigurowanej karcie NVIDIA").grid(
            row=6, column=2, sticky="w", padx=(12, 0), pady=4
        )

        ttk.Label(frame, text="Docelowo słów w kafelku:").grid(row=7, column=0, sticky="w", pady=4)
        ttk.Spinbox(frame, from_=1, to=30, textvariable=self.target_words, width=14).grid(
            row=7, column=1, sticky="w", pady=4
        )
        ttk.Label(frame, text="Wskazówka do równoważenia kafelków").grid(
            row=7, column=2, sticky="w", padx=(12, 0), pady=4
        )

        ttk.Label(frame, text="Maks. słów w kafelku:").grid(row=8, column=0, sticky="w", pady=4)
        ttk.Spinbox(frame, from_=1, to=30, textvariable=self.max_words, width=14).grid(
            row=8, column=1, sticky="w", pady=4
        )
        ttk.Label(frame, text="Twardy limit, z wyjątkiem liczby z jednostką").grid(
            row=8, column=2, sticky="w", padx=(12, 0), pady=4
        )

        ttk.Label(frame, text="Maks. znaków w kafelku:").grid(row=9, column=0, sticky="w", pady=4)
        ttk.Spinbox(frame, from_=8, to=160, textvariable=self.max_chars, width=14).grid(
            row=9, column=1, sticky="w", pady=4
        )
        ttk.Label(frame, text="Krótkie kafelki dostają więcej czasu w pauzie").grid(
            row=9, column=2, sticky="w", padx=(12, 0), pady=4
        )

        ttk.Label(frame, text="Pauza dzieląca kafelki (s):").grid(row=10, column=0, sticky="w", pady=4)
        ttk.Spinbox(
            frame, from_=0.01, to=5.0, increment=0.01, textvariable=self.pause_seconds, width=14
        ).grid(row=10, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="Domyślnie 0,40 s; liczona między słowami").grid(
            row=10, column=2, sticky="w", padx=(12, 0), pady=4
        )

        ttk.Label(frame, text="Wydłużenie kafelka (s):").grid(row=11, column=0, sticky="w", pady=4)
        ttk.Spinbox(
            frame, from_=0.0, to=5.0, increment=0.05, textvariable=self.tile_extension_seconds, width=14
        ).grid(row=11, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="Domyślnie 0,50 s; tylko do następnego kafelka").grid(
            row=11, column=2, sticky="w", padx=(12, 0), pady=4
        )

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(25, 8))
        ttk.Label(frame, textvariable=self.status, wraplength=650).grid(row=13, column=0, columnspan=3, sticky="w")
        self.start_button = ttk.Button(frame, text="Generuj napisy SRT", command=self.start)
        self.start_button.grid(row=14, column=0, columnspan=3, pady=(20, 0))

    @staticmethod
    def _path_row(parent, row, label, variable, command, button) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text=button, command=command).grid(row=row, column=2, padx=(10, 0), pady=4)

    def choose_input(self) -> None:
        path = filedialog.askopenfilename(title="Wybierz nagranie", filetypes=[("Nagrania", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.mp4 *.mov *.mkv *.avi *.webm"), ("Wszystkie pliki", "*.*")])
        if path:
            self.input_path.set(path)
            self.output_path.set(str(Path(path).with_suffix(".srt")))

    def choose_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Zapisz napisy", defaultextension=".srt", filetypes=[("Napisy SubRip", "*.srt")])
        if path:
            self.output_path.set(path)

    def start(self) -> None:
        source = Path(self.input_path.get())
        output = Path(self.output_path.get())
        if not source.is_file():
            messagebox.showerror("Brak nagrania", "Wybierz istniejący plik audio lub wideo.")
            return
        if output.suffix.lower() != ".srt":
            output = output.with_suffix(".srt")
            self.output_path.set(str(output))
        try:
            target_words = self.target_words.get()
            max_words = self.max_words.get()
            max_chars = self.max_chars.get()
            pause_seconds = self.pause_seconds.get()
            tile_extension_seconds = self.tile_extension_seconds.get()
        except tk.TclError:
            target_words = 0
            max_words = 0
            max_chars = 0
            pause_seconds = 0
            tile_extension_seconds = -1
        if (
            not 1 <= target_words <= max_words <= 30
            or not 8 <= max_chars <= 160
            or not 0.01 <= pause_seconds <= 5.0
            or not 0.0 <= tile_extension_seconds <= 5.0
        ):
            messagebox.showerror(
                "Nieprawidłowe ustawienia",
                "Docelowa liczba słów musi wynosić od 1 do maksimum (do 30), limit znaków od 8 do 160, pauza od 0,01 do 5 s, a wydłużenie od 0 do 5 s.",
            )
            return
        self.start_button.configure(state="disabled")
        self.progress.start(12)
        self.status.set("Uruchamianie Whispera — pierwsze użycie może pobrać model…")
        threading.Thread(
            target=self._transcribe,
            args=(
                source,
                output,
                target_words,
                max_words,
                max_chars,
                pause_seconds,
                tile_extension_seconds,
            ),
            daemon=True,
        ).start()

    def _transcribe(
        self,
        source: Path,
        output: Path,
        target_words: int,
        max_words: int,
        max_chars: int,
        pause_seconds: float,
        tile_extension_seconds: float,
    ) -> None:
        try:
            requested_device = self.device.get()
            if requested_device == "cuda":
                enable_cuda_libraries()
            from faster_whisper import WhisperModel

            # Model construction does not always reveal missing CUDA DLLs; therefore
            # automatic mode intentionally uses CPU for a dependable first run.
            device = "cuda" if requested_device == "cuda" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            model = WhisperModel(self.model_name.get(), device=device, compute_type=compute_type)
            language = None if self.language.get() == "auto" else self.language.get()
            self.events.put(("status", "Transkrypcja w toku…"))
            segments, info = model.transcribe(
                str(source),
                language=language,
                vad_filter=True,
                beam_size=5,
                word_timestamps=True,
            )
            count = write_srt(
                segments,
                output,
                target_words=target_words,
                max_words=max_words,
                max_chars=max_chars,
                pause_seconds=pause_seconds,
                tile_extension_seconds=tile_extension_seconds,
            )
            detected = getattr(info, "language", "nieznany")
            self.events.put(("done", (count, output, detected)))
        except ImportError:
            self.events.put(("error", "Brakuje biblioteki faster-whisper. Uruchom: pip install -r requirements.txt"))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def _read_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status.set(str(value))
                elif kind == "done":
                    count, output, language = value
                    self.progress.stop()
                    self.start_button.configure(state="normal")
                    self.status.set(f"Gotowe: {count} napisów, wykryty język: {language}.")
                    messagebox.showinfo("Gotowe", f"Zapisano {count} napisów:\n{output}\n\nZaimportuj ten plik SRT do Premiere Pro.")
                elif kind == "error":
                    self.progress.stop()
                    self.start_button.configure(state="normal")
                    self.status.set("Wystąpił błąd.")
                    messagebox.showerror("Błąd transkrypcji", str(value))
        except queue.Empty:
            pass
        self.after(100, self._read_events)


if __name__ == "__main__":
    WhisperSrtApp().mainloop()
