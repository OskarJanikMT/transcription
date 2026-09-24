# Napisy do Premiere — Whisper → SRT

Prosta aplikacja Windows do lokalnej transkrypcji nagrań audio i wideo. Tworzy plik `.srt` z czasami, który można przeciągnąć do sekwencji w Adobe Premiere Pro.

## Uruchomienie

1. Uruchom dwukrotnie `uruchom.bat`.
2. Przy pierwszym uruchomieniu skrypt automatycznie zainstaluje Python 3.13 (przez Windows App Installer), a następnie wymagania.
3. Przy pierwszej transkrypcji pobierze wybrany model Whisper.

Komputer musi mieć połączenie z internetem przy pierwszym uruchomieniu. Jeżeli Windows nie ma programu `winget` / App Installer, skrypt poda link do ręcznej instalacji Pythona.

## Użycie

1. Kliknij **Wybierz plik** i wskaż nagranie.
2. Wybierz miejsce zapisu napisu `.srt` (domyślnie obok nagrania).
3. Ustaw `pl` dla polskich nagrań oraz model `small` jako bezpieczny kompromis jakości i szybkości.
4. Kliknij **Generuj napisy SRT**.
5. W Premiere zaimportuj plik SRT jak media, a potem przeciągnij go na sekwencję.

Program działa lokalnie. `faster-whisper` korzysta z FFmpeg dostarczanego przez pakiet PyAV; w razie nietypowego formatu pliku może być potrzebny systemowy FFmpeg.

## Modele

- `tiny` / `base` — najszybsze, lecz mniej dokładne;
- `small` — zalecany dla większości nagrań;
- `medium` / `large-v3` — wyższa jakość, ale wyraźnie większe wymagania.

`auto` przy sprzęcie spróbuje użyć karty NVIDIA, a w razie problemu przełączy się na procesor.
