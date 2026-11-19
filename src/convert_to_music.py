import os
import numpy as np
from scipy.io import wavfile
import sounddevice as sd

NOTE_OFFSETS = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

DURATION_BEATS = {
    'double_whole': 8.0, 'whole': 4.0, 'half': 2.0, 'quarter': 1.0,
    'eighth': 0.5, 'sixteenth': 0.25, 'thirty_second': 0.125,
    'sixty_fourth': 0.0625
}

# example PrIMuS encoding for testing
SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sheet music", "primus", "package_aa", "110002955-1_12_1", "110002955-1_12_1.semantic",
)


def pitch_to_freq(pitch_str):
    """Converts 'C4' to frequency in Hz (A4 = 440Hz)."""
    note_name = pitch_str[:-1]
    octave = int(pitch_str[-1])
    midi_num = 12 * (octave + 1) + NOTE_OFFSETS[note_name]
    return 440.0 * (2.0 ** ((midi_num - 69) / 12.0))


def synthesize_note(freq, duration_sec, sample_rate=44100):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    if freq == 0:  # Rest
        return np.zeros_like(t)

    # combine fundamental frequency + soft harmonics for a warmer tone
    wave = 0.6 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)

    # apply short fade-in and fade-out envelope to prevent audio clicks
    fade = min(int(0.015 * sample_rate), len(t) // 4)
    envelope = np.ones_like(t)
    if fade > 0:
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)

    return wave * envelope


def parse_duration(dur_str):
    """'eighth.' -> 0.75 beats (in quarter notes). Supports multiple dots."""
    dots = len(dur_str) - len(dur_str.rstrip('.')) # for dotted notes (each dot adds half of the duration of the original note)
    beats = DURATION_BEATS[dur_str.rstrip('.')]
    return beats * (2 - 0.5 ** dots)


def semantic_to_audio(tokens, bpm=120, sample_rate=44100):
    """
    Converts a list of semantic tokens (clef, key/time signature, notes, rests,
    multirests, barlines) into audio. bpm is quarter notes per minute.

    Pitches in the semantic encoding are already absolute names (e.g. Bb4), so
    the clef and key signature are parsed but don't change pitch. They are left in case they 
    could be used for some other extension (maybe recreating the sheet music?). The time
    signature sets the length of a measure because multirest-N (N measures of silence) needs 
    this to calculate how long to rest.
    """
    sec_per_beat = 60.0 / bpm
    clef = None
    key_signature = None
    beats_per_measure = 4.0  # in quarter-note beats; default 4/4
    audio_segments = []

    for token in tokens: # parse each token in the semantic encoding (clef vs time signature vs rest vs note etc)
        if token.startswith("clef-"):
            clef = token.split("-", 1)[1] # not strictly needed for audio synthesis since pitch is already handled in the note encodings
        elif token.startswith("keySignature-"):
            key_signature = token.split("-", 1)[1] # also not strictly needed for audio synthesis since pitch is already handled in the note encodings
        elif token.startswith("timeSignature-"): # set the length of a measure
            ts = token.split("-", 1)[1]
            if ts == "C": # common time signature, equivalent to 4/4
                num, den = 4, 4
            elif ts == "C/": # cut time signature, equivalent to 2/2
                num, den = 2, 2
            else: # otherwise a numerical split time signature
                num, den = (int(x) for x in ts.split("/"))
            beats_per_measure = num * 4.0 / den # in terms of quarter notes
        elif token.startswith("multirest-"): # multirest-N means N measures of silence, so we need to convert that to beats
            measures = int(token.split("-", 1)[1])
            beats = measures * beats_per_measure
            audio_segments.append(synthesize_note(0.0, beats * sec_per_beat, sample_rate))
        elif token.startswith("rest-"): # no pitch needed, just duration
            beats = parse_duration(token.split("-", 1)[1])
            audio_segments.append(synthesize_note(0.0, beats * sec_per_beat, sample_rate))
        elif token.startswith("note-"): # pitch and duration are both needed to synthesize the note
            pitch_str, dur_str = token.split("-", 1)[1].split("_", 1)
            beats = parse_duration(dur_str)
            freq = pitch_to_freq(pitch_str)
            audio_segments.append(synthesize_note(freq, beats * sec_per_beat, sample_rate))
        # barlines and any other tokens are ignored

    if not audio_segments:
        return np.zeros(0)
    return np.concatenate(audio_segments)


def load_semantic(path):
    """Reads a .semantic file (tab-separated tokens) into a list of tokens."""
    with open(path) as f:
        return f.read().split()


def sample_to_audio(sample=None, bpm=120, sample_rate=44100, play=True):
    """
    sample: a list of semantic tokens, a path to a .semantic file, or None to
    use the example PrIMuS encoding at SAMPLE_PATH.
    """
    if sample is None:
        sample = SAMPLE_PATH
    if isinstance(sample, str):
        sample = load_semantic(sample)

    if not isinstance(sample, list):
        raise ValueError("sample must be a list of tokens or a path to a .semantic file")

    audio = semantic_to_audio(sample, bpm=bpm, sample_rate=sample_rate)

    if play: # play it outloud!!
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
    return audio


if __name__ == "__main__":
    import sys

    # Usage: python src/convert_to_music.py [path/to/file.semantic]
    path = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else None
    audio = sample_to_audio(path)
