import json

from clearscribe.formats import _timestamp, to_json, to_srt, to_txt, to_vtt
from clearscribe.transcribe import Segment, Transcript


def sample_transcript():
    return Transcript(
        segments=[
            Segment(0.0, 2.5, "Hello and welcome."),
            Segment(2.5, 5.043, "This is ClearScribe."),
            Segment(3661.2, 3663.999, "One hour in."),
        ],
        language="en",
        duration=3664.0,
    )


def test_timestamp_srt_and_vtt():
    assert _timestamp(0.0, ",") == "00:00:00,000"
    assert _timestamp(5.043, ",") == "00:00:05,043"
    assert _timestamp(3661.2, ".") == "01:01:01.200"
    assert _timestamp(-1.0, ",") == "00:00:00,000"  # clamped


def test_to_txt():
    txt = to_txt(sample_transcript())
    assert "Hello and welcome. This is ClearScribe." in txt
    assert txt.endswith("\n")


def test_to_srt_structure():
    srt = to_srt(sample_transcript())
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nHello and welcome.")
    assert "\n2\n" in srt
    assert "01:01:01,200 --> 01:01:03,999" in srt


def test_to_vtt_structure():
    vtt = to_vtt(sample_transcript())
    assert vtt.startswith("WEBVTT")
    assert "00:00:02.500 --> 00:00:05.043" in vtt


def test_to_json_roundtrip():
    data = json.loads(to_json(sample_transcript()))
    assert data["language"] == "en"
    assert len(data["segments"]) == 3
    assert data["segments"][1]["text"] == "This is ClearScribe."
