import fire
import pyaudiowpatch as pyaudio
from loguru import logger
from pydantic import BaseModel


class AudioDevice(BaseModel):
    name: str
    index: int
    sample_rate: int
    channels: int


def get_default_loopback_device() -> AudioDevice:
    p = pyaudio.PyAudio()
    device = p.get_default_wasapi_loopback()
    return AudioDevice(
        name=device["name"],
        index=device["index"],
        sample_rate=int(device["defaultSampleRate"]),
        channels=device["maxInputChannels"],
    )


default_device = get_default_loopback_device()


def loopback_stream(*, device: AudioDevice, chunk_seconds: float = 1.0):
    """
    Create and return an audio stream for loopback recording.

    Args:
        device (AudioDevice): The audio device to use for recording.
        chunk_seconds (float, optional): The duration of each audio chunk in seconds. Defaults to 1.0.

    Returns:
        PyAudio.Stream: The audio stream object for recording.
    """
    # Initialize the PyAudio object
    audio = pyaudio.PyAudio()

    chunk = int(chunk_seconds * device.sample_rate)

    return audio.open(
        format=pyaudio.paFloat32,
        input_device_index=device.index,
        channels=device.channels,
        rate=device.sample_rate,
        input=True,
        frames_per_buffer=chunk,
    )


if __name__ == "__main__":
    fire.Fire(get_default_loopback_device)
