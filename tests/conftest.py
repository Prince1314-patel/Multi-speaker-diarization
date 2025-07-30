import pytest
from unittest import mock

@pytest.fixture
def sample_audio_path(tmp_path):
    """Fixture that provides a path to a sample audio file for testing.

    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest.

    Returns:
        str: Path to the sample audio file.
    """
    # This is a placeholder; replace with actual test audio setup if needed
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"fake audio data")
    return str(audio_file)

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to mock environment variables for tests.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test_token")
    monkeypatch.setenv("SOME_OTHER_ENV", "value")

@pytest.mark.parametrize("input_value,expected", [
    (1, 2),
    (2, 3),
])
def test_param_example(input_value, expected):
    """Example parameterized test to demonstrate usage in conftest.py.

    Args:
        input_value (int): Input value for the test.
        expected (int): Expected result.
    """
    assert input_value + 1 == expected 