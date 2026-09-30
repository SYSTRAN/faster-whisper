# Contributing to faster-whisper

Contributions are welcome! Here are some pointers to help you install the library for development and validate your changes before submitting a pull request.

## Install the library for development

We recommend installing the module in editable mode with the `dev` extra requirements:

```bash
git clone https://github.com/SYSTRAN/faster-whisper.git
cd faster-whisper/
pip install -e .[dev]
```

## Validate the changes before creating a pull request

1. Make sure the existing tests are still passing (and consider adding new tests as well!):

```bash
pytest tests/
```

2. Reformat and validate the code with the following tools:

```bash
black .
isort .
flake8 .
```

These steps are also run automatically in the CI when you open the pull request.

## Change control and validation

For behavior changes, describe the requirement and observable acceptance
criteria in the pull request. Add unit tests for boundary cases and an
integration test when the change crosses an audio or transcription boundary.
Record the exact test commands and preserve validation artifacts without
including private recordings, credentials, or unnecessary personal data.

Please merge the pull request when you feel comfortable that the review,
checks, and required provenance are complete.
