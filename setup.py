import os

from setuptools import find_packages, setup


def get_requirements(path):
    with open(path, encoding="utf-8") as requirements:
        return [requirement.strip() for requirement in requirements]


base_dir = os.path.dirname(os.path.abspath(__file__))
install_requires = get_requirements(os.path.join(base_dir, "requirements.txt"))
conversion_requires = get_requirements(
    os.path.join(base_dir, "requirements.conversion.txt")
)

setup(
    name="faster-whisper",
    version="0.1.0",
    description="Faster Whisper transcription with CTranslate2",
    author="Guillaume Klein",
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require={
        "conversion": conversion_requires,
    },
    packages=find_packages(),
)
