import os

from setuptools import find_packages, setup

base_dir = os.path.dirname(os.path.abspath(__file__))


def get_long_description():
    readme_path = os.path.join(base_dir, "README.md")
    with open(readme_path, encoding="utf-8") as readme_file:
        return readme_file.read()


def get_requirements(path):
    with open(path, encoding="utf-8") as requirements:
        return [requirement.strip() for requirement in requirements]


install_requires = get_requirements(os.path.join(base_dir, "requirements.txt"))
conversion_requires = get_requirements(
    os.path.join(base_dir, "requirements.conversion.txt")
)

setup(
    name="faster-whisper",
    version="0.2.0",
    license="MIT",
    description="Faster Whisper transcription with CTranslate2",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Guillaume Klein",
    url="https://github.com/guillaumekln/faster-whisper",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="openai whisper speech ctranslate2 inference quantization transformer",
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "conversion": conversion_requires,
        "dev": conversion_requires
        + [
            "black==23.*",
            "flake8==6.*",
            "isort==5.*",
            "pytest==7.*",
        ],
    },
    packages=find_packages(),
)
