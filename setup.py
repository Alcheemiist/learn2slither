from setuptools import setup, find_packages

setup(
    name="snake_rl",
    version="0.1.0",
    description="Learn2Slither reinforcement-learning snake project",
    author="Alchemist",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pygame",
        "pydantic",
    ],
    extras_require={
        "dev": ["pytest", "flake8"],
    },
)