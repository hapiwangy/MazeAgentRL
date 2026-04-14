# Prompt Storage

This directory is reserved for storing prompt versions used by the project.

## Structure

- `reward_shaping/`: prompts related to LLM-based reward generation.

## Versioning Convention

- Create one file per prompt version.
- Prefer descriptive names such as `v1_openailm_reward_range.md`.
- Keep the prompt text, variable placeholders, and notes about where the version came from.

## Current Status

- The runtime code still builds prompts directly inside `OpenAILLM.py`.
- This folder is currently a prompt archive and versioning location.
- Future work can switch runtime loading from inline strings to files in this directory.
