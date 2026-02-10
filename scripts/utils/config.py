"""
config.py - Configuration loading and merging

Handles the layered config system: code defaults → preset file → CLI overrides.
Each layer only overrides what it specifies; unset values carry through.

This module handles the MERGING LOGIC only. Default values live in each
script's own DEFAULTS dict.

Preset lookup order:
    1. projects/[project]/presets/[category]/[name].json  (project-level)
    2. presets/[category]/[name].json                     (universal)
"""

import os
import json


# project root is two levels up from scripts/utils/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)


def load_config(defaults, preset_name=None, preset_category=None,
                project_dir=None, cli_overrides=None):
    """
    Build final config by layering defaults → preset → CLI overrides.

    Args:
        defaults: Dict of default values from the calling script's DEFAULTS
        preset_name: Name of a preset file (without .json extension), or None
        preset_category: Which preset subfolder to look in (e.g., 'blend', 'post')
                         if None, tries to infer from the calling script's location
        project_dir: Project directory name (e.g., 'dark-city-loop'), or None
        cli_overrides: Dict of values from CLI flags that were explicitly set, or None

    Returns:
        Merged config dictionary
    """
    config = dict(defaults)

    # layer on preset if one was specified
    if preset_name:
        preset = _load_preset(preset_name, preset_category, project_dir)
        if preset:
            # don't let the preset's "name" key override anything functional
            preset_values = {k: v for k, v in preset.items() if k != 'name'}
            config.update(preset_values)

    # layer on CLI overrides (only keys that were explicitly set)
    if cli_overrides:
        config.update(cli_overrides)

    return config


def _load_preset(preset_name, category=None, project_dir=None):
    """
    Find and load a preset JSON file.

    Looks in project directory first (if provided), then universal presets.

    Args:
        preset_name: Preset filename without .json
        category: Subfolder (e.g., 'blend', 'post', 'audio')
        project_dir: Project directory name, or None

    Returns:
        Parsed preset dictionary, or None if not found
    """
    filename = f"{preset_name}.json"
    search_paths = []

    # 1. project-level presets
    if project_dir:
        project_path = os.path.join(PROJECT_ROOT, 'projects', project_dir, 'presets')
        if category:
            search_paths.append(os.path.join(project_path, category, filename))
        search_paths.append(os.path.join(project_path, filename))

    # 2. universal presets
    universal_path = os.path.join(PROJECT_ROOT, 'presets')
    if category:
        search_paths.append(os.path.join(universal_path, category, filename))
    search_paths.append(os.path.join(universal_path, filename))

    # try each path in order
    for path in search_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    preset = json.load(f)
                print(f"Loaded preset: {path}")
                return preset
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading preset {path}: {e}")
                return None

    print(f"Preset not found: {preset_name}")
    if search_paths:
        print(f"  Searched: {', '.join(search_paths)}")
    return None
