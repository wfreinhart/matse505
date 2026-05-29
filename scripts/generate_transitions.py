#!/usr/bin/env python
"""
Generate LLM-authored transitions and illustrative images between adjacent modules.

Usage:
    python scripts/generate_transitions.py --lecture HSDemo_Regression
    python scripts/generate_transitions.py --all
    python scripts/generate_transitions.py --lecture HSDemo_Regression --force

Requires:
    GEMINI_API_KEY environment variable (or .env file in project root)

Output per lecture:
    lecture_defs/<LectureID>.transitions.yaml   -- sidecar cache
    lectures/assets/<id>_illustration.png        -- per-dataset illustration
    lectures/assets/<LectureID>_illustration.png -- per-lesson illustration

The sidecar tags every entry with a content hash. Items are only regenerated
when the underlying module content changes, or when --force is passed.
"""

import os
import sys
import json
import base64
import hashlib
import argparse
import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import yaml
from dotenv import load_dotenv
from litellm import completion, image_generation

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DEFS_DIR = 'lecture_defs'
MODULES_DIR = 'library/modules'
ASSETS_DIR = 'lectures/assets'
DEFAULT_MODEL = 'gemini/gemini-3-flash-preview'
IMAGE_MODEL = 'gemini/gemini-3-pro-image-preview'

TRANSITION_SYSTEM_PROMPT = """\
You are a curriculum author writing bridging text for a high school mathematics lesson notebook.
Your task is to write a short (2-4 sentence) transition paragraph between two adjacent lesson modules.

The transition must:
- Acknowledge what students just did/learned in one sentence
- Explain in plain language why the next topic follows naturally
- Use accessible language appropriate for high school students
- Not re-explain the full details of either module -- only bridge them

Output only the transition text as plain markdown prose. No headings, no bullet points, no code.
"""

IMAGE_PROMPT_SYSTEM = """\
You write concise, effective prompts for an AI image generation model.
The images will appear in a high school math or science lesson notebook.
They are illustrative/schematic -- they accompany real Python code and charts,
so they should convey concepts visually, not replicate data plots.

Rules:
- Describe a clean, educational schematic or illustration
- No text, labels, axes, or charts in the image -- those come from the code
- Flat or semi-realistic illustration style, suitable for a textbook
- Two sentences max
"""


def api_key():
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise EnvironmentError(
            'GEMINI_API_KEY is not set.\n'
            'Add it to .env or: export GEMINI_API_KEY=your_key_here'
        )
    return key


def load_lecture_def(lecture_id):
    with open(os.path.join(DEFS_DIR, f'{lecture_id}.yaml')) as f:
        return yaml.safe_load(f)


def load_module(module_id):
    with open(os.path.join(MODULES_DIR, f'{module_id}.json')) as f:
        return json.load(f)


def expand_module_ids(lecture_def):
    ids = []
    for item in lecture_def.get('modules', []):
        if 'id' in item:
            if os.path.exists(os.path.join(MODULES_DIR, f"{item['id']}.json")):
                ids.append(item['id'])
    return ids


def concept_hash(mod):
    """Hash based only on the conceptual description (markdown). Used for image caching.
    Code implementation changes (e.g. adding a filter) do not bust image cache."""
    return hashlib.md5(mod.get('markdown_content', '').encode()).hexdigest()[:12]


def pair_hash(mod_a, mod_b):
    """Full hash of both modules' markdown + code. Used for transition caching."""
    raw = (
        mod_a.get('markdown_content', '') + mod_a.get('raw_code', '') +
        mod_b.get('markdown_content', '') + mod_b.get('raw_code', '')
    )
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def describe_module(mod):
    title_lines = [l for l in mod.get('markdown_content', '').splitlines() if l.startswith('#')]
    title = title_lines[0].lstrip('# ').strip() if title_lines else mod['id']
    paras = [
        l.strip() for l in mod.get('markdown_content', '').splitlines()
        if l.strip() and not l.strip().startswith('#')
    ]
    summary = ' '.join(paras[:3])[:400]
    return f'Title: "{title}"\nSummary: {summary}'


# ── Text generation ────────────────────────────────────────────────────────────

def generate_transition(mod_a, mod_b, model=DEFAULT_MODEL):
    user_msg = (
        f'The module students just completed:\n{describe_module(mod_a)}\n\n'
        f'The module they are about to start:\n{describe_module(mod_b)}\n\n'
        'Write the transition paragraph.'
    )
    response = completion(
        model=model,
        messages=[
            {'role': 'system', 'content': TRANSITION_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_msg},
        ],
        api_key=api_key(),
    )
    return response.choices[0].message.content.strip()


def generate_image_prompt(description, kind, model=DEFAULT_MODEL):
    """Ask the text LLM to write a good image-gen prompt given context."""
    user_msg = (
        f'Write an image generation prompt for a {kind} illustration.\n\n'
        f'Context:\n{description}'
    )
    response = completion(
        model=model,
        messages=[
            {'role': 'system', 'content': IMAGE_PROMPT_SYSTEM},
            {'role': 'user', 'content': user_msg},
        ],
        api_key=api_key(),
    )
    return response.choices[0].message.content.strip()


# ── Image generation ───────────────────────────────────────────────────────────

def generate_and_save_image(prompt, out_path, image_model=IMAGE_MODEL):
    response = image_generation(
        model=image_model,
        prompt=prompt,
        n=1,
        api_key=api_key(),
    )
    img_data = response.data[0]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if img_data.b64_json:
        with open(out_path, 'wb') as f:
            f.write(base64.b64decode(img_data.b64_json))
    elif img_data.url:
        import urllib.request
        urllib.request.urlretrieve(img_data.url, out_path)
    else:
        raise ValueError('image_generation returned neither b64_json nor url')

    return out_path


# ── Sidecar I/O ────────────────────────────────────────────────────────────────

def load_sidecar(lecture_id):
    path = os.path.join(DEFS_DIR, f'{lecture_id}.transitions.yaml')
    if os.path.exists(path):
        return yaml.safe_load(open(path)) or {}
    return {}


def save_sidecar(lecture_id, data):
    path = os.path.join(DEFS_DIR, f'{lecture_id}.transitions.yaml')
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f'  Saved -> {path}')


# ── Main processor ─────────────────────────────────────────────────────────────

def process_lecture(lecture_id, force=False, model=DEFAULT_MODEL, image_model=IMAGE_MODEL):
    print(f'\n=== {lecture_id} ===')

    lect_def = load_lecture_def(lecture_id)
    module_ids = expand_module_ids(lect_def)

    if len(module_ids) < 2:
        print('  Only one module -- nothing to generate.')
        return

    sidecar = load_sidecar(lecture_id)
    cached_transitions = {
        (t['after_id'], t['before_id']): t
        for t in sidecar.get('transitions', [])
    }
    cached_images = sidecar.get('images', {})

    updated = False
    new_transitions = []

    for i in range(len(module_ids) - 1):
        id_a = module_ids[i]
        id_b = module_ids[i + 1]
        pair_key = (id_a, id_b)

        mod_a = load_module(id_a)
        mod_b = load_module(id_b)
        h = pair_hash(mod_a, mod_b)

        entry = dict(cached_transitions.get(pair_key, {}))
        if entry and entry.get('hash') == h and not force:
            print(f'  [transition {id_a} -> {id_b}]  cached')
        else:
            print(f'  [transition {id_a} -> {id_b}]  generating...', end='', flush=True)
            entry = {
                'after_id': id_a,
                'before_id': id_b,
                'hash': h,
                'markdown': generate_transition(mod_a, mod_b, model=model),
            }
            print(' done')
            updated = True

        new_transitions.append(entry)

    # ── Dataset image (tied to the first module in the lecture) ───────────────
    dataset_mod_id = module_ids[0]
    dataset_mod = load_module(dataset_mod_id)
    dataset_hash = concept_hash(dataset_mod)
    dataset_img_path = os.path.join(ASSETS_DIR, f'{dataset_mod_id}_illustration.png')

    cached_ds = cached_images.get('dataset', {})
    if cached_ds.get('hash') == dataset_hash and os.path.exists(dataset_img_path) and not force:
        print(f'  [image dataset]  cached ({dataset_img_path})')
        new_dataset_image = cached_ds
    else:
        print(f'  [image dataset]  generating prompt...', end='', flush=True)
        ds_prompt = generate_image_prompt(describe_module(dataset_mod), kind='dataset', model=model)
        print(' generating image...', end='', flush=True)
        generate_and_save_image(ds_prompt, dataset_img_path, image_model=image_model)
        print(' done')
        new_dataset_image = {
            'module_id': dataset_mod_id,
            'hash': dataset_hash,
            'path': dataset_img_path,
            'alt': f'Illustration representing the {dataset_mod_id} dataset',
        }
        updated = True

    # ── Lesson image (specific to this dataset + task combination) ────────────
    lesson_img_path = os.path.join(ASSETS_DIR, f'{lecture_id}_illustration.png')
    all_mods = [load_module(mid) for mid in module_ids]
    lesson_hash = hashlib.md5(
        ''.join(m.get('markdown_content', '') for m in all_mods).encode()
    ).hexdigest()[:12]

    cached_lesson = cached_images.get('lesson', {})
    if cached_lesson.get('hash') == lesson_hash and os.path.exists(lesson_img_path) and not force:
        print(f'  [image lesson]   cached ({lesson_img_path})')
        new_lesson_image = cached_lesson
    else:
        lesson_context = (
            f'Dataset: {describe_module(dataset_mod)}\n\n'
            f'Analysis task: {describe_module(load_module(module_ids[-1]))}'
        )
        print(f'  [image lesson]   generating prompt...', end='', flush=True)
        lesson_prompt = generate_image_prompt(lesson_context, kind='lesson combining data and analysis', model=model)
        print(' generating image...', end='', flush=True)
        generate_and_save_image(lesson_prompt, lesson_img_path, image_model=image_model)
        print(' done')
        new_lesson_image = {
            'hash': lesson_hash,
            'path': lesson_img_path,
            'alt': f'Concept illustration for {lecture_id}',
        }
        updated = True

    if updated or not sidecar:
        save_sidecar(lecture_id, {
            'lecture': lecture_id,
            'generated_at': datetime.datetime.utcnow().isoformat(),
            'model': model,
            'image_model': image_model,
            'transitions': new_transitions,
            'images': {
                'dataset': new_dataset_image,
                'lesson': new_lesson_image,
            },
        })
    else:
        print('  All entries up-to-date, sidecar unchanged.')


def main():
    parser = argparse.ArgumentParser(
        description='Generate LLM transitions and illustrations for lecture modules.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lecture', help='Lecture ID (e.g. HSDemo_Regression)')
    group.add_argument('--all', action='store_true', help='Process all lectures in lecture_defs/')
    parser.add_argument('--force', action='store_true', help='Regenerate all entries even if cached')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'Text LiteLLM model slug (default: {DEFAULT_MODEL})')
    parser.add_argument('--image-model', default=IMAGE_MODEL,
                        help=f'Image LiteLLM model slug (default: {IMAGE_MODEL})')
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    if args.lecture:
        process_lecture(args.lecture, force=args.force, model=args.model, image_model=args.image_model)
    else:
        defs = sorted(Path(DEFS_DIR).glob('*.yaml'))
        lecture_defs = [d.stem for d in defs if '.transitions' not in d.stem]
        for lid in lecture_defs:
            process_lecture(lid, force=args.force, model=args.model, image_model=args.image_model)


if __name__ == '__main__':
    main()
