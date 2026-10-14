"""Check repository Markdown links offline using only the Python standard library.

Supports inline/reference links, ATX headings and explicit HTML anchors used here.
Fenced examples and inline code are ignored. External URLs and absolute evidence
paths are not opened; this checks navigation, not scientific or operational truth.
"""

from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

TARGET = r'(?:<([^>\n]+)>|((?:\\.|[^()\s]|\([^()\n]*\))+))'
INLINE_LINK = re.compile(r'\[[^\]\n]*\]\(\s*' + TARGET + r'''(?:\s+["'][^\n]*?["'])?\s*\)''')
DEFINITION = re.compile(r'^ {0,3}\[([^\]\n]+)\]:\s*' + TARGET)
REFERENCE = re.compile(r'\[([^\]\n]+)\]\[([^\]\n]*)\]')
INLINE_CODE = re.compile(r'(`+).*?\1(?!`)')


def prose(text: str) -> str:
    """Mask comments and fenced blocks while preserving diagnostic line numbers."""
    text = re.sub(r'<!--.*?-->', lambda m: '\n' * m[0].count('\n'), text, flags=re.S)
    lines = []
    fence = ''
    for line in text.splitlines():
        opening = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if fence:
            closing = r' {0,3}' + re.escape(fence[0]) + '{' + str(len(fence)) + r',}\s*'
            if re.fullmatch(closing, line):
                fence = ''
            lines.append('')
        elif opening:
            fence = opening[1]
            lines.append('')
        else:
            lines.append(line)
    return '\n'.join(lines)


def anchors(text: str) -> set[str]:
    values = set(re.findall(r'<[^>]+\b(?:id|name)=["\']([^"\']+)["\']', text))
    used: set[str] = set()
    for heading in re.findall(r'^ {0,3}#{1,6}\s+(.+)$', text, re.M):
        heading = re.sub(r'\s+#+\s*$', '', heading)
        heading = re.sub(r'!?\[([^\]]+)\]\([^)]*\)', r'\1', heading)
        heading = html.unescape(re.sub(r'<[^>]+>', '', heading))
        slug = re.sub(r'[^\w\-\s]', '', heading.lower()).replace(' ', '-')
        base, suffix = slug, 0
        while slug in used:
            suffix += 1
            slug = f'{base}-{suffix}'
        used.add(slug)
        values.add(slug)
    return values


def reference_key(label: str) -> str:
    return ' '.join(label.split()).casefold()


def destinations(text: str) -> list[tuple[int, str]]:
    lines = INLINE_CODE.sub('', text).splitlines()
    definitions = {}
    links = []
    for line in lines:
        if match := DEFINITION.match(line):
            definitions.setdefault(reference_key(match[1]), match[2] or match[3])
    for number, line in enumerate(lines, 1):
        if DEFINITION.match(line):
            continue
        links.extend((number, m[1] or m[2]) for m in INLINE_LINK.finditer(line))
        line = INLINE_LINK.sub('', line)
        for match in REFERENCE.finditer(line):
            key = reference_key(match[2] or match[1])
            if key in definitions:
                links.append((number, definitions[key]))
        line = REFERENCE.sub('', line)
        for match in re.finditer(r'\[([^\]\n]+)\]', line):
            if reference_key(match[1]) in definitions:
                links.append((number, definitions[reference_key(match[1])]))
    return links


def check_documents(root: Path, documents: list[Path]) -> tuple[list[str], int, int]:
    root = root.resolve()
    errors: list[str] = []
    checked = anchor_checks = 0
    anchor_cache: dict[Path, set[str]] = {}
    for relative in documents:
        source = root / relative
        text = prose(source.read_text(encoding='utf-8'))
        for line, target in destinations(text):
            target = html.unescape(re.sub(r'\\([() ])', r'\1', target))
            # Absolute evidence paths are intentionally outside this repository check.
            if target.startswith('/') or urlsplit(target).scheme:
                continue
            url = urlsplit(target)
            destination = (source.parent / unquote(url.path)).resolve() if url.path else source
            prefix = f'{relative}:{line}: {target}'
            checked += 1
            if not destination.is_relative_to(root):
                errors.append(f'{prefix}: relative link escapes the repository')
            elif not destination.exists():
                errors.append(f'{prefix}: missing local path')
            elif url.fragment and destination.suffix.lower() == '.md':
                anchor_checks += 1
                if destination not in anchor_cache:
                    anchor_cache[destination] = anchors(prose(destination.read_text('utf-8')))
                if unquote(url.fragment) not in anchor_cache[destination]:
                    errors.append(f'{prefix}: missing Markdown anchor')
    return errors, checked, anchor_checks


def main(root: Path | None = None) -> int:
    root = root or Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', '*.md'],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        print('Cannot enumerate repository Markdown files with git ls-files.')
        return 2
    documents = sorted({Path(name) for name in result.stdout.split('\0') if name})
    errors, links, anchor_checks = check_documents(root, documents)
    for error in errors:
        print(error)
    print(f'Checked {len(documents)} Markdown files, {links} local links, '
          f'{anchor_checks} anchors; {len(errors)} errors.')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
