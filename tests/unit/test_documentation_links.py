"""Exercise broken-navigation failures without project dependencies or external data."""

from __future__ import annotations

import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.check_docs import check_documents, main


class DocumentationLinksTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return Path(name)

    def test_chinese_encoded_paths_duplicate_headings_and_explicit_anchors(self):
        self.write('docs/说明 文件.md', '# 数据 `F/S`\n# 数据 `F/S`\n<a id="evidence"></a>\n')
        source = self.write('README.md', '[一](docs/%E8%AF%B4%E6%98%8E%20文件.md#数据-fs)\n'
                            '[二](<docs/说明 文件.md#数据-fs-1>)\n'
                            '[三](<docs/说明 文件.md#evidence> "证据")\n')
        self.assertEqual(check_documents(self.root, [source]), ([], 3, 3))

    def test_missing_file_and_stale_anchor_fail_with_source_lines(self):
        self.write('guide.md', '# Renamed\n')
        source = self.write('README.md', '[missing](absent.md)\n[old](guide.md#old)\n')
        errors, count, anchors = check_documents(self.root, [source])
        self.assertEqual((count, anchors), (2, 1))
        self.assertIn('README.md:1: absent.md: missing local path', errors)
        self.assertIn('README.md:2: guide.md#old: missing Markdown anchor', errors)

    def test_examples_comments_external_urls_and_absolute_evidence_are_not_opened(self):
        source = self.write('README.md', '```md\n[example](missing.md)\n```\n'
                            '`[example](missing.md)`\n<!-- [hidden](missing.md) -->\n'
                            '[web](https://example.invalid/not-fetched)\n'
                            '[evidence](/home/other/artifacts/run.md)\n')
        self.assertEqual(check_documents(self.root, [source]), ([], 0, 0))

    def test_reference_links_images_titles_and_balanced_parentheses(self):
        self.write('guide(v2).md', '# Use\n')
        source = self.write('README.md', '[inline](guide(v2).md#use "guide")\n'
                            '[named][Guide]\n[guide][]\n![guide]\n'
                            '[guide]: guide(v2).md#use "guide"\n')
        self.assertEqual(check_documents(self.root, [source]), ([], 4, 4))

    def test_broken_reference_destination_is_checked(self):
        source = self.write('README.md', '[guide][ref]\n\n[ref]: gone.md\n')
        errors, _, _ = check_documents(self.root, [source])
        self.assertEqual(len(errors), 1)
        self.assertIn('gone.md: missing local path', errors[0])

    def test_relative_escape_is_rejected_and_parent_link_inside_repo_is_valid(self):
        self.write('README.md', '# Home\n')
        source = self.write('docs/guide.md', '[home](../README.md#home)\n'
                            '[escape](../../outside.md)\n')
        errors, _, _ = check_documents(self.root, [source])
        self.assertEqual(len(errors), 1)
        self.assertIn('relative link escapes the repository', errors[0])

    def test_heading_inside_example_does_not_satisfy_real_anchor(self):
        source = self.write('README.md', '# Real\n[link](#example)\n````md\n'
                            '```\n# Example\n```\n````\n')
        errors, _, _ = check_documents(self.root, [source])
        self.assertEqual(len(errors), 1)
        self.assertIn('missing Markdown anchor', errors[0])

    def test_cli_includes_pending_documents_but_not_ignored_artifacts(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        self.write('.gitignore', '/artifacts/\n')
        self.write('artifacts/result.md', '[broken](missing.md)\n')
        self.write('README.md', '# Home\n')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.root), 0)
            self.write('new.md', '[broken](missing.md)\n')
            self.assertEqual(main(self.root), 1)


if __name__ == '__main__':
    unittest.main()
