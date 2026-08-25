"""Tests for Moon Chess browser DOM observation JavaScript."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

#: 独立应用已退役归档（C6）：JS 仍保留在 archive/legacy_play_apps/ 供本测试引用。
JS_PATH = Path("archive/legacy_play_apps/play_moon_chess/static/js/dom_observation.js")


def run_node_case(case_name: str) -> dict[str, object]:
    """Run one DOM observation JavaScript test case under Node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    script = f"""
const assert = require('assert');
const helpers = require('./{JS_PATH.as_posix()}');

class ClassList {{
  constructor(classes) {{
    this.classes = new Set(classes);
  }}
  contains(name) {{
    return this.classes.has(name);
  }}
}}

class Element {{
  constructor({{ classes = [], index = null, children = [] }} = {{}}) {{
    this.classList = new ClassList(classes);
    this.children = children;
    this.attrs = {{}};
    if (index !== null) this.attrs['data-index'] = String(index);
  }}
  getAttribute(name) {{
    return Object.prototype.hasOwnProperty.call(this.attrs, name) ? this.attrs[name] : null;
  }}
  querySelectorAll(selector) {{
    if (selector === '.cell') {{
      return this.children.filter((child) => child.classList.contains('cell'));
    }}
    if (selector === '.piece') {{
      return this.children.filter((child) => child.classList.contains('piece'));
    }}
    throw new Error(`unsupported selector: ${{selector}}`);
  }}
}}

function piece(cls) {{
  return new Element({{ classes: ['piece', cls] }});
}}

function board(cells) {{
  return new Element({{ children: cells }});
}}

function cell(index, cls = null) {{
  const children = cls === null ? [] : [piece(cls)];
  return new Element({{ classes: ['cell'], index, children }});
}}

const cases = {{
  missing_board_errors() {{
    assert.throws(() => helpers.readBoardFromDom(null), /#board/);
  }},
  empty_board() {{
    assert.deepStrictEqual(helpers.readBoardFromDom(board(Array.from({{ length: 9 }}, (_, i) => cell(i)))), [
      [null, null, null],
      [null, null, null],
      [null, null, null],
    ]);
  }},
  only_x() {{
    assert.deepStrictEqual(helpers.readBoardFromDom(board([
      cell(0, 'x'), cell(1), cell(2), cell(3), cell(4), cell(5), cell(6), cell(7), cell(8),
    ]))[0][0], 'X');
  }},
  only_o() {{
    assert.deepStrictEqual(helpers.readBoardFromDom(board([
      cell(0, 'o'), cell(1), cell(2), cell(3), cell(4), cell(5), cell(6), cell(7), cell(8),
    ]))[0][0], 'O');
  }},
  mixed() {{
    assert.deepStrictEqual(helpers.readBoardFromDom(board([
      cell(0, 'x'), cell(1), cell(2, 'o'),
      cell(3), cell(4, 'x'), cell(5),
      cell(6, 'o'), cell(7), cell(8),
    ])), [
      ['X', null, 'O'],
      [null, 'X', null],
      ['O', null, null],
    ]);
  }},
  shuffled_indexes() {{
    assert.deepStrictEqual(helpers.readBoardFromDom(board([
      cell(8, 'o'), cell(0, 'x'), cell(4, 'x'),
      cell(1), cell(2, 'o'), cell(3),
      cell(5), cell(6, 'o'), cell(7),
    ])), [
      ['X', null, 'O'],
      [null, 'X', null],
      ['O', null, 'O'],
    ]);
  }},
  missing_cell_errors() {{
    assert.throws(() => helpers.readBoardFromDom(board(Array.from({{ length: 8 }}, (_, i) => cell(i)))), /9 .cell/);
  }},
  duplicate_index_errors() {{
    assert.throws(() => helpers.readBoardFromDom(board(Array.from({{ length: 9 }}, (_, i) => cell(i === 8 ? 7 : i)))), /duplicate/);
  }},
  non_numeric_index_errors() {{
    const cells = Array.from({{ length: 9 }}, (_, i) => cell(i));
    cells[8].attrs['data-index'] = 'bad';
    assert.throws(() => helpers.readBoardFromDom(board(cells)), /invalid data-index/);
  }},
  invalid_index_errors() {{
    assert.throws(() => helpers.readBoardFromDom(board(Array.from({{ length: 9 }}, (_, i) => cell(i === 8 ? 99 : i)))), /out of range/);
  }},
  unknown_piece_errors() {{
    assert.throws(() => helpers.readBoardFromDom(board([
      cell(0, 'z'), cell(1), cell(2), cell(3), cell(4), cell(5), cell(6), cell(7), cell(8),
    ])), /unknown piece class/);
  }},
}};

cases['{case_name}']();
console.log(JSON.stringify({{ ok: true, caseName: '{case_name}' }}));
"""
    result = subprocess.run(
        [node, "-e", script],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "case_name",
    [
        "missing_board_errors",
        "empty_board",
        "only_x",
        "only_o",
        "mixed",
        "shuffled_indexes",
        "missing_cell_errors",
        "duplicate_index_errors",
        "non_numeric_index_errors",
        "invalid_index_errors",
        "unknown_piece_errors",
    ],
)
def test_dom_observation_js_cases(case_name: str) -> None:
    result = run_node_case(case_name)

    assert result == {"ok": True, "caseName": case_name}
