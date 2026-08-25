/* Moon Chess DOM observation helpers. */

'use strict';

const DOM_BOARD_SIZE = 3;
const DOM_CELL_COUNT = DOM_BOARD_SIZE * DOM_BOARD_SIZE;

function createEmptyBoardObservation() {
  return Array.from({ length: DOM_BOARD_SIZE }, () => Array.from({ length: DOM_BOARD_SIZE }, () => null));
}

function createFullConfidence(value) {
  return Array.from({ length: DOM_BOARD_SIZE }, () => Array.from({ length: DOM_BOARD_SIZE }, () => value));
}

function pieceClassToObservation(piece) {
  const known = ['x', 'o', 'p_black', 'p_white'];
  const found = known.filter((name) => piece.classList.contains(name));
  if (found.length === 0) {
    throw new Error('unknown piece class');
  }
  if (found.length > 1) {
    throw new Error(`ambiguous piece class: ${found.join(', ')}`);
  }
  if (found[0] === 'x' || found[0] === 'p_black') {
    return 'X';
  }
  return 'O';
}

function readBoardFromDom(boardElement = document.querySelector('#board')) {
  if (!boardElement) {
    throw new Error('board container #board does not exist');
  }

  const cells = Array.from(boardElement.querySelectorAll('.cell'));
  if (cells.length !== DOM_CELL_COUNT) {
    throw new Error(`board must contain ${DOM_CELL_COUNT} .cell elements, got ${cells.length}`);
  }

  const seen = new Set();
  const boardObservation = createEmptyBoardObservation();

  for (const cell of cells) {
    const rawIndex = cell.getAttribute('data-index');
    if (!/^\d+$/.test(rawIndex || '')) {
      throw new Error(`invalid data-index: ${rawIndex}`);
    }

    const index = Number(rawIndex);
    if (index < 0 || index >= DOM_CELL_COUNT) {
      throw new Error(`data-index out of range: ${rawIndex}`);
    }
    if (seen.has(index)) {
      throw new Error(`duplicate data-index: ${rawIndex}`);
    }
    seen.add(index);

    const pieces = Array.from(cell.querySelectorAll('.piece'));
    if (pieces.length > 1) {
      throw new Error(`cell ${index} contains multiple pieces`);
    }
    if (pieces.length === 1) {
      boardObservation[Math.floor(index / DOM_BOARD_SIZE)][index % DOM_BOARD_SIZE] = pieceClassToObservation(pieces[0]);
    }
  }

  if (seen.size !== DOM_CELL_COUNT) {
    throw new Error(`board data-index set is incomplete: got ${seen.size}`);
  }

  return boardObservation;
}

function buildDomObservation({
  gameId = 'moon_demo_001',
  frameSeq = 0,
  observedAt = Date.now(),
  boardElement = document.querySelector('#board'),
} = {}) {
  return {
    gameId,
    source: 'dom',
    frameSeq,
    boardObservation: readBoardFromDom(boardElement),
    confidence: createFullConfidence(1.0),
    observedAt,
  };
}

if (typeof module !== 'undefined') {
  module.exports = {
    buildDomObservation,
    createEmptyBoardObservation,
    createFullConfidence,
    pieceClassToObservation,
    readBoardFromDom,
  };
}
