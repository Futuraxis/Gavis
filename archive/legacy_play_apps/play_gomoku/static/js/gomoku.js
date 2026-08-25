/* Stochastic Gomoku — human vs AI play app frontend. */

'use strict';

const $ = (id) => document.getElementById(id);
const SIZE = 9;
const COLORS = { p_black: '黑棋', p_white: '白棋' };

let gameId = null;
let busy = false;
let lastHumanMove = null;

/* ── API helpers ─────────────────────────────────────────────────── */

async function post(path, payload) {
  const resp = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  if (!resp.ok || !data.ok) {
    throw new Error(data.error || `HTTP ${resp.status}`);
  }
  return data;
}

/* ── Rendering ───────────────────────────────────────────────────── */

function buildBoard() {
  const board = $('board');
  board.innerHTML = '';
  for (let i = 0; i < SIZE * SIZE; i++) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.dataset.index = String(i);
    cell.addEventListener('click', () => onCellClick(i));
    board.appendChild(cell);
  }
}

function render(session) {
  const cells = $('board').children;
  for (let i = 0; i < SIZE * SIZE; i++) {
    const owner = session.board[i];
    const cell = cells[i];
    cell.innerHTML = '';
    cell.classList.remove('occupied', 'ai-move', 'vanished');
    if (owner) {
      const piece = document.createElement('div');
      piece.className = `piece ${owner}`;
      cell.appendChild(piece);
      cell.classList.add('occupied');
    }
    if (session.last_ai_move === i) {
      cell.classList.add('ai-move');
    }
    if (session.last_vanish === i) {
      cell.classList.add('vanished');
    }
  }

  // Vanish toast: the human's latest stone disappeared.
  const toast = $('vanish-toast');
  if (session.last_vanish !== null && session.last_vanish !== undefined) {
    toast.classList.remove('hidden');
  } else {
    toast.classList.add('hidden');
  }

  if (session.over) {
    setStatus(renderResult(session), false);
  } else {
    const isHumanTurn = session.turn === session.player_color;
    const vanishNote = session.last_vanish !== null && session.last_vanish !== undefined
      ? '（上一手落子消失了）' : '';
    setStatus(`${isHumanTurn ? '轮到你了' : 'AI 思考中…'} · 第 ${session.round + 1} 手 ${vanishNote}`, !isHumanTurn);
    for (const cell of cells) {
      cell.classList.toggle('disabled', !isHumanTurn);
    }
  }
}

function renderResult(session) {
  const result = $('result');
  result.classList.remove('hidden', 'win', 'lose', 'draw');
  let cls = 'draw';
  let text = '平局（棋盘已满）';
  if (session.winner) {
    const humanWon = session.winner === session.player_color;
    cls = humanWon ? 'win' : 'lose';
    text = humanWon ? '🎉 你赢了！' : '🤖 AI 获胜';
  }
  result.classList.add(cls);
  result.textContent = text;
  $('restart-btn').classList.remove('hidden');
  return text;
}

function setStatus(text, thinking) {
  const status = $('status');
  status.textContent = text;
  status.classList.toggle('ai-thinking', thinking);
  busy = thinking;
}

function showSetup() {
  $('setup').classList.remove('hidden');
  $('game').classList.add('hidden');
  $('restart-btn').classList.add('hidden');
}

/* ── Actions ─────────────────────────────────────────────────────── */

async function startGame() {
  buildBoard();
  busy = true;
  setStatus('AI 思考中…', true);
  $('setup').classList.add('hidden');
  $('game').classList.remove('hidden');
  try {
    const data = await post('/api/start', {
      playerColor: $('player-color').value,
      difficulty: $('difficulty').value,
    });
    gameId = data.session.game_id;
    render(data.session);
  } catch (err) {
    setStatus(`开局失败: ${err.message}`, false);
    showSetup();
  } finally {
    busy = false;
  }
}

async function onCellClick(index) {
  if (busy || !gameId) return;
  const cells = $('board').children;
  if (cells[index].classList.contains('occupied')) return;
  busy = true;
  lastHumanMove = index;
  setStatus('AI 思考中…', true);
  try {
    const data = await post('/api/move', { gameId, cellIndex: index });
    render(data.session);
  } catch (err) {
    setStatus(`落子失败: ${err.message}`, false);
    busy = false;
  }
}

/* ── Init ────────────────────────────────────────────────────────── */

$('start-btn').addEventListener('click', startGame);
$('restart-btn').addEventListener('click', showSetup);
buildBoard();
