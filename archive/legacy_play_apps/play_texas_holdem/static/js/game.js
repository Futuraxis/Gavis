/* Texas Hold'em — human vs AI play app frontend. */

'use strict';

const $ = (id) => document.getElementById(id);
const SUITS = { s: '♠', h: '♥', d: '♦', c: '♣' };
const RANKS = { T: '10', J: 'J', Q: 'Q', K: 'K', A: 'A' };
const SEATS = { p_sb: '小盲', p_bb: '大盲' };
const STREETS = ['翻前', '翻牌', '转牌', '河牌'];

let gameId = null;
let busy = false;

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

/* ── Card rendering ──────────────────────────────────────────────── */

function cardEl(cardId, faceDown) {
  const el = document.createElement('div');
  el.className = 'card';
  if (faceDown || !cardId) {
    el.classList.add('back');
    return el;
  }
  const rank = RANKS[cardId[1]] || cardId[1];
  const suit = SUITS[cardId[0]] || '';
  if (cardId[0] === 'h' || cardId[0] === 'd') el.classList.add('red');
  el.innerHTML = `<span class="rank">${rank}</span><span class="suit">${suit}</span>`;
  return el;
}

function fillCards(containerId, ids, faceDown) {
  const box = $(containerId);
  box.innerHTML = '';
  const n = containerId === 'community' ? 5 : 2;
  for (let i = 0; i < n; i++) {
    const el = cardEl(ids && ids[i], faceDown);
    if (!ids || !ids[i]) el.classList.add('placeholder');
    box.appendChild(el);
  }
}

/* ── Rendering ───────────────────────────────────────────────────── */

function render(session) {
  const me = session.player_pid;
  const ai = session.ai_pid;

  $('my-name').textContent = `😊 你（${SEATS[me]}）`;
  $('ai-name').textContent = `🤖 AI（${SEATS[ai]}）`;
  $('my-stack').textContent = `${session.my_stack} 筹码`;
  $('ai-stack').textContent = `${session.ai_stack} 筹码`;
  $('my-committed').textContent = session.my_committed ? `已下注 ${session.my_committed}` : '';
  $('ai-committed').textContent = session.ai_committed ? `已下注 ${session.ai_committed}` : '';

  fillCards('my-cards', session.my_hole, false);
  fillCards('ai-cards', session.ai_hole, !session.revealed);
  fillCards('community', session.community, false);
  $('pot').textContent = `底池 ${session.pot}`;

  $('my-panel').classList.toggle('acting', !session.over && session.turn === me);
  $('ai-panel').classList.toggle('acting', !session.over && session.turn === ai);

  renderLog(session);

  if (session.over) {
    setStatus(renderResult(session), false);
    $('controls').classList.add('hidden');
  } else {
    const myTurn = session.turn === me;
    setStatus(
      myTurn
        ? `轮到你 · ${STREETS[session.street]}${session.call_to > session.my_committed ? ` · 跟注 ${session.call_to - session.my_committed}` : ''}`
        : 'AI 思考中…',
      !myTurn,
    );
    renderControls(session, myTurn);
  }
}

function renderLog(session) {
  const who = session.last_actor === session.player_pid ? '你' : 'AI';
  const map = {
    fold: '弃牌', call: '跟注', check: '过牌', raise: '加注', showdown: '摊牌',
  };
  const act = map[session.last_action];
  const log = $('log');
  if (!act || !who) { log.textContent = ''; return; }
  if (session.last_action === 'raise') {
    log.textContent = `${who} 加注到 ${session.last_actor === session.player_pid ? session.my_committed : session.ai_committed}`;
  } else if (session.last_action === 'call' || session.last_action === 'check') {
    log.textContent = `${who} ${act}`;
  } else {
    log.textContent = `${who} ${act}`;
  }
}

function renderControls(session, enabled) {
  $('controls').classList.remove('hidden');
  $('fold-btn').disabled = !enabled;
  $('call-btn').disabled = !enabled;
  $('raise-btn').disabled = !enabled;
  $('allin-btn').disabled = !enabled;
  $('raise-amount').disabled = !enabled;

  const callAdd = Math.max(0, session.call_to - session.my_committed);
  $('call-btn').textContent = callAdd > 0 ? `跟注 ${callAdd}` : '过牌';

  const amounts = session.raise_amounts.length ? session.raise_amounts : [session.call_to + 2];
  const sel = $('raise-amount');
  sel.innerHTML = '';
  for (const amt of amounts) {
    const opt = document.createElement('option');
    opt.value = String(amt);
    opt.textContent = amt >= 100 ? `${amt}（全下）` : `到 ${amt}`;
    sel.appendChild(opt);
  }
}

function renderResult(session) {
  const result = $('result');
  result.classList.remove('hidden', 'win', 'lose', 'draw');
  let cls = 'draw';
  let text = '平局 · 平分底池';
  if (session.winner) {
    const humanWon = session.winner === session.player_pid;
    cls = humanWon ? 'win' : 'lose';
    const sign = session.payoff > 0 ? '+' : '';
    text = humanWon ? `🎉 你赢了 ${sign}${session.payoff}` : `🤖 AI 获胜（你 ${sign}${session.payoff}）`;
  }
  if (session.revealed) {
    text += ` · 你的牌 ${session.my_hand_name} vs AI ${session.ai_hand_name}`;
  }
  result.classList.add(cls);
  result.textContent = text;
  $('restart-btn').classList.remove('hidden');
  return text;
}

function setStatus(text, thinking) {
  const status = $('status');
  status.textContent = text;
  status.classList.toggle('thinking', thinking);
  busy = thinking;
}

function showSetup() {
  $('setup').classList.remove('hidden');
  $('game').classList.add('hidden');
  $('restart-btn').classList.add('hidden');
}

/* ── Actions ─────────────────────────────────────────────────────── */

async function startGame() {
  busy = true;
  setStatus('AI 思考中…', true);
  $('setup').classList.add('hidden');
  $('game').classList.remove('hidden');
  $('result').classList.add('hidden');
  $('restart-btn').classList.add('hidden');
  try {
    const data = await post('/api/start', {
      playerColor: $('player-seat').value,
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

async function sendAction(choice, amount) {
  if (busy || !gameId) return;
  busy = true;
  setStatus('AI 思考中…', true);
  $('controls').classList.add('hidden');
  try {
    const data = await post('/api/move', { gameId, choice, amount: amount ?? null });
    render(data.session);
  } catch (err) {
    setStatus(`操作失败: ${err.message}`, false);
    $('controls').classList.remove('hidden');
  } finally {
    busy = false;
  }
}

function onFold() { sendAction('fold', 0); }
function onCall() { sendAction('call', null); }
function onRaise() { sendAction('raise', Number($('raise-amount').value)); }
function onAllIn() { sendAction('raise', 100); }

/* ── Init ────────────────────────────────────────────────────────── */

$('start-btn').addEventListener('click', startGame);
$('restart-btn').addEventListener('click', showSetup);
$('fold-btn').addEventListener('click', onFold);
$('call-btn').addEventListener('click', onCall);
$('raise-btn').addEventListener('click', onRaise);
$('allin-btn').addEventListener('click', onAllIn);
