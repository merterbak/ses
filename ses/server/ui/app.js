/* ses playground.
 *
 * Sections, in order:
 *   helpers · navigation · model pickers · chat options · speech playback
 *   markdown · conversations · chat streaming · speak page · transcribe page
 *   models page · startup
 */

'use strict';

const $ = selector => document.querySelector(selector);

const DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant. Answer clearly and conversationally.';
const MIN_INDICATOR_MS = 650;      // keep the waiting indicator on screen long enough to read
const LOAD_SECONDS_PER_GB = 1.7;   // rough cold-load estimate for the progress bar
const HUB_PAGE_SIZE = 10;  // a top ten reads at a glance; thirty rows is a wall

// ---------------------------------------------------------------- helpers

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      message = (await response.json()).error.message;
    } catch (ignored) {
      // no JSON body; the status line is the best we have
    }
    throw new Error(message);
  }
  return response;
}

const getJSON = async path => (await api(path)).json();

const postJSON = (path, body) => api(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

/** Is this model already in memory? Lets us warn about a slow first run. */
async function isWarm(model) {
  try {
    const { models } = await getJSON('/api/ps');
    return models.some(loaded => loaded.name === model);
  } catch (ignored) {
    return true; // can't tell — don't cry wolf
  }
}

function setErr(element, message) {
  element.textContent = message;
  element.classList.add('err');
}

function setMeta(element, message) {
  element.textContent = message;
  element.classList.remove('err');
}

/** Show "<label> · Ns", counting up until the returned stop function is called. */
function metaLoading(element, label) {
  const started = performance.now();
  const tick = () => setMeta(element, `${label} · ${((performance.now() - started) / 1000).toFixed(0)}s`);
  tick();
  const timer = setInterval(tick, 250);
  return () => clearInterval(timer);
}

const escapeHtml = value => String(value).replace(
  /[&<>"]/g,
  character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[character]),
);

function fmtCount(n) {
  const count = Number(n);
  if (!Number.isFinite(count)) return n == null || n === '' ? '—' : String(n);
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(0)}k`;
  return `${count}`;
}

function fmtSize(gb) {
  const size = Number(gb);
  if (!Number.isFinite(size) || size <= 0) return '—';
  return size >= 1 ? `${size.toFixed(1)} GB` : `${Math.round(size * 1024)} MB`;
}

function fmtParams(count) {
  if (!count) return '—';
  return count >= 1e9 ? `${(count / 1e9).toFixed(count >= 1e10 ? 0 : 1)}B` : `${Math.round(count / 1e6)}M`;
}

function fmtField(value) {
  if (Array.isArray(value)) return value.map(item => String(item)).join(', ');
  if (value && typeof value === 'object' && value.label) return String(value.label);
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  return value == null ? '' : String(value);
}

function safeLink(value) {
  if (!value) return null;
  try {
    const url = new URL(String(value), window.location.href);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : null;
  } catch (ignored) {
    return null;
  }
}

function githubLink(value) {
  if (!value) return null;
  let candidate = String(value).trim();
  if (!/^[a-z][a-z0-9+.-]*:/i.test(candidate)) {
    candidate = candidate.startsWith('github.com/')
      ? `https://${candidate}`
      : `https://github.com/${candidate.replace(/^\/+/, '')}`;
  }
  return safeLink(candidate);
}

function pullCommand(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function copyCmd(element, command) {
  navigator.clipboard.writeText(command);
  const original = element.textContent;
  element.textContent = 'copied!';
  setTimeout(() => { element.textContent = original; }, 1000);
}

// ---------------------------------------------------------------- navigation

let catalogLoaded = false;

function showPage(name) {
  document.querySelectorAll('.page').forEach(page => {
    page.classList.toggle('active', page.id === `page-${name}`);
  });
  document.querySelectorAll('nav.tabs button').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.page === name);
  });

  if (name === 'chat') $('#chatText').focus();
  if (name === 'models' && !catalogLoaded) {
    catalogLoaded = true;
    loadCatalog();
  }
}

document.querySelectorAll('nav.tabs button').forEach(tab => {
  tab.addEventListener('click', () => showPage(tab.dataset.page));
});

// ---------------------------------------------------------------- model pickers

function fillSelect(select, models, emptyMessage) {
  select.innerHTML = '';
  if (!models.length) {
    select.append(new Option(emptyMessage, ''));
    return;
  }
  models.forEach(model => select.append(new Option(model.id, model.id)));
}

/** Load a TTS model's voices into a <select>, preferring af_heart. */
async function loadVoices(modelSelect, voiceSelect) {
  const select = $(voiceSelect);
  if (!select) return;
  select.innerHTML = '';
  select.append(new Option('loading voices…', ''));
  try {
    const model = encodeURIComponent($(modelSelect).value);
    const { voices } = await getJSON(`/v1/audio/voices?model=${model}`);
    select.innerHTML = '';
    voices.forEach(voice => select.append(new Option(voice, voice)));
    select.value = voices.includes('af_heart') ? 'af_heart' : voices[0];
  } catch (ignored) {
    select.innerHTML = '';
    select.append(new Option('voices unavailable', ''));
  }
}

async function init() {
  try {
    await api('/healthz');
    $('#dot').classList.add('on');
    $('#statusText').textContent = 'server up';
  } catch (ignored) {
    $('#statusText').textContent = 'server unreachable';
    return;
  }

  const { data } = await getJSON('/v1/models');
  const tts = data.filter(model => model.ses && model.ses.kind === 'tts');
  const stt = data.filter(model => model.ses && model.ses.kind === 'stt');

  fillSelect($('#ttsModel'), tts, 'no TTS model — run: ses pull kokoro');
  fillSelect($('#sttModel'), stt, 'no STT model — run: ses pull whisper-base');
  fillSelect($('#chatTtsModel'), tts, 'no TTS model');
  fillSelect($('#chatSttModel'), stt, 'no STT model');

  $('#speakBtn').disabled = !tts.length;
  $('#recBtn').disabled = !stt.length;
  if (!stt.length) $('#chatRecBtn').disabled = true;

  if (tts.length) {
    loadVoices('#ttsModel', '#voice');
    loadVoices('#chatTtsModel', '#chatVoice');
  }
}

// ---------------------------------------------------------------- chat options

/** Everything `ses talk` accepts as a flag, read straight off the panel. */
const chatOptions = {
  get ttsModel() { return $('#chatTtsModel').value || $('#ttsModel').value; },
  get voice() { return $('#chatVoice').value || $('#voice').value; },
  get speed() { return +$('#chatSpeed').value || 1; },
  get sttModel() { return $('#chatSttModel').value || $('#sttModel').value; },
  get think() { return $('#chatThink').checked; },
  get system() { return $('#chatSystem').value.trim() || DEFAULT_SYSTEM_PROMPT; },
  get speakReplies() { return $('#speakReplies').checked; },
};

$('#chatSystem').placeholder = DEFAULT_SYSTEM_PROMPT;

$('#optionsBtn').addEventListener('click', event => {
  event.stopPropagation();
  $('#optionsPanel').hidden = !$('#optionsPanel').hidden;
});

document.addEventListener('click', event => {
  const panel = $('#optionsPanel');
  const insidePanel = panel.contains(event.target);
  const onButton = event.target === $('#optionsBtn');
  if (!panel.hidden && !insidePanel && !onButton) panel.hidden = true;
});

$('#chatTtsModel').addEventListener('change', () => loadVoices('#chatTtsModel', '#chatVoice'));
$('#chatSpeed').addEventListener('input', () => {
  $('#chatSpeedVal').textContent = `${(+$('#chatSpeed').value).toFixed(1)}×`;
});

// ---------------------------------------------------------------- speech playback

// Clips are fetched in sentence order and played back to back. `generation`
// invalidates anything queued or in flight when we stop or start a new turn.
const replyAudio = new Audio();
let audioQueue = [];
let audioPlaying = false;
let speechChain = Promise.resolve();
let generation = 0;

replyAudio.addEventListener('ended', () => {
  audioPlaying = false;
  playNextClip();
});

function playNextClip() {
  if (audioPlaying || !audioQueue.length) return;
  audioPlaying = true;
  replyAudio.src = URL.createObjectURL(audioQueue.shift());
  replyAudio.play().catch(() => { audioPlaying = false; });
}

/** Cut the current reply off: stop audio, drop the queue, cancel pending fetches. */
function stopSpeaking() {
  generation++;
  audioQueue = [];
  audioPlaying = false;
  try {
    replyAudio.pause();
    replyAudio.currentTime = 0;
  } catch (ignored) {
    // nothing was playing
  }
  speechChain = Promise.resolve();
  updateStopButton();
}

function updateStopButton() {
  const button = $('#stopSpeakBtn');
  if (button) button.disabled = !audioPlaying && !audioQueue.length;
}
setInterval(updateStopButton, 400);

function enqueueSpeech(text) {
  const model = chatOptions.ttsModel;
  if (!chatOptions.speakReplies || !model) return;

  const clean = (text || '').replace(/[*_#`]+/g, '').slice(0, 2000);
  if (!clean.trim()) return;

  const mine = generation;
  speechChain = speechChain.then(async () => {
    if (mine !== generation) return; // cancelled before we asked
    const response = await postJSON('/v1/audio/speech', {
      model,
      input: clean,
      voice: chatOptions.voice,
      speed: chatOptions.speed,
      response_format: 'wav',
    });
    if (mine !== generation) return; // cancelled while fetching
    audioQueue.push(await response.blob());
    updateStopButton();
    playNextClip();
  }).catch(() => {});
}

$('#stopSpeakBtn').addEventListener('click', stopSpeaking);

/** Mirror of ses/core/sentences.py, so audio can start on the first sentence. */
function makeChunker(minChars) {
  let buffer = '';
  const boundary = /(?<=[.!?…])["')\]]*\s+|\n{2,}/;

  return {
    feed(text) {
      buffer += text;
      const parts = buffer.split(boundary);
      if (parts.length <= 1) return [];

      buffer = parts.pop();
      const sentences = [];
      let pending = '';
      for (const part of parts) {
        pending = pending ? `${pending} ${part}` : part;
        if (pending.length >= minChars) {
          sentences.push(pending);
          pending = '';
        }
      }
      if (pending) buffer = `${pending} ${buffer}`.trim();
      return sentences;
    },
    flush() {
      const rest = buffer.trim();
      buffer = '';
      return rest;
    },
  };
}

// ---------------------------------------------------------------- markdown

function renderInline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

/** Line-based on purpose: it has to look right mid-stream, fences still open. */
function mdToHtml(source) {
  let html = '';
  let inCodeBlock = false;
  let code = '';
  let listTag = null;

  const closeList = () => {
    if (listTag) {
      html += `</${listTag}>`;
      listTag = null;
    }
  };

  for (const line of (source || '').split('\n')) {
    if (line.startsWith('```')) {
      if (inCodeBlock) {
        html += `<pre><code>${escapeHtml(code)}</code></pre>`;
        inCodeBlock = false;
        code = '';
      } else {
        closeList();
        inCodeBlock = true;
        code = '';
      }
      continue;
    }
    if (inCodeBlock) {
      code += `${line}\n`;
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = Math.min(heading[1].length + 1, 4); // h1 would fight the page title
      html += `<h${level}>${renderInline(heading[2])}</h${level}>`;
      continue;
    }

    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+\.\s+(.*)$/);
    if (bullet || numbered) {
      const wanted = bullet ? 'ul' : 'ol';
      if (listTag !== wanted) {
        closeList();
        html += `<${wanted}>`;
        listTag = wanted;
      }
      html += `<li>${renderInline((bullet || numbered)[1])}</li>`;
      continue;
    }

    if (!line.trim()) {
      closeList();
      continue;
    }
    closeList();
    html += `<p>${renderInline(line)}</p>`;
  }

  if (inCodeBlock) html += `<pre><code>${escapeHtml(code)}</code></pre>`;
  closeList();
  return html;
}

// ---------------------------------------------------------------- conversations

const chatState = { busy: false, available: false, currentId: null };

async function initBrain() {
  const status = $('#brainStatus');
  const select = $('#brainModel');
  try {
    const brain = await getJSON('/api/brain');
    if (!brain.available) {
      status.textContent = 'no local LLM — start Ollama or LM Studio, then reload';
      $('#chatSendBtn').disabled = true;
      $('#chatRecBtn').disabled = true;
      $('#newChatBtn').disabled = true;
      return;
    }

    select.innerHTML = '';
    brain.models.forEach(model => select.append(new Option(model, model)));
    if (brain.default) select.value = brain.default;
    status.textContent = `via ${brain.backend}`;
    chatState.available = true;
    await loadConversations();
  } catch (ignored) {
    status.textContent = 'brain check failed';
  }
}

async function loadConversations() {
  try {
    const { conversations } = await getJSON('/api/conversations');
    renderConversations(conversations);
  } catch (ignored) {
    // sidebar just stays as it was
  }
}

function renderConversations(list) {
  const container = $('#convList');
  container.innerHTML = '';
  if (!list.length) {
    container.innerHTML = '<div class="convempty">No saved chats yet.</div>';
    return;
  }

  list.forEach(conversation => {
    const item = document.createElement('div');
    item.className = 'convitem' + (conversation.id === chatState.currentId ? ' active' : '');
    item.innerHTML = '<span class="ct"></span><span class="cx" title="delete">✕</span>';
    item.querySelector('.ct').textContent = conversation.title || 'New chat';
    item.querySelector('.ct').onclick = () => selectConversation(conversation.id);
    item.querySelector('.cx').onclick = event => {
      event.stopPropagation();
      deleteConversation(conversation.id);
    };
    container.append(item);
  });
}

function clearLog(hint) {
  $('#chatlog').innerHTML = hint ? `<div class="hint">${escapeHtml(hint)}</div>` : '';
}

async function selectConversation(id) {
  stopSpeaking(); // switching chats silences the previous reply
  chatState.currentId = id;
  clearLog('');
  try {
    const conversation = await getJSON(`/api/conversations/${id}`);
    (conversation.messages || []).forEach(message => {
      bubble(message.content, message.role === 'user' ? 'you' : 'bot');
    });
    if (!conversation.messages || !conversation.messages.length) {
      clearLog('Empty conversation — say something.');
    }
  } catch (ignored) {
    // leave the log empty
  }
  await loadConversations();
}

async function newChat() {
  stopSpeaking();
  chatState.currentId = null;
  clearLog('New chat — ask anything.');
  await loadConversations();
  $('#chatText').focus();
}

async function deleteConversation(id) {
  try {
    await api(`/api/conversations/${id}`, { method: 'DELETE' });
  } catch (ignored) {
    // it may already be gone
  }
  if (chatState.currentId === id) await newChat();
  await loadConversations();
}

$('#newChatBtn').addEventListener('click', newChat);

// ---------------------------------------------------------------- chat streaming

function bubble(text, kind) {
  const log = $('#chatlog');
  if (log.firstElementChild && log.firstElementChild.classList.contains('hint')) log.innerHTML = '';

  const element = document.createElement('div');
  element.className = `msg ${kind}`;
  if (kind === 'bot') renderReply(element, '', text);
  else element.textContent = text;

  log.append(element);
  element.scrollIntoView({ block: 'end' });
  return element;
}

/** Update a reply in place, so expanding the reasoning survives each repaint. */
function renderReply(element, thinking, content) {
  if (thinking && thinking.trim()) {
    let details = element.querySelector('details.think');
    if (!details) {
      details = document.createElement('details');
      details.className = 'think';
      details.open = true; // visible by default while generating
      details.innerHTML = '<summary>💭 thinking</summary><div class="think-body"></div>';
      element.prepend(details);
    }
    details.querySelector('.think-body').textContent = thinking.trim();
  }

  let body = element.querySelector('.mdbody');
  if (!body) {
    body = document.createElement('div');
    body.className = 'mdbody';
    element.append(body);
  }
  body.innerHTML = mdToHtml(content);
}

/** Animated waiting indicator. With an estimate it shows a percentage instead. */
function waitingIndicator(label, sub, estimateSeconds) {
  const log = $('#chatlog');
  if (log.firstElementChild && log.firstElementChild.classList.contains('hint')) log.innerHTML = '';

  const element = document.createElement('div');
  element.className = 'msg bot think';
  element.innerHTML =
    '<div class="thinkcap"><span class="dots"><i></i><i></i><i></i></span>' +
    `<span class="tlabel"></span><span class="telapsed">${estimateSeconds ? '0%' : '0s'}</span></div>` +
    `<div class="thinkbar${estimateSeconds ? ' det' : ''}"><i></i></div>` +
    (sub ? '<div class="thinksub"></div>' : '');
  element.querySelector('.tlabel').textContent = label;
  if (sub) element.querySelector('.thinksub').textContent = sub;

  log.append(element);
  element.scrollIntoView({ block: 'end' });

  const fill = element.querySelector('.thinkbar i');
  const readout = element.querySelector('.telapsed');
  const started = performance.now();
  const timer = setInterval(() => {
    const elapsed = (performance.now() - started) / 1000;
    if (estimateSeconds) {
      // Loading weights reports no real progress, so this is time-based and
      // deliberately stops short of 100% until the model is actually ready.
      const percent = Math.min(95, (100 * elapsed) / estimateSeconds);
      fill.style.width = `${percent.toFixed(0)}%`;
      readout.textContent = `${percent.toFixed(0)}%`;
    } else {
      readout.textContent = `${elapsed.toFixed(0)}s`;
    }
  }, 200);

  return {
    /** Turn this indicator into the reply bubble itself. */
    toBubble() {
      clearInterval(timer);
      element.className = 'msg bot';
      element.innerHTML = '';
      return element;
    },
  };
}

// Brain sizes, for the cold-load estimate. Fetched once from the catalog.
let brainSizes = null;

async function brainSizeGb(model) {
  if (brainSizes === null) {
    brainSizes = {};
    try {
      const catalog = await getJSON('/api/catalog');
      (catalog.curated.brains || []).forEach(brain => { brainSizes[brain.name] = brain.size_gb; });
    } catch (ignored) {
      // fall through to the default below
    }
  }
  return brainSizes[model] || brainSizes[`${model}:latest`] || 5;
}

async function sendChat(text) {
  text = (text || '').trim();
  if (!text || chatState.busy || !chatState.available) return;

  stopSpeaking(); // a new question cuts off the previous spoken reply
  chatState.busy = true;
  $('#chatSendBtn').disabled = true;
  const meta = $('#chatMeta');
  setMeta(meta, '');

  if (!chatState.currentId) {
    try {
      const created = await (await postJSON('/api/conversations', {
        brain: $('#brainModel').value,
      })).json();
      chatState.currentId = created.id;
    } catch (error) {
      setErr(meta, error.message);
      chatState.busy = false;
      $('#chatSendBtn').disabled = false;
      return;
    }
  }

  bubble(text, 'you');

  const model = $('#brainModel').value;
  const cold = await isModelCold(model);
  const estimate = cold ? Math.max(4, (await brainSizeGb(model)) * LOAD_SECONDS_PER_GB) : 0;
  const indicator = waitingIndicator(
    cold ? `Loading ${model} into memory` : `${model.split(':')[0] || 'assistant'} is thinking`,
    cold ? 'first run only — reading the model into memory, then it stays warm' : '',
    estimate,
  );
  const minVisible = new Promise(resolve => setTimeout(resolve, MIN_INDICATOR_MS));

  const started = performance.now();
  const chunker = makeChunker(24);
  const showThinking = chatOptions.think; // fixed for this turn, even if toggled mid-reply
  let answer = '';
  let thinking = '';
  let modelName = '';
  let replyBubble = null;

  const repaint = () => {
    renderReply(replyBubble, thinking, answer);
    replyBubble.scrollIntoView({ block: 'end' });
  };

  try {
    const response = await postJSON('/api/chat', {
      model,
      conversation: chatState.currentId,
      prompt: text,
      system: chatOptions.system,
      think: showThinking,
      stream: true,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });

      let newline;
      while ((newline = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, newline);
        pending = pending.slice(newline + 1);
        if (!line.trim()) continue;

        const event = JSON.parse(line);
        if (event.error) throw new Error(event.error);
        modelName = event.model || modelName;

        // Swap the indicator for the reply bubble on the first thing we'll show.
        const firstVisible = event.token || (event.thinking && showThinking);
        if (firstVisible && !replyBubble) {
          await minVisible;
          replyBubble = indicator.toBubble();
        }
        // Some models inline <think> tags whatever we ask, so filter here too —
        // that's what makes the toggle reliable rather than a suggestion.
        if (event.thinking && showThinking) {
          thinking += event.thinking;
          repaint();
        }
        if (event.token) {
          answer += event.token;
          repaint();
          chunker.feed(event.token).forEach(enqueueSpeech);
        }
        if (event.done && event.message) answer = event.message.content;
      }
    }

    const remainder = chunker.flush();
    if (remainder) enqueueSpeech(remainder);

    if (!replyBubble) {
      await minVisible;
      replyBubble = indicator.toBubble();
    }
    renderReply(replyBubble, thinking, answer || '(empty reply)');
    setMeta(meta, `${modelName || 'brain'} · streamed in ` +
      `${((performance.now() - started) / 1000).toFixed(1)}s — saved to history`);
    await loadConversations();
  } catch (error) {
    const target = replyBubble || indicator.toBubble();
    target.textContent = error.message;
    target.classList.add('err');
  }

  chatState.busy = false;
  $('#chatSendBtn').disabled = false;
}

async function isModelCold(model) {
  try {
    const brain = await getJSON('/api/brain');
    return Boolean(brain.loaded) && !brain.loaded.includes(model);
  } catch (ignored) {
    return false;
  }
}

function sendFromInput() {
  sendChat($('#chatText').value);
  $('#chatText').value = '';
}

$('#chatSendBtn').addEventListener('click', sendFromInput);
$('#chatText').addEventListener('keydown', event => {
  if (event.key === 'Enter') sendFromInput();
});

// ---------------------------------------------------------------- speak page

const speakState = { wavBlob: null };

$('#ttsModel').addEventListener('change', () => loadVoices('#ttsModel', '#voice'));
$('#speed').addEventListener('input', () => {
  $('#speedVal').textContent = `${(+$('#speed').value).toFixed(1)}×`;
});

$('#speakBtn').addEventListener('click', async () => {
  const button = $('#speakBtn');
  const meta = $('#ttsMeta');
  button.disabled = true;
  button.textContent = 'generating…';
  setMeta(meta, '');

  const model = $('#ttsModel').value;
  const stopLoading = (await isWarm(model))
    ? null
    : metaLoading(meta, `loading ${model} into memory (first run)`);
  const started = performance.now();

  try {
    const response = await postJSON('/v1/audio/speech', {
      model,
      input: $('#ttsText').value,
      voice: $('#voice').value,
      speed: +$('#speed').value,
      response_format: 'wav',
    });
    if (stopLoading) stopLoading();

    speakState.wavBlob = await response.blob();
    const player = $('#player');
    player.src = URL.createObjectURL(speakState.wavBlob);
    player.style.display = 'block';
    player.play();
    $('#dlBtn').disabled = false;
    setMeta(meta, `generated in ${((performance.now() - started) / 1000).toFixed(1)}s — on this machine`);
  } catch (error) {
    if (stopLoading) stopLoading();
    setErr(meta, error.message);
  }

  button.disabled = false;
  button.textContent = 'Generate & play';
});

$('#dlBtn').addEventListener('click', () => {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(speakState.wavBlob);
  link.download = 'ses.wav';
  link.click();
});

// ---------------------------------------------------------------- transcribe page

const dropZone = $('#drop');
dropZone.addEventListener('click', () => $('#file').click());
$('#file').addEventListener('change', event => {
  if (event.target.files[0]) transcribeFile(event.target.files[0]);
});

['dragover', 'dragleave', 'drop'].forEach(name => {
  dropZone.addEventListener(name, event => {
    event.preventDefault();
    dropZone.classList.toggle('hover', name === 'dragover');
    if (name === 'drop' && event.dataTransfer.files[0]) transcribeFile(event.dataTransfer.files[0]);
  });
});

async function requestTranscript(file, model) {
  const form = new FormData();
  form.append('file', file, file.name || 'recording.wav');
  form.append('model', model || $('#sttModel').value);
  form.append('response_format', 'json');
  const { text } = await (await api('/v1/audio/transcriptions', { method: 'POST', body: form })).json();
  return (text || '').trim();
}

async function transcribeFile(file) {
  const output = $('#sttOut');
  const meta = $('#sttMeta');
  output.classList.remove('err');

  const model = $('#sttModel').value;
  const warm = await isWarm(model);
  output.textContent = `transcribing ${file.name || 'recording'}…`;
  setMeta(meta, '');
  const stopLoading = warm ? null : metaLoading(meta, `loading ${model} into memory (first run)`);
  const started = performance.now();

  try {
    const text = await requestTranscript(file);
    if (stopLoading) stopLoading();
    output.textContent = text || '(silence)';
    $('#copyBtn').disabled = !text;
    setMeta(meta, `transcribed in ${((performance.now() - started) / 1000).toFixed(1)}s — on this machine`);
  } catch (error) {
    if (stopLoading) stopLoading();
    setErr(output, error.message);
  }
}

$('#copyBtn').addEventListener('click', () => navigator.clipboard.writeText($('#sttOut').textContent));

// ---------------------------------------------------------------- microphone

const RECORDER_BUFFER = 4096;
const WAV_HEADER_BYTES = 44;

/** Wire a button as press-to-record, press-again-to-stop. */
function setupRecorder(button, errorElement, onRecorded) {
  const idleLabel = button.textContent;
  let session = null;

  button.addEventListener('click', async () => {
    if (session) {
      const { context, node, stream, blocks } = session;
      session = null;
      node.disconnect();
      stream.getTracks().forEach(track => track.stop());
      const rate = context.sampleRate;
      await context.close();

      button.classList.remove('rec-on');
      button.textContent = idleLabel;
      onRecorded(new File([encodeWav(blocks, rate)], 'recording.wav', { type: 'audio/wav' }));
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const context = new AudioContext();
      const source = context.createMediaStreamSource(stream);
      const node = context.createScriptProcessor(RECORDER_BUFFER, 1, 1);
      const blocks = [];

      node.onaudioprocess = event => blocks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      source.connect(node);
      node.connect(context.destination);

      session = { context, node, stream, blocks };
      button.classList.add('rec-on');
      button.textContent = '■ stop';
    } catch (error) {
      setErr(errorElement, `mic access denied: ${error.message}`);
    }
  });
}

/** Raw float32 blocks to a 16-bit mono WAV, so the server needs no codec. */
function encodeWav(blocks, sampleRate) {
  const total = blocks.reduce((sum, block) => sum + block.length, 0);
  const samples = new Int16Array(total);

  let offset = 0;
  for (const block of blocks) {
    for (let i = 0; i < block.length; i++) {
      const value = Math.max(-1, Math.min(1, block[i]));
      samples[offset++] = value < 0 ? value * 0x8000 : value * 0x7fff;
    }
  }

  const buffer = new ArrayBuffer(WAV_HEADER_BYTES + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (at, text) => [...text].forEach((char, i) => view.setUint8(at + i, char.charCodeAt(0)));

  writeText(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, 'WAVE');
  writeText(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);            // PCM
  view.setUint16(22, 1, true);            // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);            // block align
  view.setUint16(34, 16, true);           // bits per sample
  writeText(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  new Int16Array(buffer, WAV_HEADER_BYTES).set(samples);

  return new Blob([buffer], { type: 'audio/wav' });
}

setupRecorder($('#recBtn'), $('#sttMeta'), transcribeFile);
setupRecorder($('#chatRecBtn'), $('#chatMeta'), async file => {
  const meta = $('#chatMeta');
  setMeta(meta, 'hearing…');
  try {
    const text = await requestTranscript(file, chatOptions.sttModel);
    setMeta(meta, text ? '' : '(heard nothing — try again)');
    if (text) sendChat(text);
  } catch (error) {
    setErr(meta, error.message);
  }
});

// ---------------------------------------------------------------- models page

const hubState = { task: 'llm', window: 'trending' };

const WINDOW_LABELS = {
  trending: 'trending right now',
  month: 'downloads over the last 30 days',
  liked: 'likes, all-time',
};

function fitBadge(fit, unavailable) {
  if (unavailable) return '<span class="fit unknown">not on this OS</span>';
  const level = (fit && fit.level) || 'unknown';
  const label = (fit && fit.label) || '—';
  return `<span class="fit ${level}">${escapeHtml(label)}</span>`;
}

function columnHeader(first, second, third) {
  const row = document.createElement('div');
  row.className = 'mrow head';
  row.innerHTML =
    `<span>${first}</span><span style="text-align:right">${second}</span>` +
    `<span>runs here?</span><span style="text-align:right">${third}</span>`;
  return row;
}

/** One row shared by the curated library and the Hub browser, so they match. */
function modelRow({
  name,
  link,
  description,
  metadata = [],
  note,
  fit,
  unavailable,
  installed,
  pull,
  rank,
  recommended,
}) {
  const row = document.createElement('div');
  row.className = 'mrow';
  const label = link
    ? `<a href="${escapeHtml(link)}" target="_blank" rel="noopener">${escapeHtml(name)}</a>`
    : escapeHtml(name);

  row.innerHTML =
    '<div class="mdetails"><div class="mname">' +
      (rank ? `<span class="rank">${rank}</span>` : '') +
      `<span class="txt">${label}</span>` +
      (recommended ? `<span class="pick" title="${escapeHtml(recommended)}">recommended</span>` : '') +
      '</div>' +
      (description ? `<div class="mdesc">${escapeHtml(description)}</div>` : '') +
    '</div>' +
    `<div class="mnum">${escapeHtml(note || '—')}</div>` +
    `<div>${fitBadge(fit, unavailable)}</div>` +
    '<div class="mact"></div>';

  if (metadata.length) {
    const fields = document.createElement('div');
    fields.className = 'mfields';
    metadata.forEach(item => {
      if (!item || !item.value) return;
      const field = document.createElement(item.href ? 'a' : 'span');
      field.className = `mfield${item.wide ? ' wide' : ''}`;
      field.textContent = `${item.label}: ${item.value}`;
      if (item.href) {
        field.href = item.href;
        field.target = '_blank';
        field.rel = 'noopener';
      }
      fields.append(field);
    });
    if (fields.childElementCount) row.querySelector('.mdetails').append(fields);
  }

  const actions = row.querySelector('.mact');
  if (installed) {
    actions.innerHTML = '<span class="installed">✓ installed</span>';
  } else if (unavailable) {
    actions.innerHTML = '<span class="mnum">—</span>';
  } else if (pull) {
    const button = document.createElement('span');
    button.className = 'pullcmd';
    button.textContent = pull.length > 22 ? 'copy install' : pull;
    button.title = pull;
    button.onclick = () => copyCmd(button, pull);
    actions.append(button);
  }
  return row;
}

function speechMetadata(model) {
  const metadata = [];
  const add = (label, value, options = {}) => {
    const displayed = fmtField(value);
    if (displayed) metadata.push({ label, value: displayed, ...options });
  };

  add('engine', model.engine);
  add('format', model.format);
  add('hardware', model.accelerators);
  if (model.required_ram_gb) add('RAM', fmtSize(model.required_ram_gb));
  if (model.required_vram_gb) add('VRAM', fmtSize(model.required_vram_gb));
  if (model.recommended_vram_gb) {
    add('GPU planning', `${fmtSize(model.recommended_vram_gb)} VRAM`);
  }
  if (model.extra) {
    add('runtime', model.runtime_installed ? 'installed' : `install ses[${model.extra}]`);
  }
  if (Array.isArray(model.backends) && model.backends.length > 1) {
    add(
      'backends',
      [...new Set(model.backends.map(backend => backend.engine).filter(Boolean))],
      { wide: true },
    );
  }
  add('languages', model.languages, { wide: true });
  add('license', model.license);

  if (model.github) {
    const href = githubLink(model.github);
    add('GitHub', href ? 'source ↗' : model.github, { href });
  }
  return metadata;
}

/* The library splits by direction, because that's how you pick one: you either
 * want to be heard or to be understood. A single "speech models" list hid the
 * transcribe half behind fifty voices. */
const SPEECH_SECTIONS = [
  {
    kind: 'stt',
    title: '👂 Transcribe',
    note: 'Speech → text. Used by the Transcribe page and by voice input in Chat.',
  },
  {
    kind: 'tts',
    title: '🔊 Speak',
    note: 'Text → speech. Used by the Speak page and by spoken replies in Chat.',
  },
];

/** A library section: the models worth trying first, with the rest one click away. */
function speechSection(title, note, models) {
  const featured = models.filter(model => model.featured);
  const rest = models.filter(model => !model.featured);

  const block = section(title, models.length, note);
  block.append(columnHeader('model', 'size', 'install'));

  const row = model => modelRow({
    name: model.name,
    link: githubLink(model.github),
    description: model.recommended || model.description,
    recommended: model.recommended,
    metadata: speechMetadata(model),
    note: fmtSize(model.size_gb),
    fit: model.fit,
    unavailable: model.available === false,
    installed: model.installed,
    pull: pullCommand(model.pull),
  });

  featured.forEach(model => block.append(row(model)));
  if (!rest.length) return block;

  const more = document.createElement('div');
  more.className = 'more';
  const toggle = document.createElement('button');
  toggle.className = 'showall';
  toggle.textContent = `show ${rest.length} more`;
  toggle.onclick = () => {
    const shown = more.classList.toggle('open');
    toggle.textContent = shown ? 'show less' : `show ${rest.length} more`;
    if (shown && !more.dataset.filled) {
      more.dataset.filled = '1';
      rest.forEach(model => more.append(row(model)));
    }
  };
  block.append(toggle, more);
  return block;
}

function runtimeStatus(status) {
  if (status && typeof status === 'object') {
    return {
      label: fmtField(status),
      level: fmtField(status.level).toLowerCase(),
    };
  }
  const label = fmtField(status) || 'unknown';
  const normalized = label.toLowerCase();
  let level = 'neutral';
  if (/stable|ready|supported|active|built-in/.test(normalized)) level = 'good';
  if (/preview|experimental|planned|partial|optional/.test(normalized)) level = 'warn';
  if (/unsupported|blocked|unavailable|deprecated/.test(normalized)) level = 'bad';
  return { label, level };
}

function popularRuntimes(runtimes) {
  const checked = fmtField((runtimes.find(runtime => runtime.stars_checked) || {}).stars_checked);
  const block = section(
    '⚙️ Popular speech projects',
    runtimes.length,
    'Widely used model families and engines, with Windows support. ' +
      (checked ? `GitHub stars are an approximate ${checked} snapshot, not a quality score.` : ''),
  );
  const list = document.createElement('div');
  list.className = 'runtime-list';
  list.innerHTML =
    '<div class="runtime-row head"><span>project / runtime</span><span>stars</span>' +
    '<span>Windows</span><span>accelerator</span><span>status</span></div>';

  runtimes.forEach(runtime => {
    const row = document.createElement('div');
    row.className = 'runtime-row';
    const link = safeLink(runtime.link) || githubLink(runtime.github);
    const name = fmtField(runtime.name || runtime.engine || runtime.id) || 'runtime';
    const status = runtimeStatus(runtime.status);
    const windows = typeof runtime.windows === 'boolean'
      ? (runtime.windows ? '✓ supported' : 'not supported')
      : (fmtField(runtime.windows) || 'unknown');

    const main = document.createElement('div');
    main.className = 'runtime-main';
    const title = document.createElement(link ? 'a' : 'span');
    title.textContent = name;
    if (link) {
      title.href = link;
      title.target = '_blank';
      title.rel = 'noopener';
    }
    main.append(title);
    if (runtime.description) {
      const description = document.createElement('span');
      description.className = 'runtime-desc';
      description.textContent = fmtField(runtime.description);
      main.append(description);
    }

    const cell = (label, value, className = '') => {
      const element = document.createElement('div');
      element.className = `runtime-cell ${className}`.trim();
      element.dataset.label = label;
      element.textContent = value;
      return element;
    };

    row.append(
      main,
      cell('stars', runtime.stars == null ? '—' : `${fmtCount(runtime.stars)} ★`),
      cell('Windows', windows, runtime.windows === true ? 'supported' : ''),
      cell('accelerator', fmtField(runtime.accelerator) || '—'),
    );

    const statusCell = cell('status', '', `runtime-status ${status.level}`);
    const statusLabel = document.createElement(link ? 'a' : 'span');
    statusLabel.textContent = `${status.label}${link ? ' ↗' : ''}`;
    if (link) {
      statusLabel.href = link;
      statusLabel.target = '_blank';
      statusLabel.rel = 'noopener';
    }
    statusCell.append(statusLabel);
    row.append(statusCell);
    list.append(row);
  });

  block.append(list);
  return block;
}

function section(title, count, note) {
  const block = document.createElement('div');
  block.className = 'catsec';
  block.innerHTML =
    `<div class="sechead"><h2>${title}</h2>` +
    (count != null ? `<span class="count">${count}</span>` : '') +
    '</div>' +
    (note ? `<div class="note">${note}</div>` : '');
  return block;
}

async function loadCatalog() {
  const container = $('#catalog');
  try {
    const catalog = await getJSON('/api/catalog');
    renderMachine(catalog.system || {});
    container.innerHTML = '';

    const brains = section(
      '🧠 Voice brains',
      catalog.curated.brains.length,
      'The LLM behind the Assistant. These run in Ollama — installing one calls <code>ollama pull</code>.',
    );
    brains.append(columnHeader('model', 'size', 'install'));
    catalog.curated.brains.forEach(model => brains.append(modelRow({
      name: model.name,
      description: model.description,
      note: fmtSize(model.size_gb),
      fit: model.fit,
      installed: model.installed,
      pull: pullCommand(model.pull),
    })));
    container.append(brains);

    SPEECH_SECTIONS.forEach(({ kind, title, note }) => {
      const models = catalog.curated.speech.filter(model => model.kind === kind);
      container.append(speechSection(title, note, models));
    });

    if (Array.isArray(catalog.ecosystem) && catalog.ecosystem.length) {
      container.append(popularRuntimes(catalog.ecosystem));
    }

    container.append(buildHubBrowser());
    loadHubModels();
  } catch (error) {
    container.innerHTML = `<div class="hint err">couldn't load catalog: ${escapeHtml(error.message)}</div>`;
  }
}

function renderMachine(system) {
  const memory = system.ram_gb ? `${system.ram_gb} GB` : 'unknown';
  const cpu = system.cpu_count ? `${system.cpu_count} logical CPUs` : 'unknown';
  const gpu = system.gpu_name || (system.apple_silicon ? 'Apple Silicon GPU' : 'not detected');
  const vram = system.vram_gb
    ? `${system.vram_gb} GB`
    : (system.apple_silicon ? 'shared memory' : 'unknown');
  const comfortable = system.ram_gb ? Math.round(system.ram_gb * 0.65) : null;
  $('#machine').innerHTML =
    `<div class="spec"><span class="k">system</span><span class="v">${escapeHtml(system.os || '?')}</span></div>` +
    `<div class="spec"><span class="k">memory</span><span class="v">${memory}</span></div>` +
    `<div class="spec"><span class="k">CPU</span><span class="v">${escapeHtml(cpu)}</span></div>` +
    `<div class="spec"><span class="k">GPU</span><span class="v">${escapeHtml(gpu)}</span></div>` +
    `<div class="spec"><span class="k">VRAM</span><span class="v">${escapeHtml(vram)}</span></div>` +
    `<div class="spec"><span class="k">accelerator</span><span class="v">${escapeHtml(system.accelerator || '?')}</span></div>` +
    (comfortable
      ? `<div class="hint">Models up to about ${comfortable} GB run comfortably here — the badges below say which.</div>`
      : '');
}

function buildHubBrowser() {
  const block = section(
    '🔥 Browse Hugging Face',
    null,
    'Live popularity across the whole Hub, with a size estimate for a 4-bit local copy. ' +
    'Install is shown only when the API provides a verified command.',
  );
  block.id = 'hubbrowse';
  block.insertAdjacentHTML('beforeend', `
    <div class="pillgroup">
      <span class="lbl">type</span>
      <div class="pills" id="hubTask">
        <button class="pill active" data-task="llm">LLMs</button>
        <button class="pill" data-task="tts">Text-to-speech</button>
        <button class="pill" data-task="stt">Speech-to-text</button>
      </div>
      <span class="lbl">ranked by</span>
      <div class="pills" id="hubWindow">
        <button class="pill active" data-window="trending">Trending</button>
        <button class="pill" data-window="month">This month</button>
        <button class="pill" data-window="liked">Most liked</button>
      </div>
    </div>
    <div id="hubResults"><div class="hint">loading…</div></div>`);

  const wire = (groupId, key) => {
    block.querySelectorAll(`#${groupId} .pill`).forEach(pill => {
      pill.addEventListener('click', () => {
        block.querySelectorAll(`#${groupId} .pill`).forEach(other => other.classList.remove('active'));
        pill.classList.add('active');
        hubState[key] = pill.dataset[key];
        loadHubModels();
      });
    });
  };
  wire('hubTask', 'task');
  wire('hubWindow', 'window');
  return block;
}

async function loadHubModels() {
  const container = $('#hubResults');
  container.innerHTML = '<div class="hint">loading from Hugging Face…</div>';
  try {
    const query = `task=${hubState.task}&window=${hubState.window}&limit=${HUB_PAGE_SIZE}`;
    const data = await getJSON(`/api/hf?${query}`);
    if (!data.models.length) {
      container.innerHTML = '<div class="hint">nothing found — are you offline?</div>';
      return;
    }

    container.innerHTML = '';
    const caption = document.createElement('div');
    caption.className = 'note';
    caption.textContent =
      `${data.models.length} models · ${WINDOW_LABELS[hubState.window] || hubState.window}`;
    container.append(caption);
    container.append(columnHeader('model', 'popularity · size', 'install'));

    data.models.forEach((model, index) => {
      // This is a popularity leaderboard, so the count belongs on every row —
      // it used to appear on the LLM tab only.
      const metric = data.metric === 'likes'
        ? `${fmtCount(model.likes)} ♥`
        : `${fmtCount(model.downloads)} ↓`;
      const size = fmtSize(model.size_gb);

      const row = modelRow({
        name: model.id,
        link: `https://huggingface.co/${model.id}`,
        description: hubState.task === 'llm'
          ? null
          : ((model.compatibility || {}).reason || null),
        metadata: hubState.task === 'llm'
          ? []
          : [
              { label: 'engine', value: model.engine },
              { label: 'size', value: model.size_basis },
            ],
        note: size === '—' ? metric : `${metric} · ${size}`,
        fit: model.fit,
        rank: index + 1,
        pull: pullCommand(model.pull),
      });

      if (model.curated) {
        row.querySelector('.mname').insertAdjacentHTML(
          'beforeend',
          `<span class="curated">in library</span>`,
        );
      }
      container.append(row);
    });
  } catch (error) {
    container.innerHTML = `<div class="hint err">${escapeHtml(error.message)}</div>`;
  }
}

// ---------------------------------------------------------------- startup

init();
initBrain();
