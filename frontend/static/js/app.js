// Настройка Markdown
marked.setOptions({ breaks: true, gfm: true });

const els = {};
function initElements() {
  els.inputForm = document.getElementById('input-form');
  els.userInput = document.getElementById('user-input');
  els.output = document.getElementById('output');
  els.sourcesList = document.getElementById('sources-list');
  els.alertBox = document.getElementById('alert');
  els.submitBtn = document.getElementById('submit-btn');
  els.currentMode = document.getElementById('current-mode');
}

function syncUI(mode) {
  const isWrite = mode === 'write';
  document.body.dataset.mode = mode;
  els.currentMode.textContent = isWrite ? 'Режим: Запись' : 'Режим: Чтение';
  els.userInput.placeholder = isWrite ? 'Введи текст или путь к файлу...' : 'Спроси что-нибудь...';
  els.submitBtn.textContent = isWrite ? 'Сохранить' : 'Отправить';
}

async function handleSubmit(e) {
  e.preventDefault();
  const text = els.userInput.value.trim();
  if (!text) return;

  const mode = document.body.dataset.mode || 'read';
  els.userInput.disabled = true;
  els.submitBtn.disabled = true;
  els.submitBtn.textContent = '⏳';
  els.output.innerHTML = mode === 'read' ? '<p class="placeholder">🔍 Ищу и думаю...</p>' : '<p class="placeholder">⏳ Анализирую...</p>';
  els.sourcesList.innerHTML = '';

  try {
    if (mode === 'read') await handleQueryStream(text);
    else await handleIngestRequest(text);
  } catch (err) {
    console.error('❌ Request failed:', err);
    showAlert(`Ошибка: ${err.message}`, 'error');
    els.output.innerHTML = '<p style="color: var(--text-muted);">❌ Сбой соединения</p>';
  } finally {
    els.userInput.disabled = false;
    els.submitBtn.disabled = false;
    syncUI(mode);
    els.userInput.value = '';
    els.userInput.focus();
  }
}

async function handleQueryStream(question) {
  const response = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let answerBuffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop();

    for (const eventRaw of events) {
      if (!eventRaw.trim()) continue;
      const lines = eventRaw.split('\n');
      let type = 'message', data = '';

      for (const line of lines) {
        if (line.startsWith('event:')) type = line.slice(6).trim();
        if (line.startsWith('data:')) data = line.slice(5).trim(); // ✅ Робастный парсинг
      }
      if (!data) continue;

      if (type === 'sources') renderSources(JSON.parse(data));
      else if (type === 'answer') {
        answerBuffer += data;
        els.output.innerHTML = marked.parse(answerBuffer);
        els.output.scrollTop = els.output.scrollHeight;
      }
      else if (type === 'done') {
        els.submitBtn.textContent = 'Готово ✓';
        setTimeout(() => syncUI('read'), 1500);
      }
      else if (type === 'error') {
        showAlert(JSON.parse(data).message, 'error');
      }
    }
  }
}

function renderSources(sources) {
  if (!sources?.length) {
    els.sourcesList.innerHTML = '<p class="placeholder">Ничего не найдено</p>';
    return;
  }
  els.sourcesList.innerHTML = sources.map(src => `
    <div class="source-card">
      <div class="source-topic">${src.topic || 'info'}</div>
      <div class="source-preview">${src.preview}</div>
    </div>
  `).join('');
}

async function handleIngestRequest(text) {
  const res = await fetch('/api/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  const data = await res.json();
  showAlert(data.message, data.status === 'success' ? 'success' : data.status === 'duplicate' ? 'duplicate' : 'error');

  els.output.innerHTML = `<p>${data.message}</p>
    ${data.data?.topic ? `<p class="placeholder" style="margin-top:8px">Тема: ${data.data.topic}</p>` : ''}
    ${data.data?.related_to ? `<p class="placeholder">Связано: ${data.data.related_to}</p>` : ''}`;
}

function showAlert(msg, type) {
  els.alertBox.className = `alert ${type}`;
  els.alertBox.textContent = msg;
  els.alertBox.classList.add('show');
  setTimeout(() => els.alertBox.classList.remove('show'), 4000);
}

document.addEventListener('DOMContentLoaded', () => {
  initElements();
  syncUI('read'); // Начальное состояние

  // ✅ Привязка к радио-кнопкам
  document.querySelectorAll('input[name="mode"]').forEach(radio => {
    radio.addEventListener('change', e => syncUI(e.target.value));
  });

  els.inputForm.addEventListener('submit', handleSubmit);
  els.userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); els.inputForm.requestSubmit(); }
  });
  els.userInput.focus();
});