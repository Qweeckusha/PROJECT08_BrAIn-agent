/**
 * BrAIn Frontend Logic (Streaming Edition)
 */

const els = {};

function initElements() {
  els.modeToggle = document.getElementById('mode-toggle');
  els.modeLabel = document.getElementById('mode-label');
  els.inputForm = document.getElementById('input-form');
  els.userInput = document.getElementById('user-input');
  els.output = document.getElementById('output');
  els.sourcesList = document.getElementById('sources-list');
  els.alertBox = document.getElementById('alert');
  els.btnText = document.getElementById('btn-text');
  els.hint = document.getElementById('hint');
}

function toggleMode(isWrite) {
  document.body.dataset.mode = isWrite ? 'write' : 'read';

  // ✅ Обновляем ОБА лейбла
  const modeText = isWrite ? 'Запись' : 'Чтение';

  // 1. Лейбл в хедере
  els.modeLabel.innerHTML = `Режим: <strong>${modeText}</strong>`;

  // 2. ✅ Лейбл в сайдбаре (этого не хватало!)
  const sidebarLabel = document.getElementById('mode-label-sidebar');
  if (sidebarLabel) {
    sidebarLabel.textContent = modeText;
  }

  // Остальная логика...
  els.userInput.placeholder = isWrite ? 'Введи текст для сохранения в базу...' : 'Спроси что-нибудь...';
  els.btnText.textContent = isWrite ? 'Сохранить' : 'Отправить';
  els.hint.textContent = isWrite ? 'Нажми для добавления в базу' : 'Нажми Enter для отправки';
  els.output.innerHTML = '<p class="text-gray-500">Ожидание ввода...</p>';
  els.sourcesList.innerHTML = '<p class="text-sm text-gray-400 italic">Источники появятся здесь...</p>';
}
// === ГЛАВНЫЙ ОБРАБОТЧИК ОТПРАВКИ ===
async function handleSubmit(e) {
  e.preventDefault();
  const text = els.userInput.value.trim();
  if (!text) return;

  const mode = document.body.dataset.mode;

  // UI: загрузка
  els.userInput.disabled = true;
  els.btnText.textContent = '⏳';
  els.output.innerHTML = mode === 'read' ? '<p class="text-gray-500">🔍 Ищу и думаю...</p>' : '<p class="text-gray-500">⏳ Сохраняю...</p>';
  els.sourcesList.innerHTML = '';

  try {
    if (mode === 'read') {
      await handleQueryStream(text);
    } else {
      await handleIngestRequest(text);
    }
  } catch (err) {
    showError(`Ошибка: ${err.message}`);
    els.output.innerHTML = '<p class="text-red-500">❌ Сбой соединения</p>';
  } finally {
    els.userInput.disabled = false;
    els.btnText.textContent = mode === 'read' ? 'Отправить' : 'Сохранить';
    els.userInput.value = '';
    els.userInput.focus();
  }
}

// === 🌊 СТРИМИНГ ЗАПРОСА (SSE Parser) ===
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

  // Очищаем поле ответа перед стримом
  els.output.innerHTML = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop(); // Сохраняем неполное событие

    for (const eventRaw of events) {
      const lines = eventRaw.split('\n');
      let type = 'message';
      let data = '';

      for (const line of lines) {
        if (line.startsWith('event: ')) type = line.slice(7);
        if (line.startsWith('data: ')) data = line.slice(6);
      }

      if (!data) continue;

      if (type === 'sources') {
        renderSources(JSON.parse(data));
      }
      else if (type === 'answer') {
        answerBuffer += data;
        // Заменяем переносы строк на <br> для HTML
        els.output.innerHTML = answerBuffer.replace(/\n/g, '<br>');
        // Автоскролл вниз
        els.output.scrollTop = els.output.scrollHeight;
      }
      else if (type === 'done') {
        els.btnText.textContent = 'Готово ✓';
        setTimeout(() => els.btnText.textContent = 'Отправить', 2000);
      }
      else if (type === 'error') {
        showError(JSON.parse(data).message);
        els.output.innerHTML += '<p class="text-red-500 mt-2">⚠️ Ошибка генерации</p>';
      }
    }
  }
}

// === 📚 РЕНДЕР ИСТОЧНИКОВ ===
function renderSources(sources) {
  if (!sources.length) {
    els.sourcesList.innerHTML = '<p class="text-sm text-gray-400 italic">Ничего не найдено</p>';
    return;
  }
  els.sourcesList.innerHTML = sources.map(src => `
    <div class="p-3 bg-gray-50 rounded border-l-4 border-blue-500 animate-pulse-once">
      <div class="text-xs font-medium text-blue-700 mb-1">${src.topic || 'info'}</div>
      <div class="text-sm text-gray-700">${src.preview}</div>
    </div>
  `).join('');
}

// === 💾 ЗАПРОС НА ЗАПИСЬ (обычный JSON) ===
async function handleIngestRequest(text) {
  const res = await fetch('/api/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  const data = await res.json();

  const colors = { success: 'bg-green-100 text-green-800 border-green-300', duplicate: 'bg-yellow-100 text-yellow-800 border-yellow-300', error: 'bg-red-100 text-red-800 border-red-300' };
  els.alertBox.className = `fixed top-4 right-4 px-4 py-3 rounded-lg shadow-lg text-sm font-medium alert-enter z-50 border ${colors[data.status] || colors.error}`;
  els.alertBox.textContent = data.message;
  els.alertBox.classList.remove('hidden');
  setTimeout(() => els.alertBox.classList.add('hidden'), 4000);

  els.output.innerHTML = `
    <div class="p-4 bg-white rounded-lg border border-gray-200">
      <p class="font-medium">${data.message}</p>
      ${data.data?.topic ? `<p class="text-sm text-gray-500 mt-1">Тема: ${data.data.topic}</p>` : ''}
      ${data.data?.related_to ? `<p class="text-sm text-gray-500 mt-1">Связано с: ${data.data.related_to}</p>` : ''}
    </div>`;
}

function showError(msg) {
  els.alertBox.className = 'fixed top-4 right-4 px-4 py-3 rounded-lg shadow-lg text-sm font-medium alert-enter z-50 border bg-red-100 text-red-800 border-red-300';
  els.alertBox.textContent = '❌ ' + msg;
  els.alertBox.classList.remove('hidden');
  setTimeout(() => els.alertBox.classList.add('hidden'), 5000);
}

// === INIT ===
document.addEventListener('DOMContentLoaded', () => {
  initElements();
  els.modeToggle.addEventListener('change', e => toggleMode(e.target.checked));
  els.inputForm.addEventListener('submit', handleSubmit);
  els.userInput.addEventListener('keypress', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); els.inputForm.requestSubmit(); }});
  els.userInput.focus();
});