const labelEl = document.getElementById('label');
const confEl = document.getElementById('conf');
const top2El = document.getElementById('top2');
const marginEl = document.getElementById('margin');
const bufferEl = document.getElementById('buffer-text');
const statusText = document.getElementById('status-text');

async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const s = await res.json();

    labelEl.textContent = s.label;
    confEl.textContent = s.conf.toFixed(2);
    top2El.textContent = `${s.top2[0]} ${s.top2[1].toFixed(2)}`;
    marginEl.textContent = s.margin.toFixed(2);
    bufferEl.textContent = s.text;

    if (!s.hand) {
      statusText.textContent = 'No hand detected';
    } else if (s.conf >= 0.6) {
      statusText.textContent = `Prediction: ${s.label} (${s.conf.toFixed(2)})`;
    } else {
      statusText.textContent = `Low confidence → ${s.label} (${s.conf.toFixed(2)})`;
    }
  } catch (e) {
    statusText.textContent = 'Camera/Network issue…';
  }
}

setInterval(fetchStatus, 150);

async function postJSON(url, body = {}) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  return res.json();
}

document.getElementById('confirm').onclick = async () => {
  const data = await postJSON('/api/confirm');
  bufferEl.textContent = data.text;
};
document.getElementById('backspace').onclick = async () => {
  const data = await postJSON('/api/backspace');
  bufferEl.textContent = data.text;
};
document.getElementById('clear').onclick = async () => {
  const data = await postJSON('/api/clear');
  bufferEl.textContent = data.text;
};
document.getElementById('speak').onclick = async () => {
  await postJSON('/api/speak', { text: bufferEl.textContent || '' });
};

document.addEventListener('keydown', async (e) => {
  if (e.key === 'Enter') {
    const data = await postJSON('/api/confirm');
    bufferEl.textContent = data.text;
  } else if (e.key === ' ') {
    e.preventDefault();
    await postJSON('/api/speak', { text: bufferEl.textContent || '' });
  } else if (e.key.toLowerCase() === 'b') {
    const data = await postJSON('/api/backspace');
    bufferEl.textContent = data.text;
  } else if (e.key.toLowerCase() === 'c') {
    const data = await postJSON('/api/clear');
    bufferEl.textContent = data.text;
  }
});
