const modeButtons = document.querySelectorAll(".mode-btn");
const textMode = document.getElementById("text-mode");
const urlMode = document.getElementById("url-mode");
const sourceText = document.getElementById("source-text");
const sourceUrl = document.getElementById("source-url");
const loadBtn = document.getElementById("load-btn");
const sourceStatus = document.getElementById("source-status");
const chatLog = document.getElementById("chat-log");
const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question-input");

let currentMode = "text";
let hasSource = false;

modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    currentMode = btn.dataset.mode;
    modeButtons.forEach((b) => b.classList.toggle("is-active", b === btn));
    textMode.classList.toggle("is-hidden", currentMode !== "text");
    urlMode.classList.toggle("is-hidden", currentMode !== "url");
  });
});

function setStatus(message, kind = "") {
  sourceStatus.textContent = message;
  sourceStatus.className = `status${kind ? ` is-${kind}` : ""}`;
}

function clearEmptyState() {
  const empty = chatLog.querySelector(".empty-state");
  if (empty) empty.remove();
}

function appendMessage(role, bodyHtml) {
  clearEmptyState();
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  wrap.innerHTML = `
    <span class="role">${role === "user" ? "You" : "AI"}</span>
    <div class="body">${bodyHtml}</div>
  `;
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function readError(response) {
  try {
    const data = await response.json();
    return data.detail || "Something went wrong.";
  } catch {
    return "Something went wrong.";
  }
}

loadBtn.addEventListener("click", async () => {
  loadBtn.disabled = true;
  setStatus("Loading and indexing…");

  try {
    let response;
    if (currentMode === "text") {
      response = await fetch("/api/ingest/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sourceText.value }),
      });
    } else {
      response = await fetch("/api/ingest/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: sourceUrl.value }),
      });
    }

    if (!response.ok) throw new Error(await readError(response));

    const data = await response.json();
    hasSource = true;
    chatLog.innerHTML = "";
    appendMessage(
      "assistant",
      escapeHtml(
        `Ready. Indexed ${data.chunk_count} chunk(s). Ask me anything about this source.`
      )
    );
    setStatus(`Loaded ${data.chunk_count} chunk(s) into Pinecone.`, "ok");
    questionInput.focus();
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    loadBtn.disabled = false;
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  if (!hasSource) {
    setStatus("Load a paragraph or link before asking.", "error");
    return;
  }

  const askBtn = askForm.querySelector("button");
  askBtn.disabled = true;
  questionInput.value = "";
  appendMessage("user", escapeHtml(question));
  appendMessage("assistant", "Thinking…");

  const thinking = chatLog.lastElementChild;

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) throw new Error(await readError(response));

    const data = await response.json();
    const chunksHtml = (data.chunks || [])
      .map(
        (chunk, i) =>
          `<p class="chunk-item"><strong>Chunk ${i + 1}</strong><br>${escapeHtml(
            chunk
          )}</p>`
      )
      .join("");

    thinking.querySelector(".body").innerHTML = `
      <p class="body">${escapeHtml(data.answer)}</p>
      <details class="chunks">
        <summary>Show retrieved chunks</summary>
        ${chunksHtml}
      </details>
    `;
  } catch (err) {
    thinking.querySelector(".body").textContent = err.message;
    setStatus(err.message, "error");
  } finally {
    askBtn.disabled = false;
    questionInput.focus();
    chatLog.scrollTop = chatLog.scrollHeight;
  }
});
