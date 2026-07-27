document.addEventListener("DOMContentLoaded", () => {
    // --- Global State ---
    let modelInfo = null;
    let attentionData = null;
    let generationSteps = [];

    // --- Tab Navigation ---
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const target = btn.getAttribute("data-tab");
            document.getElementById(target).classList.add("active");

            if (target === "tab-attention" && !attentionData) {
                fetchAttention();
            } else if (target === "tab-tokenizer") {
                fetchTokenize();
            }
        });
    });

    // --- Range Slider Value Binding ---
    function bindSlider(sliderId, displayId) {
        const slider = document.getElementById(sliderId);
        const display = document.getElementById(displayId);
        if (slider && display) {
            slider.addEventListener("input", (e) => {
                display.textContent = e.target.value;
            });
        }
    }

    bindSlider("temp-slider", "temp-val");
    bindSlider("topk-slider", "topk-val");
    bindSlider("tokens-slider", "tokens-val");

    // --- 1. Fetch Model Info ---
    async function fetchModelInfo() {
        try {
            const res = await fetch("/api/info");
            modelInfo = await res.json();

            document.getElementById("badge-params").textContent = `${(modelInfo.num_params / 1e6).toFixed(2)}M Params`;
            document.getElementById("badge-layers").textContent = `Layers: ${modelInfo.n_layer}`;
            document.getElementById("badge-heads").textContent = `Heads: ${modelInfo.n_head}`;
            document.getElementById("badge-norm").textContent = modelInfo.use_rms_norm ? "RMSNorm" : "LayerNorm";
            document.getElementById("vocab-count").textContent = modelInfo.vocab_size;

            // Populate Attention Layer and Head Selectors
            const layerSel = document.getElementById("layer-select");
            const headSel = document.getElementById("head-select");
            layerSel.innerHTML = "";
            headSel.innerHTML = "";

            for (let i = 0; i < modelInfo.n_layer; i++) {
                layerSel.innerHTML += `<option value="${i}">Layer ${i + 1}</option>`;
            }
            for (let j = 0; j < modelInfo.n_head; j++) {
                headSel.innerHTML += `<option value="${j}">Head ${j + 1}</option>`;
            }
        } catch (err) {
            console.error("Failed to load model info:", err);
            document.getElementById("badge-params").textContent = "Offline / Demo";
        }
    }

    // --- 2. Generation Playground ---
    const generateBtn = document.getElementById("generate-btn");
    const promptInput = document.getElementById("prompt-input");
    const genText = document.getElementById("gen-text");
    const candidateBars = document.getElementById("candidate-bars");
    const stepSelector = document.getElementById("step-selector");

    generateBtn.addEventListener("click", async () => {
        const prompt = promptInput.value;
        const temperature = parseFloat(document.getElementById("temp-slider").value);
        const top_k = parseInt(document.getElementById("topk-slider").value);
        const max_tokens = parseInt(document.getElementById("tokens-slider").value);

        generateBtn.disabled = true;
        generateBtn.textContent = "Generating...";

        try {
            const res = await fetch("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt, temperature, top_k, max_tokens })
            });
            const data = await res.json();

            document.querySelector(".prompt-text").textContent = prompt;
            genText.textContent = data.new_tokens_text;
            generationSteps = data.steps;

            renderStepSelector(generationSteps);
            if (generationSteps.length > 0) {
                renderCandidateBars(generationSteps[0]);
            }
        } catch (err) {
            console.error("Generation error:", err);
            genText.textContent = " [Generation Failed]";
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = "Generate Tokens 🚀";
        }
    });

    function renderStepSelector(steps) {
        stepSelector.innerHTML = "";
        steps.forEach((step, idx) => {
            const chip = document.createElement("button");
            chip.className = `step-chip ${idx === 0 ? "active" : ""}`;
            chip.textContent = `Step ${step.step}: '${formatToken(step.sampled_token)}'`;
            chip.addEventListener("click", () => {
                document.querySelectorAll(".step-chip").forEach(c => c.classList.remove("active"));
                chip.classList.add("active");
                renderCandidateBars(step);
            });
            stepSelector.appendChild(chip);
        });
    }

    function renderCandidateBars(step) {
        candidateBars.innerHTML = "";
        step.candidates.forEach(cand => {
            const row = document.createElement("div");
            row.className = "prob-row";
            const percent = (cand.probability * 100).toFixed(1);
            row.innerHTML = `
                <span class="prob-token">'${formatToken(cand.token)}'</span>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width: ${percent}%"></div>
                </div>
                <span>${percent}%</span>
            `;
            candidateBars.appendChild(row);
        });
    }

    // --- 3. Attention Heatmap ---
    const updateAttnBtn = document.getElementById("update-attn-btn");
    const attnPromptInput = document.getElementById("attn-prompt");
    const layerSelect = document.getElementById("layer-select");
    const headSelect = document.getElementById("head-select");
    const heatmapGrid = document.getElementById("heatmap-grid");

    updateAttnBtn.addEventListener("click", fetchAttention);
    layerSelect.addEventListener("change", renderHeatmap);
    headSelect.addEventListener("change", renderHeatmap);

    async function fetchAttention() {
        const prompt = attnPromptInput.value || "ROMEO:";
        try {
            const res = await fetch("/api/attention", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt })
            });
            attentionData = await res.json();
            renderHeatmap();
        } catch (err) {
            console.error("Attention fetch error:", err);
        }
    }

    function renderHeatmap() {
        if (!attentionData) return;

        const layerIdx = parseInt(layerSelect.value) || 0;
        const headIdx = parseInt(headSelect.value) || 0;
        const tokens = attentionData.tokens;
        const matrix = attentionData.attentions[layerIdx][headIdx]; // shape (T, T)
        const T = tokens.length;

        heatmapGrid.style.gridTemplateColumns = `40px repeat(${T}, 32px)`;
        heatmapGrid.innerHTML = "";

        // Top Header Row (Keys)
        heatmapGrid.appendChild(createLabelCell(""));
        tokens.forEach(tok => heatmapGrid.appendChild(createLabelCell(formatToken(tok))));

        // Rows (Queries)
        for (let q = 0; q < T; q++) {
            heatmapGrid.appendChild(createLabelCell(formatToken(tokens[q])));
            for (let k = 0; k < T; k++) {
                const val = matrix[q][k];
                heatmapGrid.appendChild(createHeatmapCell(val, tokens[q], tokens[k]));
            }
        }
    }

    function createLabelCell(text) {
        const cell = document.createElement("div");
        cell.className = "heatmap-cell";
        cell.style.background = "transparent";
        cell.style.color = "var(--text-secondary)";
        cell.style.fontSize = "0.7rem";
        cell.textContent = text;
        return cell;
    }

    function createHeatmapCell(val, qTok, kTok) {
        const cell = document.createElement("div");
        cell.className = "heatmap-cell";
        cell.title = `Query: '${formatToken(qTok)}' → Key: '${formatToken(kTok)}'\nWeight: ${val.toFixed(4)}`;
        
        // Interpolate color from dark violet to bright cyan/pink
        const r = Math.round(15 + val * (236 - 15));
        const g = Math.round(23 + val * (72 - 23));
        const b = Math.round(42 + val * (153 - 42));
        cell.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${0.15 + val * 0.85})`;
        cell.style.color = val > 0.4 ? "#ffffff" : "transparent";
        cell.textContent = val > 0.2 ? val.toFixed(2).replace("0.", ".") : "";
        return cell;
    }

    // --- 4. Tokenizer Inspector ---
    const tokInput = document.getElementById("tok-input");
    const tokensChips = document.getElementById("tokens-chips");
    const tokCount = document.getElementById("tok-count");

    tokInput.addEventListener("input", fetchTokenize);

    async function fetchTokenize() {
        const text = tokInput.value;
        try {
            const res = await fetch("/api/tokenize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });
            const data = await res.json();

            tokensChips.innerHTML = "";
            data.tokens.forEach((char, i) => {
                const id = data.token_ids[i];
                const chip = document.createElement("div");
                chip.className = "token-chip";
                chip.innerHTML = `
                    <span class="chip-char">'${formatToken(char)}'</span>
                    <span class="chip-id">ID: ${id}</span>
                `;
                tokensChips.appendChild(chip);
            });

            tokCount.textContent = data.length;
        } catch (err) {
            console.error("Tokenizer error:", err);
        }
    }

    function formatToken(tok) {
        if (tok === "\n") return "\\n";
        if (tok === " ") return "␣";
        if (tok === "\t") return "\\t";
        return tok;
    }

    // Initialize
    fetchModelInfo();
});
