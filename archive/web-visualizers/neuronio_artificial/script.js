// ===== ESTADO GLOBAL =====
let activationFn = 'step';

// ===== FUNÇÕES DE ATIVAÇÃO =====
const activations = {
    step: (z) => z >= 0 ? 1 : 0,
    sigmoid: (z) => 1 / (1 + Math.exp(-z)),
    relu: (z) => Math.max(0, z),
    tanh: (z) => Math.tanh(z)
};

const activationNames = {
    step: 'Step (Tudo ou Nada)',
    sigmoid: 'Sigmoid (σ)',
    relu: 'ReLU',
    tanh: 'Tanh'
};

// ===== LEITURA DE VALORES =====
function getValues() {
    const x1 = parseFloat(document.getElementById('x1').value);
    const x2 = parseFloat(document.getElementById('x2').value);
    const x3 = parseFloat(document.getElementById('x3').value);
    const w1 = parseFloat(document.getElementById('w1').value);
    const w2 = parseFloat(document.getElementById('w2').value);
    const w3 = parseFloat(document.getElementById('w3').value);
    const bias = parseFloat(document.getElementById('bias').value);
    return { x1, x2, x3, w1, w2, w3, bias };
}

// ===== CÁLCULO DO NEURÔNIO =====
function compute() {
    const v = getValues();
    const z = v.x1 * v.w1 + v.x2 * v.w2 + v.x3 * v.w3 + v.bias;
    const output = activations[activationFn](z);
    return { ...v, z, output };
}

// ===== ATUALIZAR DISPLAYS =====
function updateDisplays() {
    const r = compute();

    // Displays de valor das entradas
    document.getElementById('x1-display').textContent = r.x1.toFixed(2);
    document.getElementById('x2-display').textContent = r.x2.toFixed(2);
    document.getElementById('x3-display').textContent = r.x3.toFixed(2);

    // Tipos de peso (excitatório/inibitório)
    [['w1', 'w1-type', 'input-group-1'], ['w2', 'w2-type', 'input-group-2'], ['w3', 'w3-type', 'input-group-3']].forEach(([wId, typeId, groupId]) => {
        const w = parseFloat(document.getElementById(wId).value);
        const typeEl = document.getElementById(typeId);
        const groupEl = document.getElementById(groupId);
        if (w >= 0) {
            typeEl.textContent = 'excitatório';
            typeEl.className = 'weight-type excitatory';
            groupEl.classList.remove('inhibitory');
        } else {
            typeEl.textContent = 'inibitório';
            typeEl.className = 'weight-type inhibitory';
            groupEl.classList.add('inhibitory');
        }
    });

    // Matemática
    document.getElementById('math-sum').textContent =
        `(${r.x1.toFixed(2)}×${r.w1.toFixed(1)}) + (${r.x2.toFixed(2)}×${r.w2.toFixed(1)}) + (${r.x3.toFixed(2)}×${r.w3.toFixed(1)}) + (${r.bias.toFixed(1)})`;
    document.getElementById('math-z').textContent = r.z.toFixed(4);
    document.getElementById('math-activation').textContent =
        `${activationNames[activationFn]}(${r.z.toFixed(4)}) = ${r.output.toFixed(4)}`;

    // Saída
    const outputEl = document.getElementById('output-value');
    outputEl.textContent = r.output.toFixed(4);

    const bar = document.getElementById('output-bar');
    let barWidth, barColor;

    if (activationFn === 'tanh') {
        barWidth = ((r.output + 1) / 2) * 100;
    } else {
        barWidth = Math.min(Math.abs(r.output) * 100, 100);
    }
    
    const isFiring = (activationFn === 'step' && r.output >= 1) ||
                     (activationFn !== 'step' && r.output > 0.5);

    if (isFiring) {
        barColor = 'linear-gradient(90deg, #fbbf24, #f59e0b)';
        outputEl.style.color = '#fbbf24';
    } else {
        barColor = 'linear-gradient(90deg, #4f8fff, #22d3ee)';
        outputEl.style.color = '#4f8fff';
    }

    bar.style.width = barWidth + '%';
    bar.style.background = barColor;

    const status = document.getElementById('fire-status');
    if (isFiring) {
        status.textContent = '⚡ DISPARANDO — Potencial de Ação!';
        status.className = 'fire-status firing';
    } else {
        status.textContent = '— Em repouso';
        status.className = 'fire-status resting';
    }

    // Analogia biológica
    updateBioText(r);

    // Canvas
    drawNeuron(r);
}

// ===== TEXTO BIOLÓGICO =====
function updateBioText(r) {
    const excCount = [r.w1, r.w2, r.w3].filter(w => w > 0).length;
    const inhCount = [r.w1, r.w2, r.w3].filter(w => w < 0).length;
    const isFiring = (activationFn === 'step' && r.output >= 1) ||
                     (activationFn !== 'step' && r.output > 0.5);

    let text = `<strong>${excCount} sinapses excitatórias</strong> (glutamato, w>0) e `;
    text += `<strong>${inhCount} sinapses inibitórias</strong> (GABA, w<0). `;
    text += `A soma ponderada (z=${r.z.toFixed(3)}) simula a integração no soma. `;

    if (isFiring) {
        text += `O potencial ultrapassou o limiar → <strong style="color:#fbbf24">potencial de ação gerado!</strong> `;
        text += `No biológico: canais Na⁺ abrem → despolarização → +30mV.`;
    } else {
        text += `O potencial NÃO atingiu o limiar → <strong>sem disparo</strong>. `;
        text += `No biológico: o neurônio permanece em repouso (-70mV).`;
    }

    document.getElementById('bio-text').innerHTML = text;
}

// ===== DESENHAR NEURÔNIO NO CANVAS =====
function drawNeuron(r) {
    const canvas = document.getElementById('neuronCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    const isFiring = (activationFn === 'step' && r.output >= 1) ||
                     (activationFn !== 'step' && r.output > 0.5);

    const somaX = 250, somaY = 210, somaR = 45;

    // Dendritos (linhas de entrada)
    const inputs = [
        { x: 50, y: 100, val: r.x1, w: r.w1, label: 'x₁' },
        { x: 50, y: 210, val: r.x2, w: r.w2, label: 'x₂' },
        { x: 50, y: 320, val: r.x3, w: r.w3, label: 'x₃' }
    ];

    inputs.forEach(inp => {
        const isExc = inp.w >= 0;
        const alpha = 0.3 + Math.abs(inp.val) * 0.7;
        const color = isExc ? `rgba(52,211,153,${alpha})` : `rgba(248,113,113,${alpha})`;
        const lineWidth = 1 + Math.abs(inp.w) * 3;

        // Linha do dendrito
        ctx.beginPath();
        ctx.moveTo(inp.x + 30, inp.y);
        ctx.bezierCurveTo(inp.x + 100, inp.y, somaX - 80, somaY, somaX - somaR, somaY);
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.stroke();

        // Círculo da entrada
        ctx.beginPath();
        ctx.arc(inp.x + 15, inp.y, 14, 0, Math.PI * 2);
        ctx.fillStyle = isExc ? 'rgba(52,211,153,0.2)' : 'rgba(248,113,113,0.2)';
        ctx.fill();
        ctx.strokeStyle = isExc ? '#34d399' : '#f87171';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        ctx.fillStyle = '#e8eaf0';
        ctx.font = '600 13px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(inp.label, inp.x + 15, inp.y + 5);

        // Peso
        const midX = inp.x + 90;
        const midY = inp.y + (somaY - inp.y) * 0.3;
        ctx.fillStyle = isExc ? '#34d399' : '#f87171';
        ctx.font = '500 11px JetBrains Mono';
        ctx.fillText(`w=${inp.w.toFixed(1)}`, midX, midY - 8);
    });

    // Soma (corpo celular)
    const somaGrad = ctx.createRadialGradient(somaX, somaY, 10, somaX, somaY, somaR + 10);
    if (isFiring) {
        somaGrad.addColorStop(0, 'rgba(251,191,36,0.4)');
        somaGrad.addColorStop(0.6, 'rgba(251,191,36,0.15)');
        somaGrad.addColorStop(1, 'rgba(251,191,36,0)');
        ctx.beginPath();
        ctx.arc(somaX, somaY, somaR + 15, 0, Math.PI * 2);
        ctx.fillStyle = somaGrad;
        ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(somaX, somaY, somaR, 0, Math.PI * 2);
    const bodyGrad = ctx.createRadialGradient(somaX - 10, somaY - 10, 5, somaX, somaY, somaR);
    bodyGrad.addColorStop(0, isFiring ? '#5a4a10' : '#2a3050');
    bodyGrad.addColorStop(1, isFiring ? '#3d3410' : '#1a1f35');
    ctx.fillStyle = bodyGrad;
    ctx.fill();
    ctx.strokeStyle = isFiring ? '#fbbf24' : '#4f8fff';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Label do soma
    ctx.fillStyle = '#e8eaf0';
    ctx.font = '700 13px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Σ + b', somaX, somaY - 6);
    ctx.font = '500 11px JetBrains Mono';
    ctx.fillStyle = isFiring ? '#fbbf24' : '#4f8fff';
    ctx.fillText(`z = ${r.z.toFixed(3)}`, somaX, somaY + 14);

    // Axon Hillock (função de ativação)
    const ahX = somaX + somaR + 30;
    const ahY = somaY;

    ctx.beginPath();
    ctx.moveTo(somaX + somaR, somaY);
    ctx.lineTo(ahX - 15, ahY);
    ctx.strokeStyle = isFiring ? '#fbbf24' : '#6b7394';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Triângulo do axon hillock
    ctx.beginPath();
    ctx.moveTo(ahX - 12, ahY - 15);
    ctx.lineTo(ahX + 15, ahY);
    ctx.lineTo(ahX - 12, ahY + 15);
    ctx.closePath();
    ctx.fillStyle = isFiring ? 'rgba(251,191,36,0.2)' : 'rgba(79,143,255,0.15)';
    ctx.fill();
    ctx.strokeStyle = isFiring ? '#fbbf24' : '#4f8fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#e8eaf0';
    ctx.font = '500 10px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText('f(z)', ahX, ahY + 30);

    // Axônio (saída)
    const outX = W - 60;
    ctx.beginPath();
    ctx.moveTo(ahX + 15, ahY);
    ctx.lineTo(outX - 15, ahY);
    ctx.strokeStyle = isFiring ? '#fbbf24' : '#6b7394';
    ctx.lineWidth = 3;
    ctx.setLineDash(isFiring ? [] : [6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Spike marks se disparando
    if (isFiring) {
        for (let sx = ahX + 40; sx < outX - 30; sx += 35) {
            ctx.beginPath();
            ctx.moveTo(sx, ahY);
            ctx.lineTo(sx + 5, ahY - 14);
            ctx.lineTo(sx + 10, ahY + 5);
            ctx.lineTo(sx + 15, ahY);
            ctx.strokeStyle = '#fbbf24';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }

    // Botão sináptico (saída)
    ctx.beginPath();
    ctx.arc(outX, ahY, 18, 0, Math.PI * 2);
    ctx.fillStyle = isFiring ? 'rgba(251,191,36,0.2)' : 'rgba(79,143,255,0.1)';
    ctx.fill();
    ctx.strokeStyle = isFiring ? '#fbbf24' : '#4f8fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#e8eaf0';
    ctx.font = '700 14px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(r.output.toFixed(2), outX, ahY + 5);

    // Labels das partes
    ctx.font = '500 10px Inter';
    ctx.fillStyle = '#6b7394';
    ctx.textAlign = 'center';
    ctx.fillText('DENDRITOS', 60, 380);
    ctx.fillText('SOMA', somaX, 380);
    ctx.fillText('AXON HILLOCK', ahX, 380);
    ctx.fillText('SAÍDA', outX, 380);

    // Bias label
    ctx.fillStyle = '#9ca3b8';
    ctx.font = '500 11px JetBrains Mono';
    ctx.fillText(`bias = ${r.bias.toFixed(1)}`, somaX, somaY + 60);
}

// ===== PRESETS (PORTAS LÓGICAS) =====
function loadPreset(gate) {
    const info = document.getElementById('preset-info');

    // Usar apenas 2 entradas (x3 = 0, w3 = 0) para portas lógicas
    switch (gate) {
        case 'and':
            setValues(1, 1, 0, 1, 1, 0, -1.5);
            document.querySelector('.act-btn.active')?.classList.remove('active');
            document.getElementById('btn-step').classList.add('active');
            activationFn = 'step';
            info.innerHTML = `<strong>AND:</strong> Só dispara quando AMBAS entradas = 1. Threshold = 1.5 (precisa de x₁×1 + x₂×1 = 2 > 1.5). 
            <br>👉 Teste: coloque x₁=1, x₂=0 — não dispara. x₁=1, x₂=1 — dispara!`;
            break;
        case 'or':
            setValues(1, 0, 0, 1, 1, 0, -0.5);
            document.querySelector('.act-btn.active')?.classList.remove('active');
            document.getElementById('btn-step').classList.add('active');
            activationFn = 'step';
            info.innerHTML = `<strong>OR:</strong> Dispara quando QUALQUER entrada = 1. Threshold = 0.5 (precisa de apenas 1 > 0.5).
            <br>👉 Teste: x₁=0, x₂=0 — não dispara. Qualquer um = 1 — dispara!`;
            break;
        case 'not':
            setValues(1, 0, 0, -1, 0, 0, 0.5);
            document.querySelector('.act-btn.active')?.classList.remove('active');
            document.getElementById('btn-step').classList.add('active');
            activationFn = 'step';
            info.innerHTML = `<strong>NOT:</strong> Inverte x₁. Peso negativo (-1) = sinapse inibitória (GABA). 
            <br>👉 x₁=0 → saída=1. x₁=1 → saída=0. A inibição "silencia" o neurônio.`;
            break;
        case 'xor':
            setValues(1, 0, 0, 1, 1, 0, -0.5);
            document.querySelector('.act-btn.active')?.classList.remove('active');
            document.getElementById('btn-step').classList.add('active');
            activationFn = 'step';
            info.innerHTML = `<strong style="color:#fb923c">⚠️ XOR: IMPOSSÍVEL com um único neurônio!</strong><br>
            Minsky & Papert (1969) provaram que o XOR não é linearmente separável. 
            Precisa de pelo menos 2 camadas (MLP). Este foi o argumento que causou o Inverno da IA!
            <br>👉 Teste: x₁=1,x₂=0 → 1 ✓. Mas x₁=1,x₂=1 → 1 ✗ (deveria ser 0).`;
            break;
    }
    updateDisplays();
}

function setValues(x1, x2, x3, w1, w2, w3, bias) {
    document.getElementById('x1').value = x1;
    document.getElementById('x2').value = x2;
    document.getElementById('x3').value = x3;
    document.getElementById('w1').value = w1;
    document.getElementById('w2').value = w2;
    document.getElementById('w3').value = w3;
    document.getElementById('bias').value = bias;
}

// ===== EVENT LISTENERS =====
document.querySelectorAll('.slider, .weight-input').forEach(el => {
    el.addEventListener('input', updateDisplays);
});

document.querySelectorAll('.act-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelector('.act-btn.active')?.classList.remove('active');
        btn.classList.add('active');
        activationFn = btn.dataset.fn;
        updateDisplays();
    });
});

// ===== INICIALIZAR =====
updateDisplays();
