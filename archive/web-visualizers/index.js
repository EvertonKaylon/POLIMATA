// Neuromorphic Lab UI Handler
document.addEventListener("DOMContentLoaded", () => {
    // Configurações e elementos
    const btnSimular = document.getElementById("btn_simular");
    const statusLog = document.getElementById("sim_status_log");
    const networkSvg = document.getElementById("network_svg");
    const canvas = document.getElementById("error_chart");
    
    // Sliders
    const inputs = {
        taxa_aprendizado: document.getElementById("taxa_aprendizado"),
        epocas_replay: document.getElementById("epocas_replay"),
        passos_inferencia: document.getElementById("passos_inferencia"),
        nivel_ruido: document.getElementById("nivel_ruido"),
        usar_meta_pcn: document.getElementById("usar_meta_pcn"),
        usar_gaba: document.getElementById("usar_gaba"),
        limiar_gaba: document.getElementById("limiar_gaba"),
        usar_ternario: document.getElementById("usar_ternario"),
        limiar_memristor: document.getElementById("limiar_memristor"),
        limiar_poda: document.getElementById("limiar_poda")
    };

    // Displays de valores
    const displays = {
        taxa_aprendizado: document.getElementById("val_taxa_aprendizado"),
        epocas_replay: document.getElementById("val_epocas_replay"),
        passos_inferencia: document.getElementById("val_passos_inferencia"),
        nivel_ruido: document.getElementById("val_nivel_ruido"),
        limiar_gaba: document.getElementById("val_limiar_gaba"),
        limiar_memristor: document.getElementById("val_limiar_limiar_memristor"), // Corrigido via JS fallback
        limiar_poda: document.getElementById("val_limiar_poda")
    };

    // Corrigir ids e referências adicionais de forma segura
    const valMemristor = document.getElementById("val_limiar_memristor");

    // Lógica para sincronizar Sliders com Displays textuais
    Object.keys(inputs).forEach(key => {
        const input = inputs[key];
        if (!input) return;

        if (input.type === "range") {
            input.addEventListener("input", (e) => {
                let value = parseFloat(e.target.value);
                if (key === "nivel_ruido") {
                    displays[key].innerText = `${Math.round(value * 100)}%`;
                } else if (key === "limiar_memristor" && valMemristor) {
                    valMemristor.innerText = value.toFixed(2);
                } else if (displays[key]) {
                    displays[key].innerText = value.toFixed(3).replace(/\.000$/, "");
                }
            });
        }
    });

    // Controlar visibilidade de sub-controles baseados em checkboxes
    const toggleGabaGroup = () => {
        const group = document.getElementById("gaba_threshold_group");
        group.style.opacity = inputs.usar_gaba.checked ? "1" : "0.3";
        inputs.limiar_gaba.disabled = !inputs.usar_gaba.checked;
    };
    inputs.usar_gaba.addEventListener("change", toggleGabaGroup);
    toggleGabaGroup();

    const toggleTernaryGroup = () => {
        const group = document.getElementById("memristor_threshold_group");
        group.style.opacity = inputs.usar_ternario.checked ? "1" : "0.3";
        inputs.limiar_memristor.disabled = !inputs.usar_ternario.checked;
    };
    inputs.usar_ternario.addEventListener("change", toggleTernaryGroup);
    toggleTernaryGroup();

    // Definição da Topologia da Rede Neural (Coordenadas SVG)
    const topologia = {
        camada1: { n: 10, x: 85, r: 8, label: "L1: Sensorial" },
        camada2: { n: 6, x: 300, r: 12, label: "L2: Intermediária" },
        camada3: { n: 4, x: 515, r: 15, label: "L3: Abstrata" }
    };

    // Ajustar tamanho do Canvas interno
    const resizeCanvas = () => {
        canvas.width = canvas.parentElement.clientWidth * window.devicePixelRatio;
        canvas.height = canvas.parentElement.clientHeight * window.devicePixelRatio;
    };
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

    // Desenhar grafo da rede inicialmente
    const desenharEstruturaRede = (W1 = null, W2 = null) => {
        networkSvg.innerHTML = ""; // Limpar SVG
        
        const nodes = [];
        const height = 450;
        
        // Auxiliar de cálculo de posições Y dos neurônios
        const obterPosicoesY = (n, totalH) => {
            const pos = [];
            const step = (totalH - 80) / (n - 1);
            for (let i = 0; i < n; i++) {
                pos.push(40 + i * step);
            }
            return pos;
        };

        const y1 = obterPosicoesY(topologia.camada1.n, height);
        const y2 = obterPosicoesY(topologia.camada2.n, height);
        const y3 = obterPosicoesY(topologia.camada3.n, height);

        // 1. Desenhar sinapses (Conexões/Links) antes dos círculos para sobreposição correta
        // Conexões L1 -> L2
        for (let i = 0; i < topologia.camada1.n; i++) {
            for (let j = 0; j < topologia.camada2.n; j++) {
                let w = W1 ? W1[i][j] : (Math.random() * 0.4 - 0.2);
                let color = "rgba(255,255,255,0.06)";
                let strokeWidth = 0.5;

                if (W1) {
                    if (w > 0.01) {
                        color = `rgba(6, 182, 212, ${Math.min(1.0, 0.2 + w * 0.8)})`;
                        strokeWidth = Math.min(4, 0.5 + w * 3);
                    } else if (w < -0.01) {
                        color = `rgba(236, 72, 153, ${Math.min(1.0, 0.2 + Math.abs(w) * 0.8)})`;
                        strokeWidth = Math.min(4, 0.5 + Math.abs(w) * 3);
                    } else {
                        color = "rgba(255, 255, 255, 0.02)"; // Escondida/Podada
                        strokeWidth = 0.2;
                    }
                }

                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", topologia.camada1.x);
                line.setAttribute("y1", y1[i]);
                line.setAttribute("x2", topologia.camada2.x);
                line.setAttribute("y2", y2[j]);
                line.setAttribute("stroke", color);
                line.setAttribute("stroke-width", strokeWidth);
                networkSvg.appendChild(line);
            }
        }

        // Conexões L2 -> L3
        for (let j = 0; j < topologia.camada2.n; j++) {
            for (let k = 0; k < topologia.camada3.n; k++) {
                let w = W2 ? W2[j][k] : (Math.random() * 0.4 - 0.2);
                let color = "rgba(255,255,255,0.06)";
                let strokeWidth = 0.5;

                if (W2) {
                    if (w > 0.01) {
                        color = `rgba(6, 182, 212, ${Math.min(1.0, 0.2 + w * 0.8)})`;
                        strokeWidth = Math.min(5, 0.5 + w * 4);
                    } else if (w < -0.01) {
                        color = `rgba(236, 72, 153, ${Math.min(1.0, 0.2 + Math.abs(w) * 0.8)})`;
                        strokeWidth = Math.min(5, 0.5 + Math.abs(w) * 4);
                    } else {
                        color = "rgba(255, 255, 255, 0.02)";
                        strokeWidth = 0.2;
                    }
                }

                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", topologia.camada2.x);
                line.setAttribute("y1", y2[j]);
                line.setAttribute("x2", topologia.camada3.x);
                line.setAttribute("y2", y3[k]);
                line.setAttribute("stroke", color);
                line.setAttribute("stroke-width", strokeWidth);
                networkSvg.appendChild(line);
            }
        }

        // 2. Desenhar os neurônios (Círculos)
        const desenharCamada = (x, yArr, r, color, glowClass) => {
            yArr.forEach((y, idx) => {
                const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                circle.setAttribute("cx", x);
                circle.setAttribute("cy", y);
                circle.setAttribute("r", r);
                circle.setAttribute("fill", color);
                circle.setAttribute("stroke", "rgba(255,255,255,0.4)");
                circle.setAttribute("stroke-width", "1.5");
                circle.style.filter = "drop-shadow(0px 0px 4px rgba(255,255,255,0.15))";
                networkSvg.appendChild(circle);
            });
        };

        desenharCamada(topologia.camada1.x, y1, topologia.camada1.r, "#1a162b", "l1-glow");
        desenharCamada(topologia.camada2.x, y2, topologia.camada2.r, "#0b2b3a", "l2-glow");
        desenharCamada(topologia.camada3.x, y3, topologia.camada3.r, "#3b1b4a", "l3-glow");

        // Desenhar labels de camadas
        const desenharTexto = (text, x, y) => {
            const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
            txt.setAttribute("x", x);
            txt.setAttribute("y", y);
            txt.setAttribute("fill", "#a29bb8");
            txt.setAttribute("font-family", "Space Grotesk");
            txt.setAttribute("font-size", "10px");
            txt.setAttribute("text-anchor", "middle");
            txt.textContent = text;
            networkSvg.appendChild(txt);
        };
        desenharTexto(topologia.camada1.label, topologia.camada1.x, 20);
        desenharTexto(topologia.camada2.label, topologia.camada2.x, 20);
        desenharTexto(topologia.camada3.label, topologia.camada3.x, 20);
    };

    desenharEstruturaRede();

    // Renderizar gráfico de erro customizado no Canvas
    const desenharGraficoErro = (curva) => {
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        
        // Limpar canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const w = canvas.width;
        const h = canvas.height;
        
        // Background
        ctx.fillStyle = "rgba(0, 0, 0, 0.15)";
        ctx.fillRect(0, 0, w, h);

        if (!curva || curva.length === 0) return;

        const margin = { top: 20, right: 20, bottom: 30, left: 50 };
        const graphW = w - margin.left - margin.right;
        const graphH = h - margin.top - margin.bottom;

        // Determinar escalas
        const maxEp = Math.max(...curva.map(d => d.epoca));
        const minEp = Math.min(...curva.map(d => d.epoca));
        const maxErr = 0.55;
        const minErr = 0.0;

        // Desenhar grades y horizontais
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const valY = maxErr - (i * maxErr / 4);
            const py = margin.top + (graphH * i / 4);
            
            ctx.beginPath();
            ctx.moveTo(margin.left, py);
            ctx.lineTo(w - margin.right, py);
            ctx.stroke();

            // Texto do eixo Y
            ctx.fillStyle = "#a29bb8";
            ctx.font = `${Math.round(10 * window.devicePixelRatio)}px monospace`;
            ctx.fillText(valY.toFixed(2), margin.left - 38, py + 4);
        }

        // Dividir visualmente Dia 1 e Dia 2
        const divisorEpoca = maxEp / 2;
        const divisorX = margin.left + (graphW * (divisorEpoca - minEp) / (maxEp - minEp));
        ctx.strokeStyle = "rgba(138, 43, 226, 0.35)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(divisorX, margin.top);
        ctx.lineTo(divisorX, h - margin.bottom);
        ctx.stroke();
        ctx.setLineDash([]); // Reset dash

        ctx.fillStyle = "rgba(138, 43, 226, 0.6)";
        ctx.font = `${Math.round(9 * window.devicePixelRatio)}px Space Grotesk`;
        ctx.fillText("Dia 1 (Task A)", divisorX - 75, margin.top + 15);
        ctx.fillText("Dia 2 (Task B)", divisorX + 10, margin.top + 15);

        // Desenhar linha do erro preditivo
        ctx.beginPath();
        curva.forEach((pt, idx) => {
            const px = margin.left + (graphW * (pt.epoca - minEp) / (maxEp - minEp));
            const py = margin.top + graphH - (graphH * (pt.erro - minErr) / (maxErr - minErr));
            if (idx === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        });

        ctx.strokeStyle = "#06b6d4";
        ctx.lineWidth = 3;
        ctx.shadowColor = "rgba(6, 182, 212, 0.5)";
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.shadowBlur = 0; // Reset glow

        // Eixo X labels
        ctx.fillStyle = "#a29bb8";
        ctx.font = `${Math.round(10 * window.devicePixelRatio)}px Space Grotesk`;
        ctx.fillText(`Época 0`, margin.left, h - 10);
        ctx.fillText(`Época ${maxEp}`, w - margin.right - 50, h - 10);
    };

    // Estado local para armazenar pesos atuais para interpolação suave
    let W1_atual = null;
    let W2_atual = null;

    const gerarPesosAleatorios = (rows, cols) => {
        return Array.from({ length: rows }, () => 
            Array.from({ length: cols }, () => Math.random() * 0.4 - 0.2)
        );
    };

    const interpolarMatriz = (M1, M2, t) => {
        if (!M1 || !M2) return M2;
        return M1.map((row, i) => 
            row.map((val, j) => val + (M2[i][j] - val) * t)
        );
    };

    // Submissão da simulação e chamadas Ajax
    btnSimular.addEventListener("click", () => {
        btnSimular.innerText = "Simulando Replays...";
        btnSimular.disabled = true;
        statusLog.innerText = "Inicializando redes profundas...\nIniciando Vigília: Gravando experiências no CA3 Hipocampal...\nIniciando sono e aguardando replays...";

        // Guardar pesos anteriores para iniciar a animação
        const W1_inicial = W1_atual || gerarPesosAleatorios(topologia.camada1.n, topologia.camada2.n);
        const W2_inicial = W2_atual || gerarPesosAleatorios(topologia.camada2.n, topologia.camada3.n);

        // Construir query string de parâmetros
        const queryParams = new URLSearchParams();
        Object.keys(inputs).forEach(key => {
            const input = inputs[key];
            if (!input) return;
            
            if (input.type === "checkbox") {
                queryParams.append(key, input.checked ? "true" : "false");
            } else {
                queryParams.append(key, input.value);
            }
        });

        // Fazer requisição GET para a API de simulação
        fetch(`/api/simular?${queryParams.toString()}`)
            .then(res => {
                if (!res.ok) throw new Error("Erro do Servidor HTTP.");
                return res.json();
            })
            .then(data => {
                const totalPassos = data.curva_treinamento.length;
                let passoAtual = 0;

                const animarValor = (idBar, idText, targetVal) => {
                    const bar = document.getElementById(idBar);
                    const txt = document.getElementById(idText);
                    bar.style.width = `${targetVal}%`;
                    txt.innerText = `${targetVal.toFixed(1)}%`;
                };

                // Loop de animação suave em tempo real (60fps)
                function animarFrame() {
                    if (passoAtual >= totalPassos) {
                        // Salvar pesos finais como estado atual
                        W1_atual = data.W1;
                        W2_atual = data.W2;

                        // Finalizar botões e logs
                        btnSimular.innerText = "Iniciar Ciclo Vigília-Sono";
                        btnSimular.disabled = false;

                        // Renderização final perfeita
                        desenharEstruturaRede(data.W1, data.W2);
                        desenharGraficoErro(data.curva_treinamento);

                        // Animar as barras de métricas finais
                        animarValor("bar_retencao", "metric_retencao", data.retencao_a);
                        animarValor("bar_sucesso", "metric_sucesso", data.sucesso_b);
                        animarValor("bar_asr", "metric_asr", data.asr);
                        animarValor("bar_cr", "metric_cr", data.cr);

                        let logs = `Simulação concluída com sucesso.\n`;
                        logs += `Retenção de Task A: ${data.retencao_a.toFixed(1)}% | Aprendizado de B: ${data.sucesso_b.toFixed(1)}%\n`;
                        logs += `Sparsity W1: ${data.esparsidade_w1.toFixed(1)}% | Sparsity W2: ${data.esparsidade_w2.toFixed(1)}%\n`;
                        
                        if (data.gaba_bloqueou_b) {
                            logs += `[MS-MEC Switch]: Task B detectada como ANOMALIA e bloqueada com sucesso! Prior de A protegido.`;
                        } else {
                            logs += `[MS-MEC Switch]: Task B registrada e fundida na rede.`;
                        }
                        statusLog.innerText = logs;
                        return;
                    }

                    passoAtual += 1; // Avança um passo de treinamento por frame

                    // Renderizar sub-curva de treinamento até o ponto atual
                    const curvaParcial = data.curva_treinamento.slice(0, passoAtual);
                    desenharGraficoErro(curvaParcial);

                    // Interpolar pesos sinápticos dinamicamente
                    const t = passoAtual / totalPassos;
                    const W1_interp = interpolarMatriz(W1_inicial, data.W1, t);
                    const W2_interp = interpolarMatriz(W2_inicial, data.W2, t);
                    desenharEstruturaRede(W1_interp, W2_interp);

                    // Atualizar logs rápidos da console durante o processamento do sono
                    const ponto = data.curva_treinamento[passoAtual - 1];
                    statusLog.innerText = `Processando Replay de Sono...\n[Época ${ponto.epoca}] - Erro L1: ${ponto.erro.toFixed(4)} (${ponto.fase})`;

                    // Progresso intermediário suave nas métricas
                    const interp_ret = 50 + (data.retencao_a - 50) * t;
                    const interp_suc = 50 + (data.sucesso_b - 50) * t;
                    animarValor("bar_retencao", "metric_retencao", interp_ret);
                    animarValor("bar_sucesso", "metric_sucesso", interp_suc);

                    requestAnimationFrame(animarFrame);
                }

                // Iniciar loop animado
                requestAnimationFrame(animarFrame);
            })
            .catch(err => {
                btnSimular.innerText = "Iniciar Ciclo Vigília-Sono";
                btnSimular.disabled = false;
                statusLog.innerText = `Erro: ${err.message}`;
            });
    });
});
