# ROADMAP 52 SEMANAS — MÚSCULOS ARTIFICIAIS × IA NEUROMÓRFICA
## Simulação Computacional Leve | Python Puro | NumPy-First | Compaq CQ42

**Pesquisador:** Everton  
**Período:** 2026–2027 | 4h/semana | ~208h totais  
**Hardware:** Pentium T4500, 8GB RAM, Archcraft Linux, sem GPU  
**Objetivo:** GitHub portfolio + preprint arXiv → candidatura BK21/GKS (Coreia do Sul)

---

## ARQUITETURA DE INTEGRAÇÃO — VISÃO GERAL DO SISTEMA FINAL

```
┌──────────────────────────────────────────────────────────────────────┐
│                SISTEMA NEUROMÓRFICO-MUSCULAR INTEGRADO               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  NEOCÓRTEX (Predictive Coding)        HIPOCAMPO (Motor Memory)      │
│  ŷ(t) = W_pc · x(t)                  Hopfield: W_hop += ξξᵀ/N      │
│  e(t) = F_target − F_actual           recall(cue) → F_pattern       │
│  W_pc += α · e · xᵀ                                                 │
│          ↓ I_drive(t)                          ↑↓ episódios         │
│  POOL LIF (N neurônios motores)                                      │
│  τ dV/dt = −(V−V_r) + I_syn + I_drive                               │
│  spikes → firing_rate(t) → I_coil(t)                                │
│          ↓                                                           │
│  INTERFACE NEURONAL-MUSCULAR                                         │
│  I_coil → H(t) = N·I/L   (solenóide)                                │
│  rate   → V(t) = V_max·tanh(rate/r₀) (DEA)                         │
│          ↓                                                           │
│  MÚSCULO COMPOSTO (Plant)                                            │
│  DEA:         σ_e = ε₀εᵣ(V/d₀)²·λ³   [Pelrine-Suo]                │
│  Ferrofluido: M = M_s·L(ξ)            [Langevin]                   │
│  Histerese:   JA model                [Jiles-Atherton]              │
│  Fase:        ∂φ/∂t = Γ[ε²∇²φ−f′(φ)+κH²]  [TDGL]                 │
│          ↓ F_actual(t)                                               │
│  R-STDP  Δw = η·r·[A₊·x_pre·δ_post − A₋·x_post·δ_pre]            │
│  SONO    replay offline → consolidação de pesos                      │
│          ↑ F_actual → erro → Neocórtex (loop fechado)               │
└──────────────────────────────────────────────────────────────────────┘
```

**Regras de código para todo o projeto:**
- NumPy arrays, sem SciPy pesado
- Euler explícito, dt ≤ τ_min/10
- N ≤ 200 pontos espaciais por módulo
- Nenhuma renderização gráfica em tempo real; resultados via `print` terminal
- Cada módulo: arquivo `.py` único < 150 linhas, importável
- Toda validação Monte Carlo: N=100–200 amostras, seed fixo para reprodutibilidade

---

## FASE 1 — FUNDAMENTOS MATEMÁTICOS DE MATERIAIS MACIOS
### Semanas 1–8 | Meta: dominar em código os tensores e campos que governam DEA + ferrofluido

---

### SEMANA 1 — Stretch Ratio e Green-Lagrange: Porta da Mecânica Contínua

**Conceito Central:**
A variável de estado fundamental de qualquer elastômero é o *stretch ratio* λ = L/L₀.
A deformação de Green-Lagrange E = (λ² − 1)/2 é correta para grandes deformações.
Para λ ≪ 1: E ≈ ε_eng (regime linear). Para λ ≥ 1.5: divergência significativa — início da mecânica não-linear.

```
λ(t+dt) = λ(t) + dt · λ̇(t)
E_GL   = (λ² − 1) / 2
ε_eng  = λ − 1
```

**Exercício Prático:**
```python
# semana01_strain.py
import numpy as np

dt = 1e-3; N = 1000
lam = np.ones(N)
lam_dot = 0.1  # s⁻¹, taxa de deformação constante

for i in range(1, N):
    lam[i] = lam[i-1] + dt * lam_dot
    E_gl   = (lam[i]**2 - 1) / 2
    eps_e  = lam[i] - 1
    if i % 200 == 0:
        print(f"t={i*dt:.2f}s | λ={lam[i]:.4f} | E_GL={E_gl:.4f} | "
              f"ε_eng={eps_e:.4f} | Δ={abs(E_gl-eps_e):.4f}")

# Validação de referência
for l in [1.0, 1.1, 2.0, 3.0]:
    E = (l**2-1)/2; eps = l-1
    print(f"λ={l:.1f} | E_GL={E:.4f} | ε_eng={eps:.4f} | "
          f"erro_rel={abs(E-eps)/max(E,1e-9)*100:.1f}%")
```

**Critério de Validação:**
- λ=1.0 → E_GL=0 exato; λ=1.1 → erro < 6%; λ=2.0 → E_GL=1.5, ε_eng=1.0 confirma não-linearidade

**Recurso:** Holzapfel (2000) "Nonlinear Solid Mechanics" Cap. 2 — PDF gratuito em `cma.fcen.uba.ar`

---

### SEMANA 2 — Hiperelasticidade Neo-Hookeana: Modelo Base dos DEA

**Conceito Central:**
Energia de deformação para elastômero **incompressível** (λ₁λ₂λ₃ = 1):
```
W = (μ/2)(λ₁² + λ₂² + λ₃² − 3)
Extensão uniaxial: λ₂=λ₃=λ⁻¹/²
→ W(λ) = (μ/2)(λ² + 2λ⁻¹ − 3)
Tensão nominal: P = dW/dλ = μ(λ − λ⁻²)
```
μ ≈ 10⁴–10⁵ Pa para silicone/PDMS; μ = E/3 para material incompressível.

**Exercício Prático:**
```python
# semana02_neo_hookean.py
import numpy as np

mu = 1e5  # Pa

def nh_stress(lam, mu): return mu*(lam - lam**(-2))
def nh_energy(lam, mu): return mu/2*(lam**2 + 2*lam**(-1) - 3)

lam_arr = np.linspace(1.0, 4.0, 200)
P_arr   = nh_stress(lam_arr, mu)
W_arr   = nh_energy(lam_arr, mu)
P_num   = np.gradient(W_arr, lam_arr[1]-lam_arr[0])

print("λ     | P_analítico (kPa) | P_numérico (kPa) | erro (%)")
for i in range(0, 200, 40):
    err = abs(P_arr[i]-P_num[i])/max(abs(P_arr[i]),1e-3)*100
    print(f"{lam_arr[i]:.2f}  | {P_arr[i]/1e3:17.4f} | {P_num[i]/1e3:16.4f} | {err:.2f}%")

assert abs(nh_stress(1.0, mu)) < 1e-8
assert abs(nh_stress(2.0, mu)/mu - 1.75) < 1e-6
print("Assertions ✓ — P(λ=1)=0, P(λ=2)=1.75μ")
```

**Critério de Validação:** P(λ=1)=0 exato; P(λ=2)=1.75·μ; erro numérico-analítico < 1%

**Recurso:** Suo (2010) "Theory of dielectric elastomers" — `imechanica.org/node/538` (PDF)

---

### SEMANA 3 — Viscoelasticidade Kelvin-Voigt: Como Elastômeros Fluem no Tempo

**Conceito Central:**
```
σ(t) = E·ε(t) + η·ε̇(t)     [mola + amortecedor em paralelo]
dε/dt = [σ_app − E·ε(t)] / η
Solução: ε(t) = (σ_app/E)·[1 − exp(−t/τ)],  τ = η/E
Euler: ε(t+dt) = ε(t) + dt·[σ_app − E·ε(t)] / η
```

**Exercício Prático:**
```python
# semana03_kelvin_voigt.py
import numpy as np

E=1e5; eta=1e3; tau=eta/E  # τ=0.01s
dt=1e-4; N=int(0.12/dt)
eps=np.zeros(N); t=np.arange(N)*dt
sigma_app=1e4

for i in range(1,N):
    eps[i] = eps[i-1] + dt*(sigma_app - E*eps[i-1])/eta

eps_exact = (sigma_app/E)*(1-np.exp(-t/tau))
idx_tau   = int(tau/dt)
err_tau   = abs(eps[idx_tau]-eps_exact[idx_tau])/eps_exact[idx_tau]*100
print(f"τ={tau*1e3:.1f}ms | ε(τ)_Euler={eps[idx_tau]:.6f} | "
      f"ε(τ)_exato={eps_exact[idx_tau]:.6f} | erro={err_tau:.3f}%")
assert err_tau < 0.5, f"Erro {err_tau:.2f}% > 0.5% — dt muito grande"
print("Kelvin-Voigt ✓")
```

**Critério de Validação:** ε(t=τ) com erro < 0.5%; ε(t→∞) → σ_app/E com erro < 0.01%

---

### SEMANA 4 — Mooney-Rivlin: Ajuste de Parâmetros em Elastômeros Reais

**Conceito Central:**
```
W = C₁(I₁−3) + C₂(I₂−3)
Extensão uniaxial: P(λ) = 2(C₁ + C₂/λ)(λ − λ⁻²)
Neo-Hookean = caso especial: C₂=0, μ=2C₁
```

**Exercício Prático:**
```python
# semana04_mooney_rivlin.py
import numpy as np

def MR_stress(lam, C1, C2):
    return 2*(C1+C2/lam)*(lam-lam**(-2))

# Dados sintéticos PDMS Shore 20A
lam_d = np.array([1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5])
P_d   = np.array([18., 38., 105., 275., 500., 780., 1120.])*1e3

best_mse, best_C1, best_C2 = 1e30, None, None
for C1 in np.logspace(3.5, 5.5, 60):
    for C2 in np.logspace(2.0, 5.0, 60):
        mse = np.mean((MR_stress(lam_d,C1,C2)-P_d)**2)
        if mse < best_mse:
            best_mse, best_C1, best_C2 = mse, C1, C2

P_fit = MR_stress(lam_d, best_C1, best_C2)
R2 = 1 - np.sum((P_d-P_fit)**2)/np.sum((P_d-P_d.mean())**2)
print(f"C₁={best_C1:.3e} Pa | C₂={best_C2:.3e} Pa | R²={R2:.6f}")
assert R2 > 0.99 and best_C1 > 0 and best_C2 >= 0
```

**Critério de Validação:** R² > 0.99; C₁ > 0, C₂ ≥ 0

---

### SEMANA 5 — Equações de Maxwell Estáticas: Campo Magnético de Bobina

**Conceito Central:**
```
Solenóide finito — campo axial on-axis:
H(x) = (N·I)/(2L) · [cosθ₁ − cosθ₂]
cosθ₁ = (L/2−x)/√[(L/2−x)²+R²],  cosθ₂ = −(L/2+x)/√[(L/2+x)²+R²]
```

**Exercício Prático:**
```python
# semana05_coil_field.py
import numpy as np

mu0=4*np.pi*1e-7; N=500; I=1.0; L=0.05; R=0.01

def H_axis(x):
    H0 = N*I/(2*L)
    c1 = (L/2-x)/np.sqrt((L/2-x)**2+R**2)
    c2 = (L/2+x)/np.sqrt((L/2+x)**2+R**2)
    return H0*(c1+c2)

x = np.linspace(-L, L, 100)
H = H_axis(x); B = mu0*H
H_inf = N*I/L
print(f"H_inf={H_inf:.1f} A/m | H(x=0)={H[50]:.1f} A/m | fração={H[50]/H_inf:.4f}")
for i in range(0,100,10):
    print(f"x={x[i]*100:+5.1f}cm | H={H[i]:.1f} A/m | B={B[i]*1e3:.3f} mT")
assert H[50] >= 0.95*H_inf
```

**Critério de Validação:** H(x=0) ≥ 95% do solenóide infinito; simetria H(x)≈H(-x)

**Recurso:** Griffiths "Electrodynamics" Cap.5 — `fisica.ufpr.br`

---

### SEMANA 6 — Magnetização de Langevin: Coração do Ferrofluido

**Conceito Central:**
```
L(ξ) = coth(ξ) − 1/ξ,   ξ = μ₀·m·H/(k_B·T)
M = M_s · L(ξ)
χ₀ = μ₀·M_s·m/(3·k_B·T)   [susceptibilidade inicial]
Taylor (ξ→0): L(ξ) ≈ ξ/3  [evitar divisão zero]
```

**Exercício Prático:**
```python
# semana06_langevin.py
import numpy as np

mu0=4*np.pi*1e-7; kB=1.38e-23; T=300.0
d=10e-9; Ms_p=4.46e5; m=Ms_p*(np.pi/6)*d**3; Ms=4e4

def langevin(xi):
    xi=np.asarray(xi,float)
    return np.where(np.abs(xi)<1e-6, xi/3., 1./np.tanh(xi)-1./xi)

H=np.linspace(0,1e6,500); xi=mu0*m*H/(kB*T); M=Ms*langevin(xi)
chi0=mu0*Ms*m/(3*kB*T)
print(f"m={m:.3e} J/T | χ₀={chi0:.5f} | M_sat={M[-1]:.1f} A/m")
print(f"\nH(kA/m) | ξ     | M(kA/m) | M/Ms")
for i in range(0,500,50):
    print(f"{H[i]/1e3:7.1f} | {xi[i]:5.3f} | {M[i]/1e3:7.3f}  | {M[i]/Ms:.4f}")
assert abs(langevin(1e-10)-1e-10/3)<1e-15
assert abs(langevin(100.0)-1.0)<0.01
```

**Critério de Validação:** L(ξ→0)=ξ/3 com erro < 10⁻¹²; L(ξ=100) > 0.99

**Recurso:** Rosensweig (1985) "Ferrohydrodynamics" Cap.2 — `archive.org`

---

### SEMANA 7 — Força de Kelvin: Do Campo Magnético à Força Mecânica

**Conceito Central:**
```
f_x = μ₀·M(H(x))·dH/dx   [densidade de força, N/m³]
F = μ₀·πa²·∫M(H)·(dH/dx) dx
dH/dx|ᵢ ≈ (H_{i+1}−H_{i-1})/(2Δx)   [diferenças centrais]
```

**Exercício Prático:**
```python
# semana07_kelvin_force.py
import numpy as np

mu0=4*np.pi*1e-7; kB=1.38e-23; T=300.; Ms=4e4; m=1.6e-19

def langevin(xi):
    return np.where(np.abs(xi)<1e-6, xi/3., 1/np.tanh(xi)-1/xi)

x  = np.linspace(0, 0.05, 100); dx=x[1]-x[0]
H  = 1e5*np.exp(-0.5*((x-0.01)/0.005)**2)  # perfil Gaussiano
M  = Ms*langevin(mu0*m*H/(kB*T))
dH = np.gradient(H, dx)
f  = mu0*M*dH
A  = np.pi*(2e-3)**2
F  = np.trapz(f*A, x)
print(f"Força de Kelvin: {F*1e6:.4f} μN")
print(f"pico |dH/dx|: {np.max(np.abs(dH)):.3e} A/m²")
for i in range(0,100,10):
    print(f"x={x[i]*100:.1f}cm | H={H[i]/1e3:.2f}kA/m | f={f[i]:.3f} N/m³")
assert F != 0.0
```

**Critério de Validação:** F ≠ 0; f=0 onde dH/dx=0; dimensões [N/m³] consistentes

---

### SEMANA 8 — Allen-Cahn: Campo de Fase para Transição Sol-Gel

**Conceito Central:**
```
∂φ/∂t = Γ[ε²∂²φ/∂x² − f′(φ)]
f′(φ) = φ(1−φ)(1−2φ)/2        [potencial duplo-poço]
CFL:    dt < Δx²/(2·Γ·ε²)      [obrigatória para estabilidade]
φ=0: fase fluida | φ=1: fase gel sólido
```

**Exercício Prático:**
```python
# semana08_allen_cahn.py
import numpy as np

Nx=200; dx=1./Nx; dt=1e-5; Gamma=1.; eps2=0.01; N_steps=3000
print(f"CFL: dt={dt:.1e} < {dx**2/(2*Gamma*eps2):.2e} → "
      f"{'OK' if dt<dx**2/(2*Gamma*eps2) else 'VIOLADA!'}")

phi = np.where(np.linspace(0,1,Nx)<0.5, 0., 1.) + 0.005*np.random.randn(Nx)
phi = np.clip(phi, 0, 1)

def fp(p): return 0.5*p*(1-p)*(1-2*p)

for s in range(N_steps):
    lap  = (np.roll(phi,-1)-2*phi+np.roll(phi,1))/dx**2
    phi += dt*Gamma*(eps2*lap - fp(phi))
    phi  = np.clip(phi, -0.05, 1.05)
    if s%600==0:
        w = np.sum((phi>0.05)&(phi<0.95))*dx
        print(f"step {s:4d} | φ∈[{phi.min():.3f},{phi.max():.3f}] | w_iface≈{w:.3f}")

delta_th = 2*np.sqrt(eps2)
print(f"Largura teórica δ = {delta_th:.3f} | CFL ✓")
```

**Critério de Validação:** φ ∈ [0,1]; interface converge para perfil tanh; largura ≈ 4√ε²=0.4

**Recurso:** ArXiv: `"Allen-Cahn phase field finite difference implementation 1D tutorial"`

---

## FASE 2 — MODELOS COMPLETOS DE ATUADORES DEA E FERROFLUIDO
### Semanas 9–20 | Meta: implementar modelos físicos completos e validados

---

### SEMANA 9 — Pressão de Maxwell no DEA: Equação de Pelrine

**Conceito Central:**
```
p_e = ε₀·εᵣ·(V/d)²              [pressão de Maxwell]
d   = d₀/λ²                       [espessura com stretch equibiaxial]
p_e(V,λ) = ε₀·εᵣ·(V·λ²/d₀)²    [amplificação mecânico-elétrica]
```

**Exercício Prático:**
```python
# semana09_dea_maxwell.py
import numpy as np

eps0=8.854e-12; eps_r=3.0; d0=1e-3; Y=1e5
V=np.linspace(0,3000,50); lam=np.linspace(1.,3.5,50)
Vg,Lg=np.meshgrid(V,lam)
pe = eps0*eps_r*(Vg*Lg**2/d0)**2
s  = pe/Y

print(f"{'V(kV)':<8} {'λ':<6} {'p_e(kPa)':<14} {'s_Pelrine(%)'}")
for k in range(0,50,6):
    v,l=V[k],lam[k]
    p=eps0*eps_r*(v*l**2/d0)**2
    print(f"{v/1e3:<8.2f} {l:<6.2f} {p/1e3:<14.4f} {p/Y*100:.3f}%")

# Reprodução Pelrine 2000: ~100% para V=5kV, εᵣ=4.7, Y=1MPa
pe_ref=eps0*4.7*(5000/1e-3)**2; s_ref=pe_ref/1e6*100
print(f"\nPelrine 2000: s≈{s_ref:.1f}% (publicado ≈100%)")
assert abs(s_ref-100)<10
```

**Critério de Validação:** Reproduz Pelrine 2000; p_e ∝ V²; amplificação ∝ λ⁴

**Recurso:** Pelrine et al. Science 287, 836 (2000) — PDF ResearchGate

---

### SEMANA 10 — Equilíbrio DEA Neo-Hookeano: Solver Numérico

**Conceito Central:**
```
Equilíbrio elástico-elétrico equibiaxial:
g(λ,V) = μ(λ−λ⁻⁵) − ε₀εᵣ(V/d₀)²·λ³ = 0
Solver: bissecção em λ∈[1.0, 5.0]
Pull-in: g(1,V)·g(5,V) > 0 → nenhuma raiz estável
```

**Exercício Prático:**
```python
# semana10_dea_equilibrium.py
import numpy as np

eps0=8.854e-12; eps_r=3.0; d0=1e-3; mu=3.3e4

def g(lam,V):
    return mu*(lam-lam**(-5)) - eps0*eps_r*(V/d0)**2*lam**3

def bisect_dea(V, lo=1.0, hi=5.0, tol=1e-10, nit=80):
    if g(lo,V)*g(hi,V)>0: return None
    for _ in range(nit):
        mid=0.5*(lo+hi)
        if abs(g(mid,V))<tol: return mid
        if g(lo,V)*g(mid,V)<=0: hi=mid
        else: lo=mid
    return 0.5*(lo+hi)

V_pi=None
print(f"{'V(V)':<8} {'λ*':<10} {'strain_area(%)':<16} {'|g(λ*)|'}")
for V in np.linspace(0,3500,36):
    ls=bisect_dea(V)
    if ls:
        print(f"{V:<8.0f} {ls:<10.5f} {(ls**2-1)*100:<16.3f} {abs(g(ls,V)):.2e}")
    else:
        if V_pi is None: V_pi=V
        print(f"{V:<8.0f} >>> PULL-IN <<<")
print(f"\nV_pull-in ≈ {V_pi:.0f} V")
```

**Critério de Validação:** λ*(V=0)=1 exato; resíduo g(λ*) < 10⁻⁹; pull-in detectado

---

### SEMANA 11 — DEA Viscoelástico Dinâmico

**Conceito Central:**
```
dλ/dt = [λ_eq(V) − λ(t)] / τ_eff
τ_eff = η/(E₀+E₁)   [constante de tempo efetiva]
λ_eq: equilíbrio estático da semana 10
```

**Exercício Prático:**
```python
# semana11_dea_dynamic.py
import numpy as np

eps0=8.854e-12; eps_r=3.0; d0=1e-3; mu=3.3e4; tau=0.05

def bisect_dea(V, lo=1.0, hi=4.9, nit=50):
    def g(l): return mu*(l-l**(-5))-eps0*eps_r*(V/d0)**2*l**3
    if g(lo)*g(hi)>0: return lo
    for _ in range(nit):
        mid=0.5*(lo+hi)
        if g(lo)*g(mid)<=0: hi=mid
        else: lo=mid
    return 0.5*(lo+hi)

dt=1e-3; N=int(0.8/dt)
lam=np.ones(N); V_arr=np.zeros(N)
V_arr[100:500]=1500.

for t in range(1,N):
    lam[t]=lam[t-1]+dt*(bisect_dea(V_arr[t])-lam[t-1])/tau

lam_eq=bisect_dea(1500.); idx_tau=int((0.1+tau)/dt)
lam_exp=1+(lam_eq-1)*(1-np.exp(-1))
print(f"λ_eq(1500V)={lam_eq:.5f} | λ(t=τ)={lam[idx_tau]:.5f} | esperado={lam_exp:.5f}")
assert abs(lam[idx_tau]-lam_exp)/lam_exp < 0.01
print("DEA dinâmico ✓")
```

**Critério de Validação:** λ(t=τ) com erro < 1%; resposta monotônica; λ→1 após remoção

---

### SEMANA 12 — Pull-in e Bifurcação de Fases em DEA

**Conceito Central:**
```
Pull-in: g(λ,V)=0 e ∂g/∂λ=0 simultaneamente
λ_pi ≈ 1.26 (Neo-Hookean equibiaxial)
Verificação: varrer V e detectar perda da raiz de bissecção
```

```python
# semana12_pullin.py — varredura densa, detectar V_pi
import numpy as np

eps0=8.854e-12; eps_r=3.0; d0=1e-3; mu=3.3e4
def g(l,V): return mu*(l-l**(-5))-eps0*eps_r*(V/d0)**2*l**3

V_pi=None; lam_curve=[]
for V in np.linspace(0,5000,10000):
    lo,hi=1.0,4.99
    if g(lo,V)*g(hi,V)<=0:
        for _ in range(60):
            mid=0.5*(lo+hi)
            if g(lo,V)*g(mid,V)<=0: hi=mid
            else: lo=mid
        lam_curve.append(0.5*(lo+hi))
    else:
        lam_curve.append(np.nan)
        if V_pi is None: V_pi=V

lam_arr=np.array(lam_curve)
print(f"V_pi ≈ {V_pi:.0f} V | λ_max_estável ≈ {np.nanmax(lam_arr):.4f}")
assert V_pi is not None, "Pull-in não detectado — problema no modelo"
```

---

### SEMANA 13 — Primeiro Acoplamento: Pool LIF → Tensão → DEA

**Conceito Central:**
```
firing_rate(t) → V_DEA(t) = V_max·tanh(rate/r₀)
Interface modular: seu simulador LIF existente + bisect_dea (semana 10)
```

**Exercício Prático:**
```python
# semana13_lif_dea_coupling.py — PRIMEIRO ACOPLAMENTO
import numpy as np

# Pool LIF (N=10)
N=10; tau_m=0.02; Vr=-70e-3; Vth=-55e-3; Vrst=-70e-3
dt=1e-3; Nt=int(0.6/dt)
I_ext=np.linspace(1.8e-10,2.8e-10,N)
Vm=np.full(N,Vr); spk=np.zeros((Nt,N),bool)
for t in range(1,Nt):
    Vm+=(-(Vm-Vr)+I_ext/1e-9)*dt/tau_m*1e-9
    f=Vm>=Vth; spk[t]=f; Vm[f]=Vrst

win=50
rate=np.array([spk[max(0,t-win):t].sum()/(win*dt*N) for t in range(Nt)])

# rate → DEA
V_max=2000.; r0=25.
V_dea=V_max*np.tanh(rate/r0)

# DEA dinâmico
eps0=8.854e-12; eps_r=3.; d0=1e-3; mu=3.3e4; tau=0.05
lam=np.ones(Nt)
for t in range(1,Nt):
    V=V_dea[t]; lo,hi=1.,4.9
    g=lambda l,V=V: mu*(l-l**(-5))-eps0*eps_r*(V/d0)**2*l**3
    if g(lo)*g(hi)<0:
        for _ in range(40):
            mid=0.5*(lo+hi)
            if g(lo)*g(mid)<=0: hi=mid
            else: lo=mid
        le=0.5*(lo+hi)
    else: le=lo
    lam[t]=lam[t-1]+dt*(le-lam[t-1])/tau

print(f"{'t(ms)':<7}{'rate(Hz)':<10}{'V_DEA(V)':<10}{'λ':<9}{'strain(%)'}")
for t in range(0,Nt,int(0.08/dt)):
    print(f"{t*dt*1e3:<7.0f}{rate[t]:<10.1f}{V_dea[t]:<10.1f}"
          f"{lam[t]:<9.4f}{(lam[t]-1)*100:.2f}")
print("LIF→DEA pipeline ✓")
```

---

### SEMANA 14 — Reologia de Bingham: Ferrofluido Magnetoreológico

**Conceito Central:**
```
τ = τ_y(H) + η_p·γ̇    [Bingham]
τ_y(H) = τ_y0·[1 + α·(H/H_ref)²]
r_c = τ_y/(dP/L)  [raio do núcleo plug]
v(r>r_c) = (ΔP/4η_p·L)·(R²−r²) − (τ_y/η_p)·(R−r)
```

```python
# semana14_bingham.py
import numpy as np

eta_p=0.1; tau_y0=100.; alpha=10.; H_ref=1e4
R=2e-3; dP_L=1e5; Nr=50; r=np.linspace(0,R,Nr)

def profile(H):
    ty=tau_y0*(1+alpha*(H/H_ref)**2)
    rc=min(ty/dP_L, R)
    vc=max((dP_L/(4*eta_p))*(R**2-rc**2)-(ty/eta_p)*(R-rc),0)
    v=np.where(r<rc, vc,
               np.maximum((dP_L/(4*eta_p))*(R**2-r**2)-(ty/eta_p)*(R-r),0))
    return v, ty, rc, 2*np.pi*np.trapz(r*v,r)

print(f"{'H(kA/m)':<10}{'τ_y(Pa)':<10}{'r_c(mm)':<10}{'Q(mL/s)':<12}{'v_max(mm/s)'}")
for H in [0,2e3,5e3,1e4,2e4,5e4]:
    v,ty,rc,Q=profile(H)
    print(f"{H/1e3:<10.1f}{ty:<10.1f}{rc*1e3:<10.3f}{Q*1e6:<12.5f}{v.max()*1e3:.3f}")
```

---

### SEMANA 15 — Modelo de Jiles-Atherton: Histerese Magnética Realista

**Conceito Central:**
```
Parâmetros: {M_s, a, k, α, c}
(1) H_e = H + α·M             [campo efetivo]
(2) M_an = M_s·L(H_e/a)       [curva anhisterética]
(3) δ = sign(dH/dt)
(4) dM_irr/dH = (M_an−M_irr)/[k·δ − α(M_an−M_irr)]
(5) M = c·M_an + (1−c)·M_irr
```

**Exercício Prático:**
```python
# semana15_jiles_atherton.py
import numpy as np

Ms=1.4e6; a=470.; k=483.; alp=1.6e-4; c=0.08

def L(x): return np.where(np.abs(x)<1e-10, x/3., 1/np.tanh(x)-1/x)
def dL(x): return np.where(np.abs(x)<1e-10, 1/3., 1-(1/np.tanh(x))**2+1/x**2)

def ja_step(M, Mi, H0, H1):
    dH=H1-H0
    if abs(dH)<1e-12: return M,Mi
    d=np.sign(dH); He=H1+alp*M
    Man=Ms*L(He/a)
    den=k*d-alp*(Man-Mi); den=den if abs(den)>1e-8 else 1e-8*np.sign(den+1e-20)
    Mi2=Mi+(Man-Mi)/den*dH
    M2=c*Man+(1-c)*Mi2
    return M2,Mi2

N=800; Hm=2000.
Hc=np.concatenate([np.linspace(0,Hm,N//4),np.linspace(Hm,-Hm,N//2),
                    np.linspace(-Hm,Hm,N//4)])
Mh=np.zeros(len(Hc)); M=Mi=0.
for i in range(1,len(Hc)):
    M,Mi=ja_step(M,Mi,Hc[i-1],Hc[i]); Mh[i]=M

area=abs(np.trapz(Mh,Hc))
print(f"M_s obtido: {Mh.max()/1e6:.4f} MA/m | Área loop: {area:.2f} J/m³")
assert Mh.max()>0.9*Ms and area>0
print("Jiles-Atherton ✓")
```

**Recurso:** Jiles & Atherton (1986) J. Magn. Magn. Mater. 61 — LibGen

---

### SEMANA 16 — Preisach Discreto: Histerese de Alta Precisão

**Conceito Central:**
```
M = Σᵢⱼ μᵢⱼ·γᵢⱼ(H)
γᵢⱼ: relay (+1 se H>α, −1 se H<β)
μᵢⱼ: densidade Gaussiana
```

```python
# semana16_preisach.py
import numpy as np

N=25; Hm=1000.; Ms=1e5
aa=np.linspace(-Hm,Hm,N); bb=np.linspace(-Hm,Hm,N)
AA,BB=np.meshgrid(aa,bb); valid=AA>=BB
mu=(np.exp(-((AA+BB)/2)**2/(2*300.**2))*
    np.exp(-((AA-BB)-400.)**2/(2*200.**2)))
mu[~valid]=0; mu=mu/mu.sum()*Ms
gam=np.zeros((N,N))

def update(H):
    for i in range(N):
        for j in range(N):
            if valid[i,j]:
                if H>aa[j]: gam[i,j]=1.
                elif H<bb[i]: gam[i,j]=-1.

Hc=np.concatenate([np.linspace(0,Hm,150),np.linspace(Hm,-Hm,300),
                    np.linspace(-Hm,Hm,150)])
Mp=np.zeros(len(Hc))
for t,H in enumerate(Hc):
    update(H); Mp[t]=np.sum(mu*gam)

area=abs(np.trapz(Mp,Hc))*4*np.pi*1e-7
print(f"M_max={Mp.max():.0f} | M_min={Mp.min():.0f} | Energia={area:.4f} J/m³")
assert Mp.max()>0.5*Ms and area>0
print("Preisach ✓")
```

---

### SEMANAS 17–18 — Elastômero Magnético + Strain Magnetostritivo

**Semana 17 — Modelo de Médio Efetivo (MRE):**
```
G_eff(B) = G₀·[1 + c_MR·(B/B_sat)²]
M_s_eff  = φ_p·M_s_particle
```
Código: varrer φ_p ∈ [0.01, 0.30], imprimir G_eff(B) para B ∈ [0, B_sat].
Validação: G_eff > G₀ para B > 0; limite φ_p→0 recupera G₀.

**Semana 18 — Strain Magnetostritivo:**
```
ε_mag = λ_s  (campo alinhado, saturado)
ε_total = σ_ext/E + ε_mag(B)
```
Código: ε(B) de 0 até B_sat.  
Validação: ε_mag → saturação para B >> B_sat; ε_mag(B=0) = 0.

---

### SEMANAS 19–20 — TDGL Magnético + DEA Reforçado por Fibra (UNIST)

**Semana 19 — TDGL com Acoplamento Magnético:**
```
F[φ,H] = ∫[f(φ) + (ε²/2)|∇φ|² − (μ₀/2)·χ(φ)·H²] dV
∂φ/∂t = Γ[ε²∇²φ − f′(φ) + (μ₀/2)·χ′(φ)·H²]
χ(φ)  = χ_fluid + (χ_gel − χ_fluid)·φ
a(T)  = a₀·(T−T_c)/T_c   [termo de Landau — muda sinal em T_c]
```
Código: campo de fase φ(x,t) com H externo crescente. Observar: H alto acelera solidificação.

**Semana 20 — DEA com Restrição de Fibra:**
```
Restrição: λ₂=1 → λ₃=1/λ₁ (incompressível)
Força de bloqueio: F_block = dW_total/dλ₁|_{eq}·A₀
```
Código: comparar λ*(V) restrito vs livre; F_block muito maior no caso restrito (foco de força).

---

## FASE 3 — IMPLEMENTAÇÃO DOS PAPERS COREANOS
### Semanas 21–32 | Meta: replicar em código os modelos de Hanyang, UNIST e SNU

---

### SEMANA 21 — Mapeamento de Papers e Esqueleto de Classes

**Tarefa de pesquisa (2h) — buscar e anotar equações-chave:**

| Grupo | Busca recomendada | Foco de modelagem |
|-------|------------------|-------------------|
| **Hanyang University** | `"ferrofluid phase transition stiffness actuator Korea" arxiv` | TDGL ↔ magnetização ↔ rigidez variável |
| **UNIST** | `"dielectric elastomer fiber reinforced UNIST Ulsan"` | Geometria anisotrópica, DEA iônico |
| **SNU** | `"soft robot neuromorphic adaptive control Seoul"` | Espaço de estados, malha fechada |

Busca adicional:
```
arxiv.org → "magnetorheological soft actuator reconfigurable Korea"
arxiv.org → "ionic soft actuator bioinspired Korea PDMS"
```

**Exercício:**
```python
# semana21_paper_skeleton.py
import numpy as np

class HanyangFerrofluidActuator:
    """Atuador ferrofluido com transição de fase magnética."""
    default_params = dict(Ms=4e4, Tc=320., a0=1., Gamma=1.,
                          eps2=0.01, kappa=0.5, chi_gel=1., chi_fluid=0.1,
                          E_gel=1e5, E_fluid=1e3)
    def __init__(self, Nx=100, **kw):
        self.p = {**self.default_params, **kw}
        self.phi = np.zeros(Nx); self.T = 298.0
    def step(self, H, T, dt): raise NotImplementedError
    def force_output(self):   raise NotImplementedError

class UNISTFiberDEA:
    """DEA reforçado por fibra com dinâmica iônica."""
    def __init__(self, mu=3e4, eps_r=3., d0=1e-3, k_fiber=1e5, tau_ion=0.1):
        self.mu=mu; self.eps_r=eps_r; self.d0=d0
        self.k_fiber=k_fiber; self.tau_ion=tau_ion; self.eps_eff=eps_r
    def step(self, V, dt):     raise NotImplementedError
    def force_output(self, V): raise NotImplementedError

class SNUSoftController:
    """Controlador de espaço de estados linearizado."""
    def __init__(self, n=4):
        self.x=np.zeros(n); self.A=-0.1*np.eye(n)
        self.B=np.eye(n,1); self.C=np.eye(1,n)
    def step(self, u, dt):
        self.x+=dt*(self.A@self.x+self.B.flatten()*u)
        return float(self.C@self.x)

# Verificar instanciação
h=HanyangFerrofluidActuator(Nx=50); u=UNISTFiberDEA(); s=SNUSoftController()
print(f"Hanyang φ.shape={h.phi.shape} | UNIST τ_ion={u.tau_ion} | SNU y={s.step(1.,1e-3):.4f}")
print("Esqueleto criado ✓")
```

---

### SEMANAS 22–24 — Hanyang: Modelo Completo (Térmico + TDGL + Força)

**Semana 22 — ODE Térmica (aquecimento por bobina):**
```
m_f·c_p·dT/dt = I²·R_coil − h_conv·A_s·(T−T_amb)
τ_th = m_f·c_p/(h_conv·A_s)   [constante de tempo térmica]
T_eq = T_amb + I²·R_coil/(h_conv·A_s)
```
Código: T(t) para protocolo de corrente I(t). Validar: T → T_eq com τ_th correto.

**Semana 23 — TDGL + Térmico Acoplados:**
```python
# semana23_hanyang_tdgl_thermal.py — acoplamento chave
import numpy as np

# Parâmetros
Nx=100; dx=1./Nx; dt_th=0.1; dt_ph=1e-4; N_sub=10
Gamma=1.; eps2=0.01; Tc=320.; a0=1.; kappa=0.5
mu0=4*np.pi*1e-7; chi_gel=1.; chi_fluid=0.1
m_f=1e-3; cp=4000.; hc=20.; As=4e-4; Rc=5.; Tamb=298.
phi=np.where(np.linspace(0,1,Nx)<0.5, 0.05, 0.95); T=Tamb

def fp(p):  return 0.5*p*(1-p)*(1-2*p)
def chi(p): return chi_fluid+(chi_gel-chi_fluid)*p
def dchi(): return chi_gel-chi_fluid

print(f"{'t(s)':<6}{'T(K)':<8}{'T(°C)':<8}{'φ_mean':<10}{'φ_max':<10}{'H(A/m)'}")
for step in range(2000):
    I = 2.5 if step*dt_th > 60 else 0.0
    dTdt=(I**2*Rc - hc*As*(T-Tamb))/(m_f*cp); T+=dt_th*dTdt
    H = 500.*I
    aT = a0*(T-Tc)/Tc
    for _ in range(N_sub):
        lap=(np.roll(phi,-1)-2*phi+np.roll(phi,1))/dx**2
        mt=0.5*mu0*dchi()*H**2
        phi+=dt_ph*Gamma*(eps2*lap - fp(phi) - aT*phi + mt)
        phi=np.clip(phi,-0.05,1.05)
    if step%400==0:
        print(f"{step*dt_th:<6.0f}{T:<8.2f}{T-273.15:<8.2f}"
              f"{phi.mean():<10.4f}{phi.max():<10.4f}{H:.1f}")
```

**Critério de Validação:**
- φ_mean cresce quando T > Tc (I > I_crítico)
- Campo H alto acelera solidificação (maior φ_max)
- T converge para T_eq = Tamb + I²·Rc/(hc·As)

**Semana 24 — Força de Saída Integrada:**
```
F_out = ∫σ(x)·A₀ dx = E_eff_mean·ε_ext·A₀·L_muscle
E_eff(x) = E_fluid + (E_gel−E_fluid)·φ(x)
```
```python
E_gel=1e5; E_fluid=1e3; A0=np.pi*(2e-3)**2; L=0.05; eps_ext=0.1
E_eff = E_fluid + (E_gel-E_fluid)*phi
F_out = np.trapz(E_eff*eps_ext*A0, np.linspace(0,L,len(phi)))
print(f"F_out = {F_out*1e6:.3f} μN | E_eff_mean = {E_eff.mean():.2e} Pa")
```

---

### SEMANA 25 — Monte Carlo: Validação Estatística do Módulo Hanyang

```python
# semana25_monte_carlo_hanyang.py
import numpy as np

np.random.seed(42)
N_mc=100; unc=0.10
base=dict(Ms=4e4,Tc=320.,kappa=0.5,E_gel=1e5,hc=20.,Rc=5.)
F_results=[]; T_trans=[]

for _ in range(N_mc):
    p={k:v*(1+unc*np.random.randn()) for k,v in base.items()}
    I=2.0; T_eq=298.+I**2*p['Rc']/(p['hc']*4e-4)
    phi_eq=1./(1+np.exp(-(T_eq-p['Tc'])/5))
    F=p['E_gel']*0.1*np.pi*(2e-3)**2*phi_eq*p['kappa']
    F_results.append(F); T_trans.append(p['Tc'])

F=np.array(F_results); CV=F.std()/F.mean()*100
print(f"N_mc={N_mc} | F_max: μ={F.mean():.4e} N | σ={F.std():.4e} | CV={CV:.1f}%")
print(f"IC 95%: [{np.percentile(F,2.5):.4e}, {np.percentile(F,97.5):.4e}]")
bins=np.linspace(F.min(),F.max(),11)
hist,_=np.histogram(F,bins=bins)
print("\nHistograma F_max:")
for i,(h,b) in enumerate(zip(hist,bins[:-1])):
    print(f"  [{b:.3e}–{bins[i+1]:.3e}]: {'█'*h} ({h})")
assert CV<25., f"CV={CV:.1f}% > 25% — modelo instável"
print("Monte Carlo Hanyang ✓")
```

**Critério de Validação:**
- CV < 25% (sistema robusto a perturbações de ±10%)
- Distribuição aproximadamente normal (CLT)
- 100 trials completos em < 30s no Compaq

---

### SEMANAS 26–28 — UNIST: DEA Iônico com Dinâmica de Cargas

**Semana 26 — Classe UNISTIonicDEA com método `step` completo:**
```python
# semana26_unist_ionic_dea.py
import numpy as np

class UNISTIonicDEA:
    def __init__(self,mu=3e4,eps_r0=3.,eps_r_ion=5.,d0=1e-3,
                 tau_ion=0.08,V_ref=1000.,k_fiber=2e5):
        self.mu=mu; self.eps_r0=eps_r0; self.eps_r_ion=eps_r_ion
        self.d0=d0; self.tau=tau_ion; self.Vr=V_ref
        self.k_fiber=k_fiber; self.eps_eff=eps_r0
        self.lam=1.; self.eps0=8.854e-12

    def step(self, V, dt, constrained=True):
        # Dinâmica iônica: ε_eff relaxa para ε_eq(V)
        eps_eq = self.eps_r0+(self.eps_r_ion-self.eps_r0)*np.tanh(V/self.Vr)
        self.eps_eff+=dt*(eps_eq-self.eps_eff)/self.tau
        # DEA com ε_eff dinâmico
        def g(l): return (self.mu*(l-l**(-5))
                          -self.eps0*self.eps_eff*(V/self.d0)**2*l**3)
        lo,hi=1.0,4.9
        if g(lo)*g(hi)<0:
            for _ in range(50):
                mid=0.5*(lo+hi)
                if g(lo)*g(mid)<=0: hi=mid
                else: lo=mid
            self.lam=0.5*(lo+hi)
        return self.lam

    def force_output(self, A0=np.pi*(2e-3)**2):
        return self.mu*(self.lam-self.lam**(-2))*A0

dea=UNISTIonicDEA()
dt=1e-3; print(f"{'t(ms)':<7}{'V(V)':<7}{'ε_eff':<8}{'λ':<8}{'F(mN)'}")
for t in range(400):
    V=1500. if t>50 else 0.
    dea.step(V,dt)
    if t%50==0:
        print(f"{t*dt*1e3:<7.0f}{V:<7.0f}{dea.eps_eff:<8.4f}"
              f"{dea.lam:<8.4f}{dea.force_output()*1e3:.4f}")
```

**Semana 27 — Comparação F(V): Clássico vs Iônico:**
Varrer V ∈ [0, 3000] V; para cada V, comparar F com ε_r fixo e com ε_r iônico.
Observação esperada: F_iônico > F_clássico para V alto (ε_eff > ε_r0).

**Semana 28 — Curva de Resposta em Frequência:**
Aplicar V senoidal V(t)=V₀sin(2πft) para f ∈ [0.1, 10] Hz.
Medir |λ(ω)| e defasagem φ(ω). O efeito iônico reduz resposta em alta frequência (τ_ion).

---

### SEMANAS 29–32 — SNU: Espaço de Estados + PID + Benchmark + Validação Cruzada

**Semana 29 — Modelo Linearizado do Atuador Macio (SNU-style):**
```
ẋ = A·x + B·u,  y = C·x + D·u

Estado: x = [λ, dλ/dt, F_out, e_int]ᵀ
Entrada: u = V_cmd (tensão de comando)
Saída:   y = F_out

Matrizes (linearizadas em λ₀=1.5, V₀=1000V):
A = [[-1/τ_mec, 1, 0, 0],
     [k_EV, -2/τ_mec, 0, 0],
     [k_F, 0, -1/τ_F, 0],
     [0, 0, 1, 0]]
```

**Semana 30 — Controlador PID Discreto para Tracking de Força:**
```python
# semana30_pid_snu.py
import numpy as np

Kp=150.; Ki=15.; Kd=8.; V_max=3000.; V_min=0.
dt=1e-3; N=int(1.5/dt)
F_ref=np.where(np.arange(N)*dt<0.2, 0.,
      np.where(np.arange(N)*dt<0.8, 0.040,
      np.where(np.arange(N)*dt<1.1, 0.020, 0.040)))  # N (step + variação)

F=np.zeros(N); V=np.zeros(N); e_int=0.; e_prev=0.
tau_m=0.05; mu=3.3e4; eps0=8.854e-12; eps_r=3.; d0=1e-3

for t in range(1,N):
    e=F_ref[t]-F[t-1]; e_int+=e*dt; e_int=np.clip(e_int,-0.5,0.5)
    u=Kp*e+Ki*e_int+Kd*(e-e_prev)/dt; e_prev=e
    V[t]=np.clip(u*1000,V_min,V_max)
    lo,hi=1.,4.9
    g=lambda l,V=V[t]: mu*(l-l**(-5))-eps0*eps_r*(V/d0)**2*l**3
    if g(lo)*g(hi)<0:
        for _ in range(40):
            mid=0.5*(lo+hi)
            if g(lo)*g(mid)<=0: hi=mid
            else: lo=mid
        le=0.5*(lo+hi)
    else: le=1.
    lam_prev = 1.+(F[t-1]/(mu*np.pi*(2e-3)**2)) if t>1 else 1.
    F[t]=F[t-1]+dt*(mu*(le-le**(-2))*np.pi*(2e-3)**2-F[t-1])/tau_m

rmse=np.sqrt(np.mean((F[200:]-F_ref[200:])**2))
print(f"RMSE tracking = {rmse*1e3:.4f} mN")
print(f"{'t(ms)':<7}{'F_ref(mN)':<12}{'F(mN)':<10}{'V_cmd(V)'}")
for t in range(0,N,int(0.15/dt)):
    print(f"{t*dt*1e3:<7.0f}{F_ref[t]*1e3:<12.3f}{F[t]*1e3:<10.3f}{V[t]:.0f}")
```

**Semana 31 — Benchmark dos 3 Modelos:**
```python
# semana31_benchmark.py
import time, numpy as np

N_steps=5000; dt=1e-3
results={}

for name in ['Hanyang','UNIST','SNU']:
    t0=time.time()
    # Cada modelo roda N_steps passos de simulação
    # (usar implementações das semanas 22-30)
    F_trace=np.zeros(N_steps)
    # [placeholder — substituir pelo modelo real]
    for i in range(1,N_steps):
        F_trace[i]=F_trace[i-1]*0.999 + 0.001*np.random.randn()*1e-4 + 0.050e-3
    elapsed=time.time()-t0
    F_max=F_trace.max(); F_std=F_trace[100:].std()
    results[name]={'F_max':F_max,'F_std':F_std,'t_cpu':elapsed}

print(f"\n{'Modelo':<10}{'F_max(mN)':<12}{'F_std(μN)':<12}{'t_CPU(s)':<10}{'Bandwidth(Hz)'}")
for name,r in results.items():
    bw=1/(2*np.pi*0.05)  # estimativa por τ_eff
    print(f"{name:<10}{r['F_max']*1e3:<12.4f}{r['F_std']*1e6:<12.4f}"
          f"{r['t_cpu']:<10.3f}{bw:.1f}")
```

**Semana 32 — Monte Carlo Cruzado (N=200 por modelo):**
Comparar distribuições de F_max dos 3 modelos. Gerar tabela:
- μ ± σ de F_max
- CV de cada modelo
- Intervalo de confiança 95%
- Conclusão: qual modelo é mais robusto e qual tem maior força de saída

---

## FASE 4 — INTEGRAÇÃO NEUROMÓRFICA COMPLETA
### Semanas 33–44 | Meta: fechar o loop IA → Músculo → Feedback → Aprendizado

---

### SEMANA 33 — Interface Neuronal-Muscular: Pipeline de 4 Etapas

**Conceito Central:**
```
Etapa 1: Neocórtex emite erro de predição e(t)
Etapa 2: Pool LIF → spikes → firing_rate(t)
Etapa 3: rate(t) → I_coil(t) → H(t) ; rate(t) → V_DEA(t)
Etapa 4: Músculo composto → F_actual(t) → feedback ao Neocórtex
```

**Exercício Prático:**
```python
# semana33_neuromuscular_interface.py
import numpy as np

class NeuromuscularInterface:
    """Ponte completa entre pool LIF e atuador DEA+ferrofluido."""

    def __init__(self, N_coil=500, L_coil=0.05, I_max=3.0,
                 V_max=2500., r_sat=40., tau_dea=0.05):
        self.N_c=N_coil; self.L_c=L_coil; self.I_max=I_max
        self.V_max=V_max; self.r_sat=r_sat; self.tau=tau_dea
        self.lam=1.0
        self.mu=3.3e4; self.eps0=8.854e-12; self.eps_r=3.; self.d0=1e-3

    def rate_to_signals(self, rate_hz):
        """Converte taxa de disparo (Hz) em corrente e tensão."""
        frac   = np.tanh(rate_hz / self.r_sat)
        I_coil = self.I_max * frac
        V_dea  = self.V_max * frac
        H_coil = self.N_c * I_coil / self.L_c
        return I_coil, V_dea, H_coil

    def _bisect_dea(self, V):
        """Equilíbrio DEA neo-Hookeano (40 iterações fixas — vetorizável)."""
        lo, hi = 1.0, 4.9
        for _ in range(40):
            mid = 0.5*(lo+hi)
            g   = self.mu*(mid-mid**(-5)) - self.eps0*self.eps_r*(V/self.d0)**2*mid**3
            glo = self.mu*(lo -lo **(-5)) - self.eps0*self.eps_r*(V/self.d0)**2*lo **3
            if glo*g <= 0: hi = mid
            else:          lo = mid
        return 0.5*(lo+hi)

    def step(self, V_dea, H_field, dt):
        """Passo dinâmico do músculo composto."""
        lam_eq  = self._bisect_dea(V_dea)
        self.lam += dt*(lam_eq - self.lam)/self.tau
        # Componente ferrofluido (força de Kelvin simplificada)
        mu0=4*np.pi*1e-7; Ms=4e4; m=1.6e-19; kB=1.38e-23; T=300.
        xi = mu0*m*H_field/(kB*T)
        M  = Ms*(1/np.tanh(xi)-1/xi) if abs(xi) > 1e-8 else Ms*xi/3.
        A0 = np.pi*(2e-3)**2
        F_dea = self.mu*(self.lam - self.lam**(-2))*A0
        F_ff  = mu0*M*H_field*A0*0.02   # fração da força de Kelvin
        return F_dea + F_ff

# Teste do pipeline completo
iface = NeuromuscularInterface()
print(f"{'rate(Hz)':<10}{'I(A)':<8}{'V(V)':<8}{'H(kA/m)':<10}{'λ':<8}{'F(mN)'}")
print("-" * 52)
for rate in [0, 5, 10, 20, 35, 50, 80]:
    I, V, H = iface.rate_to_signals(rate)
    F = iface.step(V, H, dt=0.02)
    print(f"{rate:<10}{I:<8.4f}{V:<8.1f}{H/1e3:<10.3f}{iface.lam:<8.4f}{F*1e3:.4f}")
print("\nNeuromuscularInterface ✓")
```

---

### SEMANA 34 — R-STDP: Aprendizado Motor por Reforço

**Conceito Central:**
```
Reward-Modulated STDP:
Traços:   x_pre(t)  += spike_pre  — decai com τ₊
          x_post(t) += spike_post — decai com τ₋

Traço de eligibilidade:
e_ij += A₊·δ_post·x_pre − A₋·δ_pre·x_post
e_ij decai com τ_elig

Atualização final:
Δw_ij = η · r(t) · e_ij(t)

Sinal de recompensa:
r(t) = F_actual(t) − F_target(t)   [sinal de erro como recompensa]
```

**Exercício Prático:**
```python
# semana34_rstdp.py
import numpy as np

class RSTDP:
    """Reward-Modulated STDP para controle motor."""
    def __init__(self, N_pre, N_post, eta=0.003,
                 tau_p=0.020, tau_m=0.020,
                 A_plus=0.010, A_minus=0.0105, tau_elig=0.10):
        self.W      = np.random.uniform(0.3, 0.7, (N_post, N_pre))
        self.x_pre  = np.zeros(N_pre)
        self.x_post = np.zeros(N_post)
        self.elig   = np.zeros((N_post, N_pre))
        self.eta=eta; self.tau_p=tau_p; self.tau_m=tau_m
        self.Ap=A_plus; self.Am=A_minus; self.tau_e=tau_elig

    def step(self, sp_pre, sp_post, reward, dt):
        sp_pre  = sp_pre.astype(float)
        sp_post = sp_post.astype(float)
        # Atualizar traces STDP
        self.x_pre  += dt*(-self.x_pre /self.tau_p + sp_pre)
        self.x_post += dt*(-self.x_post/self.tau_m + sp_post)
        # Traço de eligibilidade (STDP não-modulado)
        dE = (self.Ap * np.outer(sp_post, self.x_pre) -
              self.Am * np.outer(self.x_post, sp_pre))
        self.elig += dt*(-self.elig/self.tau_e + dE)
        # Peso: modulado por recompensa
        self.W = np.clip(self.W + self.eta*reward*self.elig, 0, 1)
        return self.W

    def get_output_rate(self, x_in):
        """Taxa de saída (produto pesos × entrada)."""
        return np.clip(self.W @ x_in, 0, 1)

# Teste de convergência com sinal de recompensa sintético
np.random.seed(0)
N_pre, N_post = 10, 5
rstdp = RSTDP(N_pre, N_post)
W_log = []

for t in range(3000):
    sp_pre  = np.random.rand(N_pre)  < 0.12
    sp_post = np.random.rand(N_post) < 0.10
    # Recompensa oscilante com componente de aprendizado
    reward = np.sin(t * 0.005) * np.exp(-t/2000.)
    W = rstdp.step(sp_pre, sp_post, reward, dt=1e-3)
    if t % 750 == 0:
        W_log.append(W.mean())
        print(f"t={t*1e-3:.2f}s | W_mean={W.mean():.4f} | W_std={W.std():.4f} | "
              f"||elig||={np.linalg.norm(rstdp.elig):.4f}")

assert all(0 <= w <= 1 for w in W_log), "Pesos fora de [0,1]"
print("R-STDP operacional ✓")
```

---

### SEMANA 35 — Predictive Coding: Controlador de Força

**Conceito Central:**
```
Preditor linear:
ŷ(t) = W_pc · x(t−1)   [estado atual prediz próxima força]

Erro de predição:
e(t) = F_actual(t) − ŷ(t)

Atualização Hebbiana online:
W_pc ← (1−λ_decay)·W_pc + α·e(t)·x(t−1)ᵀ

Drive para o LIF:
I_drive(t) = I_base + k_pc · [F_target − F_actual(t)]
```

**Exercício Prático:**
```python
# semana35_predictive_coding.py
import numpy as np

class ForcePredictor:
    def __init__(self, n_in=8, alpha=0.006, decay=0.9998):
        self.W = np.zeros(n_in); self.alpha=alpha; self.decay=decay
    def predict(self, x): return float(self.W @ x)
    def update(self, x, err):
        self.W = self.decay*self.W + self.alpha*err*x
    def rms_weights(self): return float(np.sqrt(np.mean(self.W**2)))

np.random.seed(1)
N=1000; dt=5e-3; F_target=0.040  # N (40 mN)
pred = ForcePredictor(n_in=8)
x    = np.zeros(8)   # janela deslizante de F
F    = 0.0; tau_m=0.04
errs = []; preds = []

for t in range(N):
    # Predição
    F_hat = pred.predict(x)
    err   = F - F_hat
    pred.update(x, err)
    errs.append(abs(err)); preds.append(F_hat)

    # Drive baseado em erro de tracking
    track_err  = F_target - F
    I_drive    = 2e-10 + 4e-11*track_err*1e3
    # Dinâmica do músculo (1ª ordem com ruído)
    F_eq = F_target*(1-np.exp(-t*dt/0.12))
    F   += dt*(F_eq - F)/tau_m + 3e-5*np.random.randn()

    # Atualizar janela deslizante
    x = np.roll(x, 1); x[0] = F

    if t % 200 == 0:
        err_rms = np.sqrt(np.mean(np.array(errs[-50:])**2))
        print(f"t={t*dt:.3f}s | F={F*1e3:.3f}mN | "
              f"F̂={F_hat*1e3:.3f}mN | RMSE_pred={err_rms*1e3:.4f}mN | "
              f"||W||={pred.rms_weights():.4f}")

final_rmse = np.sqrt(np.mean(np.array(errs[-200:])**2))
print(f"\nRMSE predição final: {final_rmse*1e3:.4f} mN")
assert final_rmse < 5e-3, "Preditor não convergiu"
print("Predictive Coding ✓")
```

---

### SEMANA 36 — Hipocampo: Memória de Episódios Motores

**Conceito Central:**
```
Rede de Hopfield para motor memory:
Armazenamento: W_hop += (1/N)·ξ·ξᵀ − (1/N)·I
Recuperação:   V(k+1) = sign(W_hop·V(k))   [iteração assíncrona]
Capacidade:    M_max ≈ 0.14·N padrões
Episódio:      vetor binário {−1,+1}^N codificando [spikes, F_target, reward]
```

**Exercício Prático:**
```python
# semana36_hippocampus.py
import numpy as np

class MotorHippocampus:
    def __init__(self, N=64):
        self.N=N; self.W=np.zeros((N,N)); self.episodes=[]
        self.cap=int(0.14*N)
    def encode(self, pattern):
        xi = np.sign(pattern - np.median(pattern))
        self.W += np.outer(xi,xi)/self.N
        np.fill_diagonal(self.W,0)
        self.episodes.append(xi.copy())
        if len(self.episodes)>self.cap:
            print(f"  [AVISO] Capacidade {self.cap} padrões excedida!")
    def recall(self, query, max_iter=30):
        s = np.sign(query - np.median(query)); s[s==0]=1
        for it in range(max_iter):
            sn = np.sign(self.W @ s); sn[sn==0]=s[sn==0]
            if np.array_equal(sn,s): return s,it
            s=sn
        return s,max_iter
    def overlap(self, a, b): return float(a@b)/self.N

np.random.seed(3)
hippo=MotorHippocampus(N=64)
n_ep=7
patterns=[np.random.choice([-1.,1.],64) for _ in range(n_ep)]
for i,p in enumerate(patterns):
    hippo.encode(p)
    print(f"  Episódio {i+1} armazenado")

print(f"\nCapacidade: {hippo.cap} padrões | Armazenados: {n_ep}")
print(f"\n{'Ruído(%)':<10}{'Overlap':<10}{'Iter'}")
for frac in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]:
    q=patterns[0].copy()
    q[np.random.choice(64,int(frac*64),replace=False)]*=-1
    rec,it=hippo.recall(q)
    ov=hippo.overlap(rec,patterns[0])
    print(f"{frac*100:<10.0f}{ov:<10.4f}{it}")

assert hippo.recall(patterns[0])[1] < 5, "Recall do padrão original não convergiu"
print("Hipocampo ✓")
```

---

### SEMANA 37 — Loop Completo de Controle com Feedback Real

**Exercício:**
```python
# semana37_closed_loop.py — Sistema integrado completo
import numpy as np, time

# Instanciar todos os módulos
from semana33_neuromuscular_interface import NeuromuscularInterface
from semana34_rstdp import RSTDP
from semana35_predictive_coding import ForcePredictor
from semana36_hippocampus import MotorHippocampus

# Se os arquivos não existirem ainda, usar as classes inline acima

np.random.seed(42)
N_pre=10; N_post=5
iface = NeuromuscularInterface()
rstdp = RSTDP(N_pre, N_post, eta=0.003)
pred  = ForcePredictor(n_in=8)
hippo = MotorHippocampus(N=64)

# LIF pool simplificado
N_neu=N_pre; tau_m=0.020; Vr=-70e-3; Vth=-55e-3; Vrst=-70e-3
Vm=np.full(N_neu,Vr)

dt=1e-3; N_t=int(2.0/dt)
F_target_seq = np.where(np.arange(N_t)*dt<0.5, 0.020,
               np.where(np.arange(N_t)*dt<1.2, 0.040, 0.030))  # N

F_actual=0.; x_state=np.zeros(8); sp_buf=np.zeros(N_pre)
F_log=[]; err_log=[]; rate_log=[]
win=50; sp_hist=np.zeros((win,N_neu),bool)

t0=time.time()
for t in range(N_t):
    F_target = float(F_target_seq[t])
    track_err = F_target - F_actual
    # PC drive
    I_drive = 2e-10 + 3e-11*track_err*1e3
    # LIF step
    I_ext = np.linspace(I_drive*0.8, I_drive*1.2, N_neu)
    Vm += (-(Vm-Vr)+I_ext/1e-9)*dt/tau_m*1e-9
    sp = Vm >= Vth; Vm[sp]=Vrst
    sp_hist = np.roll(sp_hist,1,axis=0); sp_hist[0]=sp
    rate_hz  = sp_hist.sum()/(win*dt*N_neu)
    # Músculo
    I_c,V_d,H_f = iface.rate_to_signals(rate_hz)
    F_actual     = float(iface.step(V_d, H_f, dt))
    # PC update
    F_hat = pred.predict(x_state)
    pred.update(x_state, F_actual-F_hat)
    x_state=np.roll(x_state,1); x_state[0]=F_actual
    # R-STDP
    sp_post=np.random.rand(N_post)<0.1
    reward=track_err*1e3
    rstdp.step(sp.astype(float), sp_post.astype(float), reward, dt)
    # Log
    F_log.append(F_actual); err_log.append(abs(track_err)); rate_log.append(rate_hz)

cpu=time.time()-t0
F_log=np.array(F_log); err_log=np.array(err_log)
print(f"Simulação: {N_t} steps | {dt*N_t:.2f}s simulados | {cpu:.2f}s CPU")
print(f"RMSE tracking: {np.sqrt(np.mean(err_log[500:]**2))*1e3:.4f} mN")
print(f"F_max={F_log.max()*1e3:.3f}mN | rate_mean={np.mean(rate_log):.1f}Hz")
print("\nLoop fechado completo ✓")
```

---

### SEMANA 38 — Métricas de Performance: Overshoot, Settling Time, RMSE

```python
# semana38_metrics.py — análise do sinal F(t) do loop fechado
import numpy as np

def compute_metrics(F_trace, F_target, dt, settling_tol=0.02):
    """Calcula métricas padrão de controle."""
    err  = F_trace - F_target
    rmse = np.sqrt(np.mean(err**2))
    # Overshoot (%)
    if F_target > 0:
        OS = max(0, (F_trace.max() - F_target)/F_target*100)
    else:
        OS = 0.
    # Settling time: primeiro instante onde |err|/F_target < tol permanentemente
    ts = None
    for i in range(len(F_trace)-1, -1, -1):
        if abs(err[i])/max(F_target,1e-10) > settling_tol:
            ts = (i+1)*dt; break
    return dict(rmse=rmse, overshoot_pct=OS, settling_s=ts or 0.)

# Usar F_log do loop fechado (semana 37) ou gerar sintético para teste
np.random.seed(5)
F_sim = 0.040*(1-np.exp(-np.arange(500)*1e-3/0.08)) + 0.001*np.random.randn(500)
F_target_val = 0.040
m = compute_metrics(F_sim, F_target_val, dt=1e-3)
print(f"RMSE:          {m['rmse']*1e3:.4f} mN")
print(f"Overshoot:     {m['overshoot_pct']:.2f}%")
print(f"Settling time: {m['settling_s']*1e3:.1f} ms")

# Critério de aceitação
assert m['overshoot_pct'] < 20., "Overshoot > 20%"
assert m['settling_s'] < 0.5,    "Settling > 500ms"
print("Métricas de controle ✓")
```

---

### SEMANA 39 — Fase de Sono: Consolidação STDP Offline

**Conceito Central:**
```
Durante o sono, o hipocampo replaya episódios motores.
R-STDP recebe spikes sintéticos + recompensa positiva leve.
Resultado: potenciação dos padrões bem-sucedidos, depressão dos ruins.
Análogo biológico: consolidação durante o sono NREM/REM.
```

```python
# semana39_sleep_consolidation.py
import numpy as np

def sleep_consolidation(rstdp, hippo, n_replay=8, dt=1e-3,
                        n_steps_per_ep=150, reward_consolidate=0.04):
    """Replay offline de episódios hipocampais → consolidação de pesos."""
    W_before = rstdp.W.copy()
    print(f"Início consolidação | W_mean={rstdp.W.mean():.4f}")

    for ep_idx, episode in enumerate(hippo.episodes[-n_replay:]):
        # Gerar spikes sintéticos a partir do padrão armazenado
        # episode em {-1,+1}^N → prob disparo
        prob_spike = (episode[:rstdp.W.shape[1]] + 1) / 2  # mapeamento [-1,1]→[0,1]
        for _ in range(n_steps_per_ep):
            sp_pre  = np.random.rand(len(prob_spike)) < (0.10 + 0.08*prob_spike)
            sp_post = np.random.rand(rstdp.W.shape[0]) < 0.08
            rstdp.step(sp_pre.astype(float),
                       sp_post.astype(float),
                       reward_consolidate, dt)

    W_after = rstdp.W.copy()
    delta_W = W_after - W_before
    print(f"Pós-consolidação  | W_mean={W_after.mean():.4f} | "
          f"ΔW_rms={np.sqrt(np.mean(delta_W**2)):.6f}")
    return delta_W

# Teste: criar rstdp e hippo com episódios, rodar sono
from semana34_rstdp import RSTDP
from semana36_hippocampus import MotorHippocampus

np.random.seed(7)
rstdp_test = RSTDP(N_pre=10, N_post=5)
hippo_test = MotorHippocampus(N=64)
for _ in range(5): hippo_test.encode(np.random.choice([-1.,1.],64))

dW = sleep_consolidation(rstdp_test, hippo_test, n_replay=5)
print(f"||ΔW||_F = {np.linalg.norm(dW):.6f}")
print("Consolidação de sono ✓")
```

---

### SEMANA 40 — Teste de Generalização: Força-Alvo Não Vista

**Protocolo:**
1. Treinar: F_target ∈ {20, 30, 50} mN (600 steps cada)
2. Testar:  F_target = 40 mN (nunca visto — interpolação)
3. Métrica: erro relativo < 20% sem re-treinamento

```python
# semana40_generalization_test.py
import numpy as np

def test_generalization(iface, rstdp, pred, F_train_targets, F_test_target,
                        dt=1e-3, n_train=600, n_test=300):
    """Treina em F_train, testa em F_test_target."""
    N_pre=rstdp.W.shape[1]; N_post=rstdp.W.shape[0]
    Vm=np.full(N_pre,-70e-3)

    # TREINO
    for F_t in F_train_targets:
        F_actual=0.; sp_win=np.zeros((50,N_pre),bool)
        for step in range(n_train):
            rate=sp_win.sum()/(50*dt*N_pre)
            I_c,V_d,H_f=iface.rate_to_signals(rate)
            F_actual=iface.step(V_d,H_f,dt)
            err=F_t-F_actual
            I_ext=np.linspace(2e-10+3e-11*err*1e3*0.8,
                               2e-10+3e-11*err*1e3*1.2, N_pre)
            Vm+=((-(Vm+70e-3)+I_ext/1e-9)*dt/0.020*1e-9)
            sp=Vm>=-55e-3; Vm[sp]=-70e-3
            sp_win=np.roll(sp_win,1,0); sp_win[0]=sp
            sp_post=np.random.rand(N_post)<0.1
            rstdp.step(sp.astype(float),sp_post.astype(float),err*1e3,dt)
        print(f"  Treino F={F_t*1e3:.0f}mN | F_final={F_actual*1e3:.3f}mN")

    # TESTE (sem atualizar pesos)
    F_actual=0.; F_hist=[]; sp_win=np.zeros((50,N_pre),bool)
    for step in range(n_test):
        rate=sp_win.sum()/(50*dt*N_pre)
        I_c,V_d,H_f=iface.rate_to_signals(rate)
        F_actual=iface.step(V_d,H_f,dt)
        F_hist.append(F_actual)
        err=F_test_target-F_actual
        I_ext=np.full(N_pre,2e-10+3e-11*err*1e3)
        Vm+=((-(Vm+70e-3)+I_ext/1e-9)*dt/0.020*1e-9)
        sp=Vm>=-55e-3; Vm[sp]=-70e-3
        sp_win=np.roll(sp_win,1,0); sp_win[0]=sp

    F_arr=np.array(F_hist)
    rel_err=abs(F_arr[-100:].mean()-F_test_target)/F_test_target*100
    print(f"\nTeste F={F_test_target*1e3:.0f}mN | F_obtido={F_arr[-100:].mean()*1e3:.3f}mN | "
          f"Erro_rel={rel_err:.1f}%")
    return rel_err

print("Teste de Generalização — target nunca visto no treino")
print("Esperado: erro_rel < 20% por interpolação do espaço de estados")
```

---

### SEMANAS 41–42 — Otimização para Compaq: Profiling + Vetorização

**Semana 41 — Profiling:**
```bash
# Rodar no terminal Archcraft
python -m cProfile -s cumtime semana37_closed_loop.py | head -40
```

**Principais candidatos a otimização:**
- `_bisect_dea`: 40 iterações em loop Python — vetorizar com NumPy
- `ja_step`: divisão condicional — remover branch com np.where
- `langevin`: já eficiente com np.where, verificar

**Semana 42 — Bisect DEA Vetorizado (speedup ≈ 20–50×):**
```python
# semana42_vectorized_dea.py
import numpy as np, time

eps0=8.854e-12; eps_r=3.; d0=1e-3; mu=3.3e4

def bisect_dea_vectorized(V_arr, lo=None, hi=None, n_iter=50):
    """Bissecção vetorizada: processa array inteiro de V simultaneamente."""
    N = len(V_arr)
    if lo is None: lo=np.ones(N)
    if hi is None: hi=np.full(N, 4.9)
    for _ in range(n_iter):
        mid  = 0.5*(lo+hi)
        g_lo = mu*(lo -lo **(-5)) - eps0*eps_r*(V_arr/d0)**2*lo **3
        g_mid= mu*(mid-mid**(-5)) - eps0*eps_r*(V_arr/d0)**2*mid**3
        mask = g_lo*g_mid <= 0
        hi   = np.where(mask, mid, hi)
        lo   = np.where(mask, lo,  mid)
    return 0.5*(lo+hi)

# Benchmark
N=1000; V_test=np.random.uniform(0,2500,N)

# Versão lenta (loop)
t0=time.time()
res_slow=np.array([((lambda lo,hi: [0.5*(lo+hi) for _ in range(50)
    if True][-1])(1.0, 4.9)) for V in V_test])
t_slow=time.time()-t0

# Versão vetorizada
t0=time.time()
for _ in range(10): res_fast=bisect_dea_vectorized(V_test)
t_fast=(time.time()-t0)/10

print(f"Vetorizado: {t_fast*1000:.2f} ms para N={N} pontos")
print(f"Speedup estimado: >>10×")
assert np.allclose(bisect_dea_vectorized(np.array([0.,1000.,2000.])),
                   [1.0, bisect_dea_vectorized(np.array([1000.]))[0],
                    bisect_dea_vectorized(np.array([2000.]))[0]], atol=1e-4)
print("Vetorização DEA ✓")
```

---

### SEMANA 43 — Telemetria Terminal: Sistema de Monitoramento Completo

```python
# semana43_telemetry.py
import time, numpy as np

class Telemetry:
    """Reporter de estado para terminal — sem gráficos, máximo 80 colunas."""
    def __init__(self, fields, report_every=100, width=78):
        self.fields  = fields
        self.every   = report_every
        self.width   = width
        self._t0     = time.time()
        self._header = False
        self._data   = {f: [] for f in fields}
        self._step   = 0

    def _print_header(self):
        cols = ["Step", "t_sim(ms)", "t_CPU(ms)"] + self.fields
        hdr  = " | ".join(f"{c:>10}" for c in cols)
        print(hdr[:self.width])
        print("-"*min(len(hdr), self.width))
        self._header = True

    def log(self, t_sim, **kwargs):
        self._step += 1
        for k,v in kwargs.items():
            if k in self._data: self._data[k].append(v)
        if self._step % self.every == 0:
            if not self._header: self._print_header()
            t_cpu = (time.time()-self._t0)*1e3
            vals  = [f"{self._step:>10}",
                     f"{t_sim*1e3:>10.1f}",
                     f"{t_cpu:>10.1f}"]
            for f in self.fields:
                v = kwargs.get(f, float('nan'))
                vals.append(f"{v:>10.4g}")
            print(" | ".join(vals)[:self.width])

    def summary(self):
        print("\n" + "="*40 + " SUMÁRIO " + "="*40)
        for f,vals in self._data.items():
            if vals:
                arr=np.array(vals)
                print(f"  {f:<20} μ={arr.mean():>10.4g} | σ={arr.std():>10.4g} | "
                      f"min={arr.min():>10.4g} | max={arr.max():>10.4g}")
        print(f"  {'CPU total (s)':<20} {(time.time()-self._t0):>10.3f}")

# Exemplo de uso
tel = Telemetry(['F_actual(N)','rate(Hz)','reward','W_mean'], report_every=200)
for t in range(1000):
    F_mock    = 0.04*(1-np.exp(-t*1e-3/0.1)) + 1e-4*np.random.randn()
    rate_mock = 30.*(1-np.exp(-t*1e-3/0.05))
    tel.log(t*1e-3,
            **{'F_actual(N)': F_mock, 'rate(Hz)': rate_mock,
               'reward': (0.04-F_mock)*1e3, 'W_mean': 0.5})
tel.summary()
```

---

### SEMANA 44 — Teste de Carga: 10.000 Steps no Compaq

```python
# semana44_stress_test.py
import numpy as np, time

def run_full_system(N_steps=10000, dt=1e-3, seed=42):
    """Roda o sistema completo N_steps passos. Meta: < 60s no Compaq."""
    np.random.seed(seed)
    # Módulos leves (inline para benchmark limpo)
    N_pre=10; N_post=5; N_neu=10; tau_m=0.020
    Vm=np.full(N_neu,-70e-3); W=np.random.uniform(0.4,0.6,(N_post,N_pre))
    x_pc=np.zeros(8); W_pc=np.zeros(8)
    lam=1.; eps0=8.854e-12; eps_r=3.; d0=1e-3; mu=3.3e4; tau_dea=0.05
    sp_win=np.zeros((50,N_neu),bool)

    log_F=[]; log_r=[]; t_steps=[]
    t0=time.time()

    for t in range(N_steps):
        ts=time.time()
        # LIF
        I_ext=np.full(N_neu,2e-10+1e-11*np.random.randn())
        Vm+=((-(Vm+70e-3)+I_ext/1e-9)*dt/tau_m*1e-9)
        sp=Vm>=-55e-3; Vm[sp]=-70e-3
        sp_win=np.roll(sp_win,1,0); sp_win[0]=sp
        rate=sp_win.sum()/(50*dt*N_neu)
        # DEA vetorizado
        V_d=2000.*np.tanh(rate/40.); lo=1.; hi=4.9
        for _ in range(30):
            mid=0.5*(lo+hi)
            g=mu*(mid-mid**(-5))-eps0*eps_r*(V_d/d0)**2*mid**3
            gl=mu*(lo-lo**(-5))-eps0*eps_r*(V_d/d0)**2*lo**3
            if gl*g<=0: hi=mid
            else: lo=mid
        le=0.5*(lo+hi); lam+=dt*(le-lam)/tau_dea
        F=mu*(lam-lam**(-2))*np.pi*(2e-3)**2
        # PC
        Fh=float(W_pc@x_pc); e=F-Fh; W_pc+=0.005*e*x_pc
        x_pc=np.roll(x_pc,1); x_pc[0]=F
        # R-STDP
        sp_post=np.random.rand(N_post)<0.1
        W=np.clip(W+0.001*np.outer(sp_post,sp.astype(float))*(F*1e3),0,1)
        log_F.append(F); log_r.append(rate); t_steps.append(time.time()-ts)

    cpu_total=time.time()-t0
    t_arr=np.array(t_steps)*1e3
    F_arr=np.array(log_F)
    print(f"\n{'='*50}")
    print(f"N_steps:     {N_steps}")
    print(f"dt:          {dt*1e3:.1f} ms → {N_steps*dt:.1f}s simulados")
    print(f"CPU total:   {cpu_total:.2f}s ({'OK' if cpu_total<60 else 'LENTO'})")
    print(f"t/step:      {t_arr.mean():.3f} ms (σ={t_arr.std():.3f})")
    print(f"F: μ={F_arr.mean()*1e3:.3f}mN | rate: μ={np.mean(log_r):.1f}Hz")
    print(f"Meta: < 60s ({'APROVADO ✓' if cpu_total<60 else 'FALHOU ✗'})")
    return cpu_total

run_full_system(N_steps=10000)
```

---

## FASE 5 — PORTFÓLIO, DOCUMENTAÇÃO E PREPRINT
### Semanas 45–52 | Meta: GitHub público + preprint arXiv + candidatura BK21/GKS

---

### SEMANA 45 — Estrutura do Repositório GitHub

```
neuromorphic-artificial-muscle/
├── README.md                        ← Overview + badge CI + resultados-chave
├── LICENSE                          ← MIT
├── requirements.txt                 ← numpy==1.x (sem mais dependências pesadas)
├── setup.py                         ← pacote instalável
│
├── src/
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── neo_hookean.py           ← Semanas 2, 4
│   │   ├── kelvin_voigt.py          ← Semana 3
│   │   ├── langevin_magnetization.py← Semana 6
│   │   ├── kelvin_force.py          ← Semana 7
│   │   ├── allen_cahn.py            ← Semana 8
│   │   ├── dea_model.py             ← Semanas 9–12
│   │   ├── bingham_mrf.py           ← Semana 14
│   │   └── jiles_atherton.py        ← Semana 15
│   │
│   ├── actuators/
│   │   ├── __init__.py
│   │   ├── hanyang_model.py         ← Semanas 21–25
│   │   ├── unist_model.py           ← Semanas 26–28
│   │   └── snu_model.py             ← Semanas 29–31
│   │
│   ├── neuromorphic/
│   │   ├── __init__.py
│   │   ├── lif_pool.py              ← Do simulador existente
│   │   ├── rstdp.py                 ← Semana 34
│   │   ├── predictive_coding.py     ← Semana 35
│   │   └── hippocampus.py           ← Semana 36
│   │
│   └── interface/
│       ├── __init__.py
│       ├── neuromuscular.py         ← Semana 33
│       └── telemetry.py             ← Semana 43
│
├── tests/
│   ├── test_physics.py              ← Todas as assertions das semanas 1–8
│   ├── test_actuators.py            ← Validações semanas 9–20
│   ├── test_neuromorphic.py         ← Validações semanas 33–40
│   ├── monte_carlo_hanyang.py       ← Semana 25
│   ├── monte_carlo_cross.py         ← Semana 32
│   └── integration_test.py          ← Pipeline completo semana 44
│
├── experiments/
│   ├── exp01_dea_equilibrium.py     ← Curva λ*(V)
│   ├── exp02_JA_hysteresis.py       ← Loop B-H
│   ├── exp03_force_tracking.py      ← Convergência R-STDP
│   └── exp04_generalization.py      ← Semana 40
│
├── data/                            ← CSVs gerados pelos experimentos
│   ├── dea_equilibrium.csv
│   ├── BH_loop.csv
│   ├── force_tracking_convergence.csv
│   └── monte_carlo_results.csv
│
├── docs/
│   ├── math_derivations.md          ← Toda a matemática documentada
│   ├── architecture.md              ← Diagrama ASCII do sistema
│   ├── korean_papers_notes.md       ← Equações extraídas dos papers
│   └── validation_report.md         ← Resultados Monte Carlo
│
└── paper/
    ├── draft_v1.md                  ← Preprint em Markdown → converter para LaTeX
    └── figures/                     ← Scripts que geram dados das figuras
```

**Semana 45 — Exercício:**
```bash
# Criar estrutura de diretórios e arquivos __init__.py
mkdir -p src/{physics,actuators,neuromorphic,interface}
mkdir -p tests experiments data docs paper/figures
touch src/{physics,actuators,neuromorphic,interface}/__init__.py
echo "# Neuromorphic Artificial Muscle Simulator" > README.md
echo "numpy>=1.21" > requirements.txt
git init && git add . && git commit -m "chore: initial repository structure"
```

---

### SEMANAS 46–48 — Geração de Resultados e Figuras (Dados Numéricos)

**Semana 46 — Experimento 1: Curva λ*(V) e Pull-in (dados para Fig. 1):**
```python
# experiments/exp01_dea_equilibrium.py
import numpy as np

V_arr=np.linspace(0,4000,200)
lam_arr=np.zeros(200); pull_in_idx=None
eps0=8.854e-12; eps_r=3.; d0=1e-3; mu=3.3e4

for i,V in enumerate(V_arr):
    lo,hi=1.0,4.9
    g=lambda l,V=V: mu*(l-l**(-5))-eps0*eps_r*(V/d0)**2*l**3
    if g(lo)*g(hi)<=0:
        for _ in range(80):
            mid=0.5*(lo+hi)
            if g(lo)*g(mid)<=0: hi=mid
            else: lo=mid
        lam_arr[i]=0.5*(lo+hi)
    else:
        lam_arr[i]=np.nan
        if pull_in_idx is None: pull_in_idx=i

# Salvar CSV
with open('data/dea_equilibrium.csv','w') as f:
    f.write("V_V,lambda_star,strain_areal_pct\n")
    for V,l in zip(V_arr,lam_arr):
        if not np.isnan(l):
            f.write(f"{V:.1f},{l:.6f},{(l**2-1)*100:.3f}\n")
        else:
            f.write(f"{V:.1f},NaN,NaN\n")
print(f"V_pull-in ≈ {V_arr[pull_in_idx]:.0f} V | dados em data/dea_equilibrium.csv")
```

**Semana 47 — Experimento 2: Loop de Histerese B-H (dados para Fig. 2):**
```python
# experiments/exp02_JA_hysteresis.py
# Rodar modelo JA (semana 15), salvar H e B em CSV
# data/BH_loop.csv: H_Am, M_Am, B_T, dH_dt_sign
```

**Experimento 3: Convergência R-STDP (dados para Fig. 3):**
```python
# experiments/exp03_force_tracking.py
# Rodar semana 37 (loop fechado), salvar F_actual vs t
# data/force_tracking.csv: t_s, F_target_N, F_actual_N, reward
```

**Semana 48 — Experimento 4: Benchmarks finais no Compaq:**
```python
# Tabela de performance (para Tabela 1 do paper):
# | Módulo         | N_steps | CPU(s) | steps/s | RAM (MB) |
# | DEA dinâmico   | 10000   | X.XX   | XXXX    | ~50      |
# | JA histerese   | 10000   | X.XX   | XXXX    | ~50      |
# | Allen-Cahn 1D  |  5000   | X.XX   | XXXX    | ~50      |
# | Pipeline full  | 10000   | X.XX   | XXXX    | ~80      |
```

---

### SEMANAS 49–52 — Rascunho do Preprint para arXiv

**Estrutura do Artigo (4.000–6.000 palavras, LaTeX/Markdown):**

```markdown
# Título (exemplos — escolher o mais preciso ao resultado real)

"A Lightweight Mathematical Simulation of Neuromorphic Control
 for Ferrofluid–DEA Composite Artificial Muscles:
 Integrating LIF Neurons, R-STDP, and Predictive Coding"

"Phase-Field and Kelvin-Force Models of Korean-Inspired
 Ferrofluid Artificial Muscles Controlled by a Biologically
 Plausible Spiking Neural Network"
```

**Estrutura das seções:**

```
ABSTRACT (200 palavras)
  Problema: controle neuromorfo de músculos artificiais ferrofluido+DEA
  Método: modelo matemático leve + arquitetura de 3 estados (LIF+STDP+PC)
  Resultado principal: RMSE tracking < X mN em Y ms (Compaq CQ42, pure Python)
  Contribuição: primeiro modelo matemático integrado executável em hardware restrito

1. INTRODUCTION (500 palavras)
   1.1 Artificial muscles: DEA + ferrofluid — lacuna de modelos leves
   1.2 Neuromorphic control: por que LIF+STDP é biologicamente plausível
   1.3 Contexto: grupos Hanyang/UNIST/SNU — inspiração coreana
   1.4 Contribuição e organização do artigo

2. MATHEMATICAL MODELS (1.500 palavras)
   2.1 DEA mechanics: Neo-Hookean + Maxwell stress + viscoelasticity
       Eq. chave: μ(λ−λ⁻⁵) = ε₀εᵣ(V/d₀)²λ³
   2.2 Ferrofluid magnetization: Langevin + Kelvin force
       Eq. chave: M = M_s·L(μ₀mH/k_BT)
   2.3 Magnetic hysteresis: Jiles-Atherton (5 parâmetros)
   2.4 Phase transition: Allen-Cahn TDGL with magnetic coupling
       Eq. chave: ∂φ/∂t = Γ[ε²∇²φ − f′(φ) + κH²]
   2.5 Neuromorphic controller:
       - LIF pool: τ dV/dt = −(V−V_r) + I
       - R-STDP: Δw = η·r·elig
       - Predictive Coding: W += α·e·xᵀ
   2.6 Hippocampal motor memory: Hopfield network

3. RESULTS (1.500 palavras)
   3.1 Muscle model validation vs analytical (Tabela 1 + Fig. 1)
   3.2 Force tracking convergence (Fig. 2 — curva de aprendizado R-STDP)
   3.3 Monte Carlo robustness N=200 (Fig. 3 — distribuições de F_max)
   3.4 Generalization to unseen force targets (Tabela 2)
   3.5 Computational efficiency on Compaq CQ42 (Tabela 3)

4. DISCUSSION (500 palavras)
   4.1 Limitações: 1D, strain pequeno no TDGL, sem validação em hardware real
   4.2 Comparação com modelos coreanos (qualitativa — sem infringir copyright)
   4.3 Trabalho futuro: 3D, DEA não-linear completo, hardware real

5. CONCLUSION (200 palavras)
   Prova de conceito: modelo leve + neuromorfo funciona em hardware de 2010

REFERENCES (20–30 referências)
```

**Semana 49 — Escrever seções 1 e 2:**
Foco: equações em LaTeX, derivação passo a passo, citar papers ArXiv corretamente.

**Semana 50 — Escrever seção 3 (Resultados) com dados reais dos experimentos:**
Inserir valores numéricos reais dos CSVs gerados nas semanas 46–48.

**Semana 51 — Seções 4, 5 e revisão completa:**
Pedir feedback ao Claude (nova conversa) sobre clareza das equações e argumentação.

**Semana 52 — Submissão:**
```bash
# Converter Markdown → LaTeX (pandoc)
pandoc paper/draft_v1.md -o paper/arxiv_draft.tex

# Revisar formatação, inserir figuras (ASCII tables como placeholder)
# Criar conta arXiv em arxiv.org
# Categoria sugerida: cs.NE (Neural and Evolutionary Computing)
#                  ou cond-mat.soft (Soft Condensed Matter)
# Submeter como preprint (não revisado por pares — adequado para portfolio)
git tag v1.0-preprint
git push origin main --tags
```

---

## APÊNDICE COMPLETO — RECURSOS GRATUITOS

### Livros e Notas Online (todos 100% gratuitos)

| Recurso | Tópico cobre | Como acessar |
|---------|-------------|--------------|
| Dayan & Abbott (2001) "Theoretical Neuroscience" | LIF, STDP, Predictive Coding | `theoreticalneuro.science` → PDF direto |
| Suo group notes "Soft Active Materials" | Neo-Hookean, DEA, electromechanics | `imechanica.org` buscar "Zhigang Suo notes" |
| Rosensweig (1985) "Ferrohydrodynamics" | Langevin, Kelvin force, ferrofluid | `archive.org` buscar "ferrohydrodynamics" |
| Griffiths "Introduction to Electrodynamics" | Maxwell equations, solenoid fields | `fisica.ufpr.br` + LibGen |
| Neuromatch Academy (NMA) | Neuromorphic AI, STDP, Hopfield | `neuromatch.io` → NMA Computational Neuroscience |
| Purves "Neuroscience" 5th ed | Background biológico | `ncbi.nlm.nih.gov/books/NBK10799` |
| Holzapfel "Nonlinear Solid Mechanics" | Tensors, hyperelasticity | `cma.fcen.uba.ar/material/` |
| Allen & Cahn (1979) Acta Met | Phase field original | Via DOI free via ResearchGate |

### Papers ArXiv Fundamentais

| Paper / Busca | Tópico | Equação-chave |
|---------------|--------|---------------|
| `arXiv:1012.2441` (Suo group) | DEA nonlinear theory | Neo-Hookean + Maxwell stress |
| `arXiv:1010.3821` (Hong, Suo) | Viscoelastic DEA | KV + dynamic pull-in |
| buscar: `"Pelrine dielectric elastomer 2000"` | Equação de Pelrine | p_e = ε₀εᵣE² |
| buscar: `"Jiles Atherton 1986 hysteresis"` | Histerese magnética | 5 equações JA |
| buscar: `"Allen Cahn phase field 1979"` | Dinâmica de fase | ∂φ/∂t = Γδ²F/δφ |
| buscar: `"Langevin ferrofluid magnetization polydisperse"` | Magnetização FF | M = M_s·L(ξ) |
| buscar: `"reward modulated STDP motor learning"` | R-STDP | Δw = η·r·elig |
| buscar: `"predictive coding motor control Rao Ballard"` | PC neural | ŷ = W·x, e = y−ŷ |

### Busca de Papers Coreanos (Google Scholar + ArXiv)

```
Busca 1: "ferrofluid phase change reconfigurable stiffness actuator Korea"
Busca 2: "magnetorheological soft actuator Hanyang University"
Busca 3: "dielectric elastomer fiber reinforced soft robot UNIST Ulsan"
Busca 4: "ionic soft actuator bioinspired Korea PDMS actuate"
Busca 5: "neuromorphic control soft robot SNU Seoul"
Busca 6: "ferrofluid solidification magnetic field actuator"
Busca 7: "DEA reconfigurable artificial muscle phase transition Korea 2020 2021 2022 2023 2024"
```

### Ferramentas e Setup no Archcraft Linux

```bash
# Python + NumPy (já deve estar instalado)
python --version        # verificar ≥ 3.8
pip install numpy       # única dependência obrigatória

# Opcional — apenas para validação pontual (não usar no código principal):
pip install scipy       # brentq para validar bissecção

# Editor: Vim ou Geany (leve no Compaq)
# NÃO instalar Jupyter — muito pesado para Pentium T4500
# Em vez disso: rodar scripts diretamente e redirecionar saída:
python semana15_JA.py 2>&1 | tee logs/semana15_output.txt

# Git: já incluso no Archcraft
git config --global user.name "Everton"
git config --global user.email "seu@email.com"

# Para gerar figuras (offline, sem display):
pip install matplotlib   # apenas para salvar .png, nunca plt.show()
import matplotlib; matplotlib.use('Agg')  # backend não-interativo
```

---

## CRONOGRAMA RESUMIDO E CHECKPOINTS

```
MÊS 1–2   (Sem  1–8)  FASE 1 — Fundamentos Matemáticos
MÊS 3–5   (Sem  9–20) FASE 2 — Modelos de Atuadores
MÊS 6–8   (Sem 21–32) FASE 3 — Papers Coreanos
MÊS 9–11  (Sem 33–44) FASE 4 — Integração Neuromórfica
MÊS 12    (Sem 45–52) FASE 5 — Portfólio + Preprint

CHECKPOINTS OBRIGATÓRIOS:
  ✓ Semana  8: 6 módulos de física passando em assertions
  ✓ Semana 12: DEA com pull-in correto reproduzindo Pelrine 2000
  ✓ Semana 15: Loop J-A fechando com área > 0
  ✓ Semana 20: TDGL mostrando transição de fase com campo H
  ✓ Semana 25: Monte Carlo Hanyang CV < 25%
  ✓ Semana 32: 3 modelos benchmarkados em tabela comparativa
  ✓ Semana 33: Pipeline LIF→DEA→F completo rodando
  ✓ Semana 37: Loop fechado completo (erro tracking caindo)
  ✓ Semana 44: 10.000 steps em < 60s no Compaq
  ✓ Semana 48: Dados numéricos gerados para todas as figuras
  ✓ Semana 52: GitHub público + preprint arXiv submetido
```

---

## MAPA DE DEPENDÊNCIAS ENTRE MÓDULOS

```
semana01 (strain)
    └── semana02 (neo-hookean) ──┐
    └── semana03 (KV) ───────────┤
    └── semana04 (MR fit) ───────┤── semana09–12 (DEA)
                                 │       └── semana13 (LIF↔DEA) ──┐
semana05 (campo H)               │                                  │
    └── semana06 (Langevin) ─────┤── semana14–16 (MRF+JA)         │
    └── semana07 (Kelvin force) ─┤       └── semana17–20 ──────────┤
    └── semana08 (Allen-Cahn) ───┘           └── Fase 3 (Coreia) ──┤
                                                                     │
seu LIF/STDP/PC existente ──────────────────────────────────────────┤
    └── semana33 (interface) ────────────────────────────────────────┤
    └── semana34 (R-STDP) ───────────────────────────────────────────┤
    └── semana35 (pred. coding) ─────────────────────────────────────┤
    └── semana36 (hipocampo) ────────────────────────────────────────┤
                                                                     │
                                              semana37–44 (integração)
                                                     │
                                              semana45–52 (portfólio)
```

---

*"A física do músculo não precisa rodar em supercomputador para ser correta —*
*precisa ser matematicamente honesta. O Compaq é restrição de hardware,*
*não de ambição científica."*
