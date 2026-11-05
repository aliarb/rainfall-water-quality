# Model formulation (Hudson River Park baseline)

This note maps Lin et al. (2024) — *Analysis of pollutant dispersion patterns in rivers under different rainfall based on an integrated water-land model* — onto Hudson River Park, and states the reduced-form model implemented in `scripts/run_baseline.py`.

The PDF is stored locally at `paper/ref/lin-2024-swmm-efdc-pollutant-dispersion.pdf` and is **not** part of this GitHub repository.

## 1. Problem

Urban rainfall connects land and river: runoff and combined-sewer overflows (CSOs) deliver a pulse of water and pollutants that the receiving water then advects, mixes, and degrades. For Hudson River Park the receiving water is the saline Hudson estuary (tidal), not a small inland river. The scientific question is the same as in the paper:

> How do pollutant (or water-quality) signals at a river section change under light / medium / heavy rainfall, and with what lag relative to the rain?

## 2. Full physical model (paper; future work)

Lin et al. couple two EPA-class models:

| Component | Role |
| --- | --- |
| **SWMM** | Land phase: rainfall → infiltration / depression storage → surface runoff → pipe network → outfall load and flow |
| **EFDC** | River phase: 2-D/3-D hydrodynamics + water-quality transport, using SWMM outfalls as lateral boundaries |

### 2.1 SWMM (conceptual)

Each subcatchment is a nonlinear reservoir. Inflow is precipitation (and upstream runoff); outflow is infiltration plus surface runoff (evapotranspiration neglected in the paper). Surface runoff uses Manning’s equation on the excess depth above depression storage. Pipe routing may use kinematic or dynamic wave (the paper uses a small time step, ≤ 1 min, for complex networks).

Water quality in SWMM is typically a washoff / first-order form. A compact statement of washoff load from an impervious surface is

\[
W(t) = C_1 \, q(t)^{C_2} \, B(t)
\]

where \(q\) is runoff rate, \(B\) is remaining buildup, and \(C_1, C_2\) are calibrated. Buildup between storms is often exponential or power-law in dry time.

### 2.2 EFDC transport

For a constituent concentration \(C\) in a shallow, vertically mixed reach (the paper uses a single vertical layer), the governing balance is advection–diffusion plus decay and lateral inflows (CSO / storm outfalls):

\[
\frac{\partial (hC)}{\partial t}
+ \nabla\cdot(h\mathbf{u}C)
= \nabla\cdot(h\mathbf{E}\nabla C)
- khC
+ s_{\text{CSO}}
+ s_{\text{runoff}}
\]

Lateral diffusion in the paper is parameterized as

\[
E_z = \alpha \, h \, u_x \qquad (0.4 \le \alpha \le 0.8;\ \alpha=0.6 \text{ used})
\]

which is Eq. (6) of Lin et al. This is implemented in `rainfall_wq.metrics.lateral_diffusion_ez` for later use once Hudson depth and a flow-rate scale are available. It is **not** identified from Pier 84 data alone.

### 2.3 Skill scores (Eqs. 1–5)

Let \(O_t\) be observed and \(M_t\) modeled:

\[
E_{\max}=\max_i \frac{|M_i-O_i|}{(M_i+O_i)/2}\times 100\%,\quad
E_{\min}=\min_i \frac{|M_i-O_i|}{(M_i+O_i)/2}\times 100\%
\]

\[
R^2=\left(
\frac{\sum (O_t-\bar O)(M_t-\bar M)}
{\sqrt{\sum(O_t-\bar O)^2}\sqrt{\sum(M_t-\bar M)^2}}
\right)^2
\]

\[
\mathrm{NSE}=1-\frac{\sum(O_t-M_t)^2}{\sum(O_t-\bar O)^2}
\]

These are implemented in `src/rainfall_wq/metrics.py`.

### 2.4 Rainfall classes (Table 4)

Daily rainfall \(P\) (mm): sunny \(0\); light \(0.1\)–\(9.9\); medium \(10\)–\(24.9\); heavy \(25\)–\(49.9\); rainstorm \(50\)–\(99.9\).

NYC CSO activation is often cited near **0.10 in (2.5 mm)**. That threshold is stored as `CSO_RAIN_THRESHOLD_MM` and used as a wet-weather / CSO-proxy flag. It is not a hydraulic model of a specific outfall.

## 3. Why not SWMM–EFDC on day one

A faithful SWMM–EFDC application for Hudson River Park needs:

- Combined-sewer / MS4 delineation and outfall locations west of Manhattan (DEP / LTCP GIS)
- Subcatchment imperviousness and slopes
- Hudson bathymetry and tidal open-boundary forcing
- Event sampling of NH3-N, TP, TN, COD (the paper’s constituents) — the Pier 84 sonde measures **temperature, specific conductance, salinity, pH, dissolved oxygen, turbidity**, not nutrients or bacteria

Those layers are in-scope for later work. NYC DEP Harbor Survey stations **N4** (W. 42nd St) and **N5** (Pier A) supply fecal coliform, enterococci, nutrients, and TSS at roughly monthly frequency and are already fetched.

## 4. Reduced-form baseline (this repository)

Treat rainfall as an observed land-side forcing and the Pier 84 sonde as the river-side response. This is the same information flow as “SWMM output → EFDC boundary”, with both ends replaced by measurements.

### 4.1 Impulse-response (transfer function)

For a water-quality series \(Y_t\) (e.g. turbidity),

\[
Y_t = \beta_0 + \beta_r \underbrace{\sum_{k=0}^{K} w_k(\tau)\, P_{t-k}}_{K_\tau(P)_t}
+ \gamma_1 \sin(2\pi t/365.25)
+ \gamma_2 \cos(2\pi t/365.25)
+ \delta S_t + \varepsilon_t
\]

with exponential weights \(w_k(\tau)\propto e^{-k/\tau}\), \(\sum_k w_k=1\). \(S_t\) is salinity (estuarine mixing / tide-range proxy) and is omitted when \(Y\) is salinity. \(\tau\) is chosen on a small grid by **validation NSE** (last 20% of the record). Seasonality captures the strong annual cycle in temperature and DO that rainfall does not explain.

Interpretation:

- \(\beta_r > 0\) for turbidity is consistent with sediment and CSO solids after rain
- \(\beta_r < 0\) for DO is consistent with an oxygen sag after wet-weather loads
- \(\tau\) is an effective basin + estuary lag (the paper found ~2 days on Shili River under light rain; the Hudson at Pier 84 is tidal and CSO outfalls are much closer, so \(\tau\) may be < 1–2 days in daily data)

This is a linear convolution, i.e. a discrete unit hydrograph from rain to water quality. It does **not** resolve individual CSO outfalls.

### 4.2 Event diagnostics

Wet days (\(P \ge 0.1\,\mathrm{mm}\)) are classified with Table 4. We report:

- Median turbidity and DO by class
- Cross-correlation of turbidity with rainfall at lags 0–7 days
- Same-day vs +1/+2/+3 day anomalies after rain (paper-style lag)

### 4.3 Targets that stand in for CSO

| Signal | Why it is a CSO / wet-weather proxy at Pier 84 |
| --- | --- |
| Turbidity | Solids from runoff and CSO; high-frequency |
| Dissolved oxygen | Oxygen sag after sewage / BOD pulse |
| Salinity / conductance | Freshwater dilution from runoff and CSO |
| Harbor Survey enterococci / fecal coliform at N4, N5 | Direct sewage indicator (sparse in time) |
| Sentinel-2 NDTI, hue / Forel-Ule, Nechad turbidity | Independent channel colour on cloud-free overpasses (not a CSO clock) |

### 4.4 Sentinel-2 water colour

The Pier 84 sonde is on the bulkhead. A 10 m Sentinel-2 pixel there mixes dock, buildings, and water, so optical samples are taken in a small mid-channel box west of the pier (`RS_BBOX` in `config.py`). Microsoft Planetary Computer serves Sentinel-2 L2A without a token. For each scene we keep the median of water pixels (SCL class 6, plus unclassified/dark pixels with MNDWI $> 0.15$).

Indices (clear-sky only):

- NDTI \(=( \rho_{\mathrm{red}} - \rho_{\mathrm{green}} ) / ( \rho_{\mathrm{red}} + \rho_{\mathrm{green}} )\), a turbidity colour index
- Hue angle and Forel-Ule class from RGB reflectance (van der Woerd / Wernand)
- Nechad (2010) red-band turbidity proxy, using Sen2Cor BOA reflectance as a stand-in for water-leaving reflectance
- NDCI from the red-edge band, a chlorophyll proxy

Optical overpasses miss most CSO hours because rain days are cloudy. The series is used as (i) a matchup check against USGS turbidity within 1 h of overpass, (ii) rain-class medians on the cloud-free subset, and (iii) an extra covariate in the daily rain-kernel model of sonde turbidity. It is not used in the hourly DLM/GAM.

## 5. Wave 1–2 implementation (`scripts/run_wave1.py`)

Hourly Pier 84 series are decomposed as

\[
Y_t = \beta_0 + \sum_k \bigl(a_k\sin(2\pi t/T_k)+b_k\cos(2\pi t/T_k)\bigr) + \varepsilon_t
\]

with \(T_k\) for M2, S2, N2, K1, O1, M4, S1, Sa, Ssa. Rain analysis uses \(\varepsilon_t\).

Storms are clusters of wet hours (\(P\ge 0.1\,\mathrm{mm}\)) separated by \(\ge 6\,\mathrm{h}\) dry. CSO-proxy events have depth \(\ge 2.54\,\mathrm{mm}\). Superposed epoch averages \(\varepsilon_t\) from \(-6\) to \(+48\,\mathrm{h}\) of event start. First flush is the median \(M(V)\) curve of excess residual turbidity versus cumulative rain.

Distributed lag and spline GAM add rain to the harmonic mean:

\[
Y_t = \text{harmonics}(t) + \sum_{k=0}^{12}\beta_k P_{t-k} + e_t
\]

\[
Y_t = \text{harmonics}(t) + f_0(P_t)+f_3(P_{t}^{3\mathrm{h}})+f_{24}(P_{t}^{24\mathrm{h}}) + e_t
\]

Skill is NSE / R² on a later-time hold-out versus harmonics alone.

Harbor Survey counts at N4/N5 use a negative-binomial GLM, with sunny as the rain-class reference:

\[
\log\mathbb{E}[Y]=\beta_0+\beta_r P_{48\mathrm{h}}+\beta_d D_{\mathrm{dry}}
\]

and a separate class specification \(\log\mathbb{E}[Y]=\beta_0+\sum_c \beta_c \mathbf{1}_{\text{class}=c}+\beta_d D_{\mathrm{dry}}\).

## 6. Sentinel-2 evaluation (`scripts/run_remote_sensing.py`)

`scripts/fetch_remote_sensing.py` writes one row per usable L2A scene. Evaluation then:

1. Matches mid-channel NDTI and Nechad turbidity to Pier 84 turbidity within 60 min.
2. Reports medians by Lin et al. rainfall class (biased toward clear days).
3. Fits the same rain-kernel + seasonal model as the baseline, with satellite NDTI as \(Y\) and, on matchup days, as a covariate for USGS turbidity.

## 7. Next physical steps

1. DEP CSO outfall GIS + LTCP event reports as a binary overflow covariate.
2. SWMM of the west-side combined districts once pipe data are available; EFDC or a simpler 1-D estuary box as the receiving model.
3. Expand beyond Hudson River Park using Harbor Survey Inner Harbor stations (N1, N3B, N6, …).
4. Wet-weather chemistry that is not in the USGS sonde or Harbor Survey grabs: tire-wear 6PPD-quinone, microplastics, and petroleum hydrocarbons / PAHs. Those analytes require storm-targeted laboratory samples; they cannot be inferred from turbidity or fecal counts. The *Urban Water Journal* draft in `paper/main.tex` states this as future work.
