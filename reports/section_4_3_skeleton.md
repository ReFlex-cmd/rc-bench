# Раздел 4.3 «Реализованные модели» — скелет для ВКР

Структура для каждой модели: 1) название класса в коде, 2) математическая формулировка вычислительного элемента, 3) ключевые гиперпараметры, 4) использованные численные методы, 5) ссылка на референсную статью. Это **структурированный конспект** для развития автором, не литературный текст.

---

## 4.3.1 ESN (Echo State Network, Jaeger-style)

**Класс:** `ESNReservoir` (`src/rc_bench/core/reservoirs/esn_service.py`).

**Уравнение:**
$$\mathbf{x}(t) = \tanh\bigl(W \cdot \mathbf{x}(t{-}1) + W_{in} \cdot \mathbf{u}(t)\bigr)$$

без leaky integration, без bias-сдвига.

**Гиперпараметры (тюнятся в HPO):**
- `spectral_radius` ∈ [0.1, 1.5] — спектральный радиус $\rho(W)$.
- `input_scaling` ∈ [0.01, 2.0] (log) — масштаб элементов $W_{in}$.
- `rc_connectivity` ∈ [0.01, 0.20] (log) — sparsity $W$.
- `readout_alpha` ∈ [1e-6, 1e+2] (log) — Ridge regularization.

**Фиксированные:**
- `n_units = 300`.
- `lr = 1.0` (классический, без leak).
- `input_connectivity = 1.0` (плотная $W_{in}$, Jaeger-style).

**Численные методы:**
- $W$: разреженная случайная матрица Бернулли × Gaussian, нормированная на спектральный радиус через `np.linalg.eigvals`.
- Reservoir-эволюция: внутри `reservoirpy.nodes.Reservoir` (v0.4.1).
- Readout: `sklearn.linear_model.Ridge` с `fit_intercept=True`.

**Реализация:** обёртка над `reservoirpy.Reservoir`. Версия библиотеки зафиксирована в `RunRecord.lib_versions` для воспроизводимости.

**Референс:** Jaeger H. "The 'echo state' approach to analysing and training recurrent neural networks". GMD Report 148, 2001. + Lukoševičius M. "A practical guide to applying echo state networks". 2012.

---

## 4.3.2 Leaky ESN

**Класс:** `LeakyESNReservoir` (`leaky_esn_service.py`).

**Уравнение:**
$$\mathbf{x}(t) = (1{-}\alpha)\,\mathbf{x}(t{-}1) + \alpha \tanh\bigl(W_{in}\,u(t) + W\,\mathbf{x}(t{-}1)\bigr)$$

При $\alpha = 1$ совпадает с классическим ESN.

**Гиперпараметры:**
- `sr` ∈ [0.1, 1.5] — спектральный радиус.
- `leak_rate` (α) ∈ [0.05, 1.0].
- `input_scaling` ∈ [0.01, 2.0] (log).
- `density` ∈ [0.01, 0.20] (log).
- `readout_alpha` ∈ [1e-6, 1e+2] (log).

**Фиксированные:** `units = 300`.

**Численные методы:** чистый numpy, нормировка через `np.linalg.eigvals`. Преимущество перед reservoirpy-ESN: формулы в коде проекта, видны напрямую.

**Референс:** Jaeger H., Lukoševičius M., Popovici D., Siewert U. "Optimization and applications of echo state networks with leaky-integrator neurons". Neural Networks 20(3), 2007.

---

## 4.3.3 Deep ESN

**Класс:** `DeepESNReservoir` (`deep_esn_service.py`).

**Архитектура:** стек $L$ leaky-ESN слоёв. Слой 0 принимает скаляр $u(t)$. Слой $\ell \geq 1$ принимает состояние слоя $\ell{-}1$ как вход. Выход readout = конкатенация состояний всех слоёв размерности $L \cdot N$.

**Уравнение слоя $\ell$:**
$$\mathbf{x}_\ell(t) = (1{-}\alpha_\ell)\,\mathbf{x}_\ell(t{-}1) + \alpha_\ell \tanh\bigl(W_{in,\ell}\,\mathbf{x}_{\ell-1}(t) + W_\ell\,\mathbf{x}_\ell(t{-}1)\bigr)$$

**Ключевая особенность реализации:** **per-layer гиперпараметры** $\rho_\ell, \alpha_\ell$ независимо тюнятся в HPO (custom suggest, см. `_suggest_deep_esn` в `search_spaces.py`). Это методологически принципиально — общие $\rho, \alpha$ свели бы Deep ESN к более глубокому ESN со связанными параметрами.

**Гиперпараметры:**
- `n_layers` ∈ {2, 3, 4, 5}.
- `sr_l` (для каждого l ∈ [0, n_layers)) ∈ [0.1, 1.5].
- `leak_rate_l` (для каждого l) ∈ [0.05, 1.0].
- `input_scaling`, `density`, `readout_alpha` — общие для всех слоёв.

**Размерность HPO-пространства:** $1 + 2 \cdot n_{layers} + 3 = 8 \ldots 14$ при $n_{layers} \in [2, 5]$.

**Фиксированные:** `units = 100` на слой.

**Референс:** Gallicchio C., Micheli A. "Deep reservoir computing: A critical experimental analysis". Neurocomputing 268, 2017.

---

## 4.3.4 LSM (Liquid State Machine)

**Класс:** `LSMReservoir` (`lsm_service.py`).

**Динамика LIF-нейрона:**
$$v_i(t{+}\Delta t) = \alpha_{mem} \cdot v_i(t) + (1{-}\alpha_{mem}) \cdot I_i(t)$$

где $\alpha_{mem} = \exp(-\Delta t / \tau_{mem})$ — точное аналитическое решение между событиями. При $v_i \geq V_{th}$: эмиссия спайка, $v_i \leftarrow V_{reset}$, рефрактерный период $t_{ref}$.

**Синаптический след (вход readout):**
$$s_i(t{+}\Delta t) = \alpha_{syn} \cdot s_i(t) + \text{spike}_i(t)$$

**Входной ток:**
$$I_i(t) = W_{in,i} \cdot u(t) + (W_{rec} \cdot \mathbf{s}(t))_i$$

**Гиперпараметры (тюнятся):**
- `tau_mem` ∈ [10, 50] ms (log).
- `tau_syn` ∈ [2, 10] ms (log).
- `v_th` ∈ [0.3, 1.0].
- `t_refractory` ∈ [2, 5] ms.
- `w_rec_scale` ∈ [1.0, 20.0] (log).
- `input_scale` ∈ [1.0, 10.0] (log).
- `density` ∈ [0.05, 0.20] (log).
- `readout_alpha` ∈ [1e-6, 1e+2] (log).

**Фиксированные:** `units = 400`, `dt = 1 ms`, `v_reset = 0`.

**Численные методы:** точное интегрирование между спайками через $\exp(-dt/\tau)$. Векторная реализация через numpy с маской `not_refrac`.

**Кодирование входа:** прямая инъекция (direct-current). Альтернативы (rate-coding, Poisson) — направление дальнейшей работы.

**Связь между нейронами:** все $W_{rec}$ берутся из $\mathcal{N}(0, w_{rec\_scale}^2/N)$ — без E/I разделения 80/20 / без Dale's law (упрощение).

**Референс:** Maass W., Natschläger T., Markram H. "Real-time computing without stable states: A new framework for neural computation based on perturbations". Neural Computation 14(11), 2002.

---

## 4.3.5 FHN (FitzHugh-Nagumo network)

**Класс:** `FHNReservoir` (`fhn_service.py`).

**Динамика двух переменных на узел:**
$$\frac{dv_i}{dt} = v_i - \frac{v_i^3}{3} - w_i + I_{ext,i}, \quad \frac{dw_i}{dt} = \varepsilon\,(v_i + a - b\,w_i)$$

**Внешний ток:**
$$I_{ext,i}(t) = W_{in,i} \cdot u(t) + (W_{rec} \cdot \mathbf{v}(t))_i$$

(резистивная связь по быстрой переменной).

**Выход reservoir:** значения $v_i$ (быстрая переменная) на каждом сэмпле.

**Гиперпараметры (тюнятся):**
- `a` ∈ [0.5, 1.0].
- `b` ∈ [0.5, 1.0].
- `epsilon` (ε = 1/τ) ∈ [0.01, 0.5] (log).
- `coupling_strength` (= спектральный радиус $W_{rec}$) ∈ [0.001, 1.0] (log).
- `input_scale` ∈ [0.01, 5.0] (log).
- `density` ∈ [0.01, 0.20] (log).
- `readout_alpha` ∈ [1e-6, 1e+2] (log).

**Фиксированные (методологически принципиально):**
- `dt = 0.01` — **не тюнится**. При больших dt RK4 на $v^3$ численно неустойчив (см. `audit/01_implementations.md` §1.5).
- `internal_steps = 1`.
- `units = 200`.

**Численные методы:** классический Runge-Kutta 4 с фиксированным шагом. Внешний ток $I_{ext}$ вычисляется один раз на основной шаг и удерживается константным во время RK4-подшагов (zero-order hold).

**Дефолтный режим возбудимости:** $a=0.7, b=0.8, \varepsilon=0.08$ — близко к классическим параметрам FitzHugh.

**Референсы:** FitzHugh R. "Impulses and physiological states in theoretical models of nerve membrane". Biophys J. 1(6), 1961. + Nagumo J., Arimoto S., Yoshizawa S. "An active pulse transmission line simulating nerve axon". Proc. IRE 50(10), 1962.

---

## 4.3.6 Logistic Reservoir

**Класс:** `LogisticReservoir` (`logistic_service.py`).

**Уравнение узла (аддитивная форма):**
$$x_i(t{+}1) = r_i \cdot x_i(t)\,(1 - x_i(t)) + \varepsilon \cdot W_{in,i} \cdot u(t)$$

с последующим клипом $x_i \in [0, 1]$.

**Ключевые особенности:**
- $r_i$ **различны** по узлам: $r_i \sim \mathcal{U}(r_{min}, r_{max})$, что обеспечивает computational diversity даже без сетевой связи.
- **Нет рекуррентной связи между узлами** (нет $W_{rec}$). Узлы взаимодействуют только через общий вход $u(t)$. Это упрощённая модель — known limitation (см. `audit/01_implementations.md` §1.6).
- **Аддитивная форма** (не convex combination) — input не вытесняет логистическую динамику.

**Гиперпараметры (тюнятся):**
- `r_min` ∈ [3.7, 3.85].
- `r_max` ∈ [3.9, 3.99].
- `coupling` ($\varepsilon$) ∈ [0.001, 1.0] (log).
- `input_scale` ∈ [0.01, 5.0] (log).
- `readout_alpha` ∈ [1e-6, 1e+2] (log).

**Фиксированные:** `units = 500`.

**Референс:** Fischer A. "Logistic-map reservoir" (нет канонической статьи; используется как baseline в работах по edge of chaos в RC, например Bertschinger & Natschläger 2004).

---

## 4.3.7 QRC (Quantum Reservoir Computing — mean-field surrogate)

**Класс:** `QRCReservoir` (`qrc_service.py`).

**⚠️ Методологический дисклеймер:** реализация — **классическая mean-field Ising-симуляция**, не унитарная квантовая эволюция. Не претендует на воспроизведение экспериментальных результатов quantum RC. См. `audit/notes_qrc.md`.

**Уравнение update (трансверсальное Ising-приближение):**
$$\mathbf{x}(t) = \tanh\bigl(J \cdot \mathbf{x}(t{-}1) + h_{in} \cdot u(t)\bigr)$$

где $\mathbf{x}_i \in [-1, 1]$ — приближение mean-field магнетизации $\langle \sigma_z^i \rangle$, $J$ — симметричная Ising-coupling-матрица с заданным спектральным радиусом, $h_{in}$ — поперечное поле (вход).

**Virtual nodes (Fujii & Nakajima 2017):** после input-driven шага выполняются $\text{depth}{-}1$ шагов свободной эволюции, состояния конкатенируются в признаковый вектор размерности $n_{qubits} \times \text{depth}$.

**Гиперпараметры (тюнятся):**
- `sr` ∈ [0.1, 1.5] — спектральный радиус $J$.
- `coupling` ∈ [0.1, 2.0] (log) — масштаб элементов $J$ перед перенормировкой.
- `depth` ∈ {2, 3, 4, 5, 6}.
- `readout_alpha` ∈ [1e-6, 1e+2] (log).

**Фиксированные:** `n_qubits = 50` (интерпретируется как hidden size, не реальное число кубитов).

**В тексте ВКР:** называть «quantum-inspired reservoir», «mean-field Ising surrogate», или «classical mean-field approximation of QRC». **НЕ называть просто «QRC»** без дисклеймера.

**Референс:** Fujii K., Nakajima K. "Harnessing disordered-ensemble quantum dynamics for machine learning". Phys. Rev. Applied 8, 024030, 2017 — для virtual nodes; собственно полная QRC — направление дальнейшей работы.

---

## 4.3.8 Сводная таблица: сравнение моделей

| Модель | Природа | Память | Сложность за шаг | Хорошо подходит для |
|---|---|---|---|---|
| ESN | Линейная рекуррентная сеть с tanh | средняя | $O(N^2)$ | универсальная NARMA, MG |
| Leaky ESN | то же + low-pass на состоянии | средняя+ | $O(N^2)$ | хаотические / closed-loop |
| Deep ESN | Стек слоёв с разными временными масштабами | высокая | $O(L \cdot N^2)$ | многомасштабные задачи |
| LSM | Спайковая (LIF) с рефрактером | средняя | $O(N^2)$ | биологическое правдоподобие |
| FHN | Continuous-time нелинейная сеть осцилляторов | низкая | $O(k \cdot N^2)$, k — RK4-substeps | excitable dynamics |
| Logistic | Хаотические дискретные узлы (без сетевой связи) | очень низкая | $O(N)$ | edge of chaos |
| QRC* | Mean-field Ising с virtual nodes | средняя | $O(\text{depth} \cdot N^2)$ | (только как baseline для архитектурного исследования) |

\* QRC — упрощённая классическая симуляция, см. дисклеймер выше.
