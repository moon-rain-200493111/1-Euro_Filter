import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

# ----- 1 Euro Filter -----
class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.set_params(freq, min_cutoff, beta, d_cutoff)
        self.reset()

    def set_params(self, freq, min_cutoff, beta, d_cutoff):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0

    def alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

# ----- 模擬資料 -----
np.random.seed(0)
t = np.linspace(0, 5, 300)
signal_clean = np.sin(2 * np.pi * 0.5 * t)
noise = np.random.normal(0, 0.1, len(t))
signal_noisy = signal_clean + noise

# ----- 初始化參數 -----
init_freq = 60
init_min_cutoff = 1.0
init_beta = 0.01
filter_obj = OneEuroFilter(freq=init_freq, min_cutoff=init_min_cutoff, beta=init_beta)
filtered_signal = np.array([filter_obj.filter(x) for x in signal_noisy])

# ===== 🎯 正確配置圖形，預留空間給滑桿區 =====
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.4)  # 預留底部空間給滑桿與 TextBox

# ----- 主圖繪製 -----
l1, = ax.plot(t, signal_noisy, label="Noisy Signal", alpha=0.5)
l2, = ax.plot(t, signal_clean, label="Ground Truth", linestyle="--")
l3, = ax.plot(t, filtered_signal, label="1 Euro Filter Output", linewidth=2)
ax.set_title("1 Euro Filter Demo")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Signal Value")
ax.legend()
ax.grid(True)

# ===== ✅ 滑桿與輸入框區域都設在圖底部 (y < 0.35) =====
ax_freq = fig.add_axes([0.15, 0.25, 0.6, 0.03])
ax_min_cutoff = fig.add_axes([0.15, 0.18, 0.6, 0.03])
ax_beta = fig.add_axes([0.15, 0.11, 0.6, 0.03])

slider_freq = Slider(ax_freq, 'Freq', 1, 120, valinit=init_freq)
slider_min_cutoff = Slider(ax_min_cutoff, 'Min Cutoff', 0.1, 5.0, valinit=init_min_cutoff)
slider_beta = Slider(ax_beta, 'Beta', 0.0, 1.0, valinit=init_beta)

# TextBox：對應滑桿右側
ax_box_freq = fig.add_axes([0.83, 0.25, 0.045, 0.03])
ax_box_cutoff = fig.add_axes([0.83, 0.18, 0.045, 0.03])
ax_box_beta = fig.add_axes([0.83, 0.11, 0.045, 0.03])
text_freq = TextBox(ax_box_freq, '', initial=str(init_freq))
text_freq.text_alignment = "center"
text_cutoff = TextBox(ax_box_cutoff, '', initial=str(init_min_cutoff))
text_cutoff.text_alignment = "center"
text_beta = TextBox(ax_box_beta, '', initial=str(init_beta))
text_beta.text_alignment = "center"

# ----- 更新圖形 -----
def update(val):
    new_freq = slider_freq.val
    new_min_cutoff = slider_min_cutoff.val
    new_beta = slider_beta.val
    filter_obj.set_params(freq=new_freq, min_cutoff=new_min_cutoff, beta=new_beta, d_cutoff=1.0)
    filter_obj.reset()
    new_filtered = np.array([filter_obj.filter(x) for x in signal_noisy])
    l3.set_ydata(new_filtered)
    fig.canvas.draw_idle()

# ----- 處理文字輸入 -----
def submit_freq(text):
    try: slider_freq.set_val(float(text))
    except: pass
def submit_cutoff(text):
    try: slider_min_cutoff.set_val(float(text))
    except: pass
def submit_beta(text):
    try: slider_beta.set_val(float(text))
    except: pass

slider_freq.on_changed(update)
slider_min_cutoff.on_changed(update)
slider_beta.on_changed(update)

text_freq.on_submit(submit_freq)
text_cutoff.on_submit(submit_cutoff)
text_beta.on_submit(submit_beta)

plt.show()
