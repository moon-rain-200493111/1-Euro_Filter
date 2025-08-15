import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, FloatText, HBox, VBox, interactive_output

# ===== 1 Euro Filter =====
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

# ===== 模擬資料 =====
np.random.seed(0)
t = np.linspace(0, 5, 300)
signal_clean = np.sin(2 * np.pi * 0.5 * t)
noise = np.random.normal(0, 0.1, len(t))
signal_noisy = signal_clean + noise

# ===== 繪圖更新函數 =====
def update_plot(freq, min_cutoff, beta):
    filter_obj = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)
    filtered_signal = np.array([filter_obj.filter(x) for x in signal_noisy])
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal_noisy, label="Noisy Signal", alpha=0.5)
    plt.plot(t, signal_clean, label="Ground Truth", linestyle="--")
    plt.plot(t, filtered_signal, label="1 Euro Filter Output", linewidth=2)
    plt.title("1 Euro Filter Demo")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===== 滑桿 & 輸入框設定 =====
slider_freq = FloatSlider(min=1, max=120, step=1, value=60, description='Freq', readout=False)
text_freq   = FloatText(value=60, layout={'width': '70px'})
slider_min_cutoff = FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0, description='Min Cutoff', readout=False)
text_cutoff = FloatText(value=1.0, layout={'width': '70px'})
slider_beta = FloatSlider(min=0.0, max=1.0, step=0.01, value=0.01, description='Beta', readout=False)
text_beta   = FloatText(value=0.01, layout={'width': '70px'})

# 滑桿與輸入框互相同步
def link_widgets(slider, textbox):
    def on_slider_change(change):
        textbox.value = change['new']
    def on_text_change(change):
        slider.value = change['new']
    slider.observe(on_slider_change, names='value')
    textbox.observe(on_text_change, names='value')
link_widgets(slider_freq, text_freq)
link_widgets(slider_min_cutoff, text_cutoff)
link_widgets(slider_beta, text_beta)

# ===== 建立互動輸出 =====
ui = VBox([
    HBox([slider_freq, text_freq]),
    HBox([slider_min_cutoff, text_cutoff]),
    HBox([slider_beta, text_beta])
])
out = interactive_output(update_plot, {
    'freq': slider_freq,
    'min_cutoff': slider_min_cutoff,
    'beta': slider_beta
})

display(ui, out)
