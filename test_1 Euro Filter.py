import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ----- 1 Euro Filter -----
class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.set_params(freq, min_cutoff, beta, d_cutoff)
        self.reset()

    def set_params(self, freq, min_cutoff, beta, d_cutoff):
        self.freq = freq                # Sampling frequency
        self.min_cutoff = min_cutoff    # Minimum cutoff frequency
        self.beta = beta                # Responsiveness to changes
        self.d_cutoff = d_cutoff        # Cutoff frequency for derivative

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
        
        # Estimate derivative
        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        
        # Adaptive cutoff based on signal speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)
        
        # Filter the signal
        x_hat = a * x + (1 - a) * self.x_prev
        
        # Update previous state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        
        return x_hat

# ----- Streamlit UI -----
st.title("1 Euro Filter Demo")

freq = st.slider("Freq", 1, 60, 24)
min_cutoff = st.slider("Min Cutoff", 0.1, 3.0, 0.2)
beta = st.slider("Beta", 0.0, 0.1, 0.02)

# ----- 模擬資料 -----
np.random.seed(0)
duration = 5   # seconds
n_samples = int(freq * duration)
t = np.linspace(0, duration, n_samples)
signal_clean = np.sin(2 * np.pi * 0.5 * t)
noise = np.random.normal(0, 0.1, len(t))
signal_noisy = signal_clean + noise

filter_obj = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)
filtered_signal = np.array([filter_obj.filter(x) for x in signal_noisy])

# ----- 畫圖 -----
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, signal_noisy, label="Noisy Signal", alpha=0.5)
ax.plot(t, signal_clean, label="Ground Truth", linestyle="--")
ax.plot(t, filtered_signal, label="1 Euro Filter Output", linewidth=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Signal Value")
ax.legend()
ax.grid(True)

st.pyplot(fig)







