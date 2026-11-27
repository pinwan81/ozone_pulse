import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------
# 1. 光学系およびパルス・エネルギーパラメータ設定
# ---------------------------------------------------------
wavelength = 266e-9         # 波長 lambda = 266 nm
f = 50e-3                   # レンズ焦点距離 f = 50 mm
diameter_in = 3.0e-3        # 初期ビーム径 = 3.0 mm
w_in = diameter_in / 2      # 初期ビーム半径 = 1.5 mm

tau_fwhm = 3.0e-9           # パルス半値全幅 3 ns

# レーザーのパルスエネルギー（例: 1.0 mJ = 1.0e-3 J）
pulse_energy = 1.0e-3       

# 焦点パラメータの計算
w0y = (wavelength * f) / (np.pi * w_in)   # 焦点ウエスト半径 y (約 2.82 um)
x_R_in = (np.pi * w_in**2) / wavelength   # 入射レイリー長
x_R2y = (np.pi * w0y**2) / wavelength     # 焦点後の鉛直レイリー長 (約 94.1 um)
d2 = f / (1 + (f / x_R_in)**2)            # 焦点位置 (z ≈ 50.0 mm)

# パルスエネルギーから絶対ピーク強度 I_0 [W/m^2] を算出
spatial_integral_factor = 0.5 * np.pi * w0y**2
temporal_integral_factor = (tau_fwhm / 2.0) * np.sqrt(np.pi / np.log(2))
I_peak_absolute = pulse_energy / (spatial_integral_factor * temporal_integral_factor)

# 表示用単位の選択 (W/m^2 -> W/cm^2 換算: 1 m^2 = 10^4 cm^2)
intensity_scale = 1e-4  
max_absolute_intensity = I_peak_absolute * intensity_scale

# ---------------------------------------------------------
# 2. 高分解能な空間および時間グリッドの定義
# ---------------------------------------------------------
dy = 0.05e-6    # y方向: 0.05 μm (50 nm)
dz = 1.0e-6     # z方向: 1.0 μm (1 μm)

y = np.arange(-10e-6, 10e-6 + dy, dy)
z_rel = np.arange(-300e-6, 300e-6 + dz, dz)

Z_rel, Y = np.meshgrid(z_rel, y)

# ビーム半径の包絡線 w_y(z) の計算
w_y_2d = w0y * np.sqrt(1 + (Z_rel / x_R2y)**2)

# 絶対値ベースの空間強度分布ベース (W/cm^2)
I_zy_abs_base = max_absolute_intensity * (w0y / w_y_2d) * np.exp(-2 * (Y / w_y_2d)**2)

# 超高時間分解能の計算ステップ (dt = 0.01 ns, 範囲: ±2 * tau_fwhm)
dt = 0.01e-9  
t_steps = np.arange(-2.0 * tau_fwhm, 2.0 * tau_fwhm + dt, dt)

# アニメーション用フレーム (300コマ、高速・滑らか)
num_animation_frames = 300
frame_indices = np.linspace(0, len(t_steps) - 1, num_animation_frames, dtype=int)
t_frames = t_steps[frame_indices]

# ---------------------------------------------------------
# 3. 2画面構成（左: 絶対値レーザー強度, 右: 時間波形）の構築
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

t_initial = t_frames[0]
I_t_initial = np.exp(-4 * np.log(2) * (t_initial / tau_fwhm)**2)
initial_I = I_zy_abs_base * I_t_initial

# --- [左側] 2D空間絶対レーザー強度分布 (包絡線つき) ---
im1 = ax1.imshow(
    initial_I,
    extent=[z_rel[0] * 1e6, z_rel[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6],
    origin='lower',
    aspect='auto',
    cmap='inferno',
    vmin=0.0,
    vmax=max_absolute_intensity  # 絶対値のピーク強度でカラーバー上限を固定
)
# ビーム半径の包絡線（白破線）を配置
ax1.plot(z_rel * 1e6, w_y_2d[len(y)//2, :] * 1e6, color='white', linestyle='--', linewidth=1.2, label='Beam Radius $w_y(z)$')
ax1.plot(z_rel * 1e6, -w_y_2d[len(y)//2, :] * 1e6, color='white', linestyle='--', linewidth=1.2)

ax1.set_xlabel(r'Propagation Distance $\Delta z$ [μm]', fontsize=11)
ax1.set_ylabel(r'Vertical Position $y$ [μm]', fontsize=11)
ax1.set_title(f'Absolute Intensity | Time: {t_initial*1e9:+.3f} ns', fontsize=12)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
fig.colorbar(im1, ax=ax1, label=r'Intensity [W/cm$^2$]')

# --- [右側] 時間パルス波形 (横軸: 時間, 縦軸: 強度) ---
t_ns = t_steps * 1e9
I_t_profile = np.exp(-4 * np.log(2) * (t_steps / tau_fwhm)**2)

ax2.plot(t_ns, I_t_profile, color='crimson', linewidth=2, label='Pulse Profile (FWHM = 3 ns)')
ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='FWHM Half-max')

time_line = ax2.axvline(t_initial * 1e9, color='cyan', linestyle='-', linewidth=1.5, label='Current Time')
time_dot, = ax2.plot([t_initial * 1e9], [I_t_initial], marker='o', color='cyan', markersize=8)

ax2.set_xlabel('Time $t$ [ns]', fontsize=11)
ax2.set_ylabel('Normalized Temporal Intensity', color='crimson', fontsize=11)
ax2.set_title('Temporal Pulse Profile (Time Evolution)', fontsize=12)
ax2.set_ylim(-0.05, 1.1)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()

# フレーム更新関数
def update(anim_idx):
    global_idx = frame_indices[anim_idx]
    t_sec = t_steps[global_idx]
    
    I_t = np.exp(-4 * np.log(2) * (t_sec / tau_fwhm)**2)
    curr_I = I_zy_abs_base * I_t
    
    # 1. 左側（絶対値レーザー強度空間分布）の更新
    im1.set_array(curr_I)
    ax1.set_title(f'Absolute Intensity | Time: {t_sec*1e9:+.3f} ns', fontsize=12)
    
    # 2. 右側（時間波形マーカー）の更新
    t_val_ns = t_sec * 1e9
    time_line.set_xdata([t_val_ns, t_val_ns])
    time_dot.set_data([t_val_ns], [I_t])
    
    return [im1, time_line, time_dot]

# 60fps で滑らかに動作
ani = animation.FuncAnimation(fig, update, frames=len(t_frames), interval=16, blit=True)

# ---------------------------------------------------------
# 4. 保存と実行
# ---------------------------------------------------------
output_dir = r"C:\Users\たいが\Documents\Code\nono"
os.makedirs(output_dir, exist_ok=True)
gif_filepath = os.path.join(output_dir, "laser_pulse_absolute_intensity_pure.gif")

print(f"GIFアニメーションを保存中...\n保存先: {gif_filepath}")
ani.save(gif_filepath, writer='pillow', fps=60)
print(f"保存が完了しました:\n{gif_filepath}")

plt.show()