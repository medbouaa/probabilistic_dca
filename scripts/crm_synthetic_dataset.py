import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Synthetic Data Generation ---

def pan_crm_rate(t, delta_p, b, J_inf, ctVp):
    """Pan CRM model: calculate production rate over time."""
    J_t = b / np.sqrt(t) + J_inf
    rate = (delta_p / J_t) * np.exp(- (2 * b * np.sqrt(t) + J_inf * t) / (ctVp))
    return rate

def generate_synthetic_pan_crm_data(days=400, delta_p=2000, b=2.6, J_inf=1.4, ctVp=180.0, noise_level=0.2, random_seed=42):
    """Generate synthetic production data using Pan CRM model with Gaussian noise."""
    np.random.seed(random_seed)
    t = np.arange(10, days + 1)  # Start from day 10 to avoid division by zero
    true_rate = pan_crm_rate(t, delta_p, b, J_inf, ctVp)

    # Add Gaussian noise: mean = 0, std = 20% of true rate at each point
    noise = np.random.normal(loc=0, scale=noise_level * true_rate)
    noisy_rate = true_rate + noise

    # Ensure no negative rates
    noisy_rate = np.clip(noisy_rate, a_min=0, a_max=None)

    # Create DataFrame
    df = pd.DataFrame({
        'Day': t,
        'True Rate': true_rate,
        'Synthetic Rate': noisy_rate
    })

    return df

# Generate the synthetic data
synthetic_data = generate_synthetic_pan_crm_data()

# --- Plotting ---

plt.figure(figsize=(8, 5))
plt.plot(synthetic_data['Day'], synthetic_data['True Rate'], label='True Rate (Pan CRM)', linewidth=2, color='blue')
plt.scatter(synthetic_data['Day'], synthetic_data['Synthetic Rate'], label='Synthetic Rate (Noisy Data)', color='orange', s=20, alpha=0.7)

plt.title('Synthetic Production Data: True vs Noisy Rate')
plt.xlabel('Time (days)')
plt.ylabel('Production Rate')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

synthetic_data.to_csv("synthetic_pan_crm_data.csv", index=False)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Parameters (your chosen ones) ---
delta_p = 820
b = 2.6
J_inf = 0.75
ctVp = 100.0

# --- Pan CRM rate function ---
def pan_crm_rate(t, delta_p, b, J_inf, ctVp):
    """Pan CRM model: calculate production rate over time."""
    J_t = b / np.sqrt(t) + J_inf
    rate = (delta_p / J_t) * np.exp(- (2 * b * np.sqrt(t) + J_inf * t) / ctVp)
    return rate

# --- Time array ---
t_days = np.arange(10, 5401)  # Day 10 to Day 5400

# --- Calculate daily rate ---
daily_rate = pan_crm_rate(t_days, delta_p, b, J_inf, ctVp)

# --- Calculate cumulative production ---
# Approximate daily production (bbl/day), integrate over time
# Since the time steps are daily, cumulative sum is fine
cumulative_production = np.cumsum(daily_rate)

# --- Create DataFrame ---
df = pd.DataFrame({
    'Day': t_days,
    'Production Rate (bbl/day)': daily_rate,
    'Cumulative Production (bbl)': cumulative_production
})

# --- Print final cumulative production at day 5400 ---
final_cum_prod = df['Cumulative Production (bbl)'].iloc[-1]
print(f"✅ Cumulative production at day 5400: {final_cum_prod:,.2f} bbl")

# --- Optional: Plot ---
plt.figure(figsize=(10, 5))

# Plot production rate
plt.subplot(1, 2, 1)
plt.plot(df['Day'], df['Production Rate (bbl/day)'], color='blue')
plt.title('Production Rate Over Time')
plt.xlabel('Day')
plt.ylabel('Rate (bbl/day)')
plt.grid(True)

# Plot cumulative production
plt.subplot(1, 2, 2)
plt.plot(df['Day'], df['Cumulative Production (bbl)'], color='green')
plt.title('Cumulative Production Over Time')
plt.xlabel('Day')
plt.ylabel('Cumulative Production (bbl)')
plt.grid(True)

plt.tight_layout()
plt.show()
