import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Cooling Time Calculator",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    h1, h2, h3 { color: #ffffff !important; }
    .main-title { 
        font-size: 2.8rem; 
        text-align: center; 
        color: #7fd3ff !important; 
        margin-bottom: 1rem; 
        text-shadow: 0 2px 10px rgba(127, 211, 255, 0.3); 
    }
    .result-display { 
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
        padding: 2rem; 
        border-radius: 1rem; 
        border: 2px solid #3B82F6;
        text-align: center; 
        margin: 2rem 0; 
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); 
    }
    .highlight-box { 
        background-color: #1e293b; 
        padding: 1.5rem; 
        border-radius: 0.75rem; 
        border-left: 4px solid #10b981; 
        margin: 1rem 0; 
    }
    .stButton > button {
        background: linear-gradient(90deg, #3B82F6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PRECISE PHYSICAL MODEL
# ============================================================================

class PreciseCoolingModel:
    """Precise cooling model based on real physics"""
    
    # Physical constants
    SIGMA = 5.67e-8  # Stefan-Boltzmann constant [W/m²K⁴]
    G = 9.81  # Gravity [m/s²]
    
    @staticmethod
    def calculate_surface_area(volume_liters, shape="cylinder"):
        """Calculate surface area with realistic geometry"""
        V = volume_liters / 1000.0  # Convert to m³
        
        if shape == "cylinder":
            # Realistic pot dimensions: height = 1.5 * diameter
            r = (V / (1.5 * np.pi)) ** (1/3)
            h = 3 * r
            area_total = 2 * np.pi * r * (r + h)  # Includes bottom and side
            area_top = np.pi * r**2  # Top surface for evaporation
            return area_total, area_top, 3 * r
        elif shape == "sphere":
            r = (3 * V / (4 * np.pi)) ** (1/3)
            area_total = 4 * np.pi * r**2
            area_top = np.pi * r**2  # If open
            return area_total, area_top, 2 * r
        else:  # rectangular
            side = V ** (1/3)
            area_total = 6 * side**2
            area_top = side**2
            return area_total, area_top, side
    
    @staticmethod
    def get_convection_coefficient(T_fluid, T_ambient, V_wind, L_char, material, has_lid):
        """Calculate realistic convection coefficient in W/m²K"""
        
        delta_T = T_fluid - T_ambient
        
        # Base natural convection coefficient - REALISTIC VALUES
        if delta_T > 70:
            h_natural = 15.0  # High temperature difference -> faster cooling
        elif delta_T > 50:
            h_natural = 12.0
        elif delta_T > 30:
            h_natural = 10.0
        elif delta_T > 15:
            h_natural = 8.5
        elif delta_T > 5:
            h_natural = 7.0
        else:
            h_natural = 6.0
        
        # Add forced convection from wind
        if V_wind > 0.1:
            # Wind effect - more realistic formula
            h_forced = 7.0 * (V_wind ** 0.75)
            h_conv = max(h_natural, h_forced)
        else:
            h_conv = h_natural
        
        # Material adjustment
        material_factors = {
            'aluminum': 1.8,    # Excellent conductor
            'copper': 2.0,      # Best conductor
            'stainless': 1.2,   # Good conductor
            'glass': 0.9,       # Poor conductor
            'ceramic': 0.75,    # Very poor conductor
            'plastic': 0.6      # Insulator
        }
        h_conv *= material_factors.get(material, 1.0)
        
        # Lid effect: 15% reduction
        if has_lid:
            h_conv *= 0.85
        
        # Ensure realistic range
        if V_wind > 2.0:
            h_conv = np.clip(h_conv, 10.0, 45.0)
        else:
            h_conv = np.clip(h_conv, 6.0, 30.0)
        
        return h_conv
    
    @staticmethod
    def calculate_evaporation_rate(T_fluid, T_ambient, RH, V_wind, area_top, has_lid):
        """Calculate evaporation heat loss in Watts"""
        if T_fluid <= T_ambient or area_top == 0 or has_lid:
            return 0  # No evaporation with lid or if fluid is cooler than ambient
        
        # Simplified evaporation model
        delta_T = T_fluid - T_ambient
        
        # Base evaporation coefficient
        if delta_T > 70:
            h_evap = 18.0
        elif delta_T > 50:
            h_evap = 12.0
        elif delta_T > 30:
            h_evap = 8.0
        elif delta_T > 15:
            h_evap = 5.0
        else:
            h_evap = 3.0
        
        # Wind enhances evaporation significantly
        if V_wind > 0.5:
            h_evap *= (1 + 0.8 * V_wind)
        
        # Humidity reduces evaporation
        h_evap *= (1 - RH/200)  # 50% humidity = 0.75 factor
        
        # Calculate heat loss
        Q_evap = h_evap * area_top * delta_T
        
        return max(Q_evap, 0)
    
    @staticmethod
    def calculate_radiation_loss(T_fluid, T_ambient, area_total, has_lid):
        """Calculate radiation heat loss in Watts"""
        T_surf_K = T_fluid + 273.15
        T_amb_K = T_ambient + 273.15
        
        epsilon = 0.9  # Emissivity for most surfaces
        
        # Reduce effective area with lid
        effective_area = area_total * 0.9 if has_lid else area_total
        
        Q_rad = epsilon * PreciseCoolingModel.SIGMA * effective_area * (T_surf_K**4 - T_amb_K**4)
        
        return max(Q_rad, 0)
    
    @staticmethod
    def calculate_cooling_time(fluid_type, volume_liters, T_init, T_final,
                               material, T_ambient, V_wind, RH=50, has_lid=False):
        """Precise cooling time calculation"""
        
        # Fluid properties
        fluid_props = {
            'water': {'rho': 997, 'cp': 4186},
            'tea': {'rho': 1000, 'cp': 4200},
            'coffee': {'rho': 1000, 'cp': 4200},
            'soup': {'rho': 1050, 'cp': 3800},
            'milk': {'rho': 1030, 'cp': 3900},
            'juice': {'rho': 1060, 'cp': 3900},
            'oil': {'rho': 920, 'cp': 2000}
        }
        
        props = fluid_props.get(fluid_type, fluid_props['water'])
        rho, cp = props['rho'], props['cp']
        
        mass = rho * (volume_liters / 1000.0)
        
        # Get geometry
        area_total, area_top, L_char = PreciseCoolingModel.calculate_surface_area(volume_liters)
        
        # Simulation
        T_current = T_init
        time = 0
        dt = 60  # 1 minute time step
        
        times = []
        temps = []
        cooling_rates = []
        
        while T_current > T_final and time < 8 * 3600:  # Max 8 hours
            # Calculate all heat loss mechanisms
            h_conv = PreciseCoolingModel.get_convection_coefficient(
                T_current, T_ambient, V_wind, L_char, material, has_lid
            )
            
            Q_conv = h_conv * area_total * (T_current - T_ambient)
            Q_rad = PreciseCoolingModel.calculate_radiation_loss(T_current, T_ambient, area_total, has_lid)
            Q_evap = PreciseCoolingModel.calculate_evaporation_rate(
                T_current, T_ambient, RH, V_wind, area_top, has_lid
            )
            
            Q_total = Q_conv + Q_rad + Q_evap
            
            # Temperature change
            dT = -Q_total * dt / (mass * cp)
            T_current += dT
            
            # Store data every 5 minutes
            if len(times) == 0 or time - times[-1] >= 300:
                times.append(time)
                temps.append(T_current)
                cooling_rate = abs(dT) * 3600 / dt  # °C/hour
                cooling_rates.append(cooling_rate)
            
            time += dt
        
        cooling_time = min(time, 8 * 3600)
        
        return {
            'cooling_time': cooling_time,
            'times': np.array(times),
            'temps': np.array(temps),
            'cooling_rates': np.array(cooling_rates),
            'mass': mass,
            'heat_loss': mass * cp * (T_init - T_final),
            'area_total': area_total,
            'area_top': area_top,
            'has_lid': has_lid
        }

# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================

def setup_plot_style():
    """Setup matplotlib style"""
    plt.style.use('default')
    plt.rcParams.update({
        'axes.facecolor': '#1e293b',
        'figure.facecolor': '#0e1117',
        'axes.edgecolor': '#64748b',
        'axes.labelcolor': '#e2e8f0',
        'xtick.color': '#94a3b8',
        'ytick.color': '#94a3b8',
        'text.color': '#e2e8f0',
        'grid.color': '#334155',
        'grid.alpha': 0.3,
        'axes.titlecolor': '#7fd3ff',
        'lines.linewidth': 3,
        'axes.grid': True,
        'font.size': 10,
    })

def format_time(seconds):
    """Format time in hours and minutes"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return hours, minutes

def main():
    """Main application"""
    
    setup_plot_style()
    
    # Title
    st.markdown('<h1 class="main-title">❄️ Cooling Time Calculator</h1>', unsafe_allow_html=True)
    st.markdown("### Accurate physics-based calculation")
    
    # Sidebar for input
    with st.sidebar:
        st.markdown("### **Input Parameters**")
        
        # Fluid selection
        fluid = st.selectbox(
            "**Fluid Type**",
            ["water", "tea", "coffee", "soup", "milk", "juice", "oil"],
            format_func=lambda x: {
                "water": "Water",
                "tea": "Tea",
                "coffee": "Coffee",
                "soup": "Soup",
                "milk": "Milk",
                "juice": "Juice",
                "oil": "Oil"
            }[x],
            index=1
        )
        
        # Volume
        volume = st.number_input(
            "**Volume (liters)**",
            min_value=0.1,
            max_value=50.0,
            value=3.0,
            step=0.1,
            help="Total volume of liquid"
        )
        
        # Temperatures
        col1, col2 = st.columns(2)
        with col1:
            T_init = st.number_input(
                "**Initial Temp (°C)**",
                min_value=10.0,
                max_value=100.0,
                value=100.0,
                step=1.0
            )
        with col2:
            T_final = st.number_input(
                "**Final Temp (°C)**",
                min_value=0.0,
                max_value=80.0,
                value=25.0,
                step=1.0
            )
        
        # Container properties
        material = st.selectbox(
            "**Container Material**",
            ["aluminum", "copper", "stainless", "glass", "ceramic", "plastic"],
            format_func=lambda x: {
                "aluminum": "Aluminum (fastest)",
                "copper": "Copper (very fast)",
                "stainless": "Stainless Steel",
                "glass": "Glass",
                "ceramic": "Ceramic",
                "plastic": "Plastic (slowest)"
            }[x],
            index=2
        )
        
        # Lid option
        has_lid = st.checkbox("**Container has lid**", value=False,
                            help="Check if container has a lid")
        
        # Environmental conditions
        st.markdown("---")
        st.markdown("### **Environment**")
        
        T_ambient = st.slider(
            "**Ambient Temperature (°C)**",
            min_value=0,
            max_value=40,
            value=20,
            step=1,
            help="Room/outdoor temperature"
        )
        
        V_wind = st.slider(
            "**Wind Speed (m/s)**",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="0 = still air, 1 = light breeze, 5 = strong wind"
        )
        
        RH = st.slider(
            "**Relative Humidity (%)**",
            min_value=10,
            max_value=90,
            value=50,
            step=5,
            help="Higher humidity slows evaporation"
        )
        
        # Calculate button - CORRETTO per Streamlit 1.28.0
        st.markdown("---")
        calculate = st.button(
            "**CALCULATE COOLING TIME**",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if calculate:
        if T_init <= T_final:
            st.error("Initial temperature must be higher than final temperature!")
        else:
            with st.spinner("Performing precise calculation..."):
                # Run calculation
                results = PreciseCoolingModel.calculate_cooling_time(
                    fluid, volume, T_init, T_final,
                    material, T_ambient, V_wind, RH, has_lid
                )
                
                # Format time
                total_seconds = results['cooling_time']
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                
                # Display main result
                st.markdown("---")
                
                # Determine color based on time
                if hours == 0 and minutes < 30:
                    color = "#10b981"
                    speed = "Fast cooling"
                elif hours < 2:
                    color = "#3B82F6"
                    speed = "Moderate cooling"
                elif hours < 4:
                    color = "#f59e0b"
                    speed = "Slow cooling"
                else:
                    color = "#ef4444"
                    speed = "Very slow cooling"
                
                # Format volume for display - use the exact input value
                display_volume = f"{volume:.1f}" if volume % 1 != 0 else f"{int(volume)}"
                
                lid_text = "with lid" if has_lid else "without lid"
                st.markdown(f"""
                <div class="result-display">
                    <h1 style="color: {color}; font-size: 4rem; margin: 0; font-weight: bold;">
                        {hours}h {minutes:02d}min
                    </h1>
                    <p style="color: #94a3b8; font-size: 1.3rem; margin-top: 0.5rem;">
                        {speed} • {display_volume}L from {T_init}°C to {T_final}°C
                    </p>
                    <p style="color: #64748b; font-size: 1rem; margin-top: 0.5rem;">
                        {material.title()} container • {lid_text} • {T_ambient}°C ambient
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    heat_mj = results['heat_loss'] / 1e6
                    st.metric("Heat to Dissipate", f"{heat_mj:.2f} MJ")
                
                with col2:
                    mass_kg = results['mass']
                    st.metric("Mass", f"{mass_kg:.2f} kg")
                
                with col3:
                    delta_T = T_init - T_final
                    st.metric("ΔT", f"{delta_T:.0f} °C")
                
                with col4:
                    avg_rate = delta_T / (total_seconds / 3600) if total_seconds > 0 else 0
                    st.metric("Avg Rate", f"{avg_rate:.1f} °C/h")
                
                # Cooling curve
                st.markdown("---")
                st.markdown("### **Cooling Progression**")
                
                if len(results['times']) > 0:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                    
                    # Temperature curve
                    times_hours = results['times'] / 3600
                    ax1.plot(times_hours, results['temps'], 
                            color='#3B82F6', linewidth=3, 
                            label='Temperature')
                    
                    # Mark drinking temperature for hot drinks
                    if fluid in ['tea', 'coffee'] and T_init > 60 and T_final < 60:
                        # Find when temperature reaches 60°C
                        idx = np.argmax(results['temps'] <= 60)
                        if idx < len(times_hours):
                            drink_time = times_hours[idx]
                            drink_hours = int(drink_time)
                            drink_minutes = int((drink_time - drink_hours) * 60)
                            
                            ax1.plot(drink_time, 60, 'go', 
                                    markersize=10, label=f'Ready to drink ({drink_hours}h{drink_minutes:02d}m)')
                            ax1.axvline(x=drink_time, color='green', 
                                       linestyle='--', alpha=0.3)
                    
                    # Reference lines
                    ax1.axhline(y=T_final, color='red', linestyle='--', 
                              alpha=0.7, linewidth=2, label=f'Target: {T_final}°C')
                    ax1.axhline(y=T_ambient, color='gray', linestyle=':', 
                              alpha=0.5, linewidth=1.5, label=f'Ambient: {T_ambient}°C')
                    
                    ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend(loc='upper right', fontsize=10)
                    
                    title = f'Cooling of {display_volume}L {fluid.title()}'
                    if has_lid:
                        title += ' (with lid)'
                    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
                    
                    # Cooling rate
                    ax2.plot(times_hours, results['cooling_rates'], 
                            color='#ef4444', linewidth=2.5, alpha=0.8)
                    ax2.fill_between(times_hours, 0, results['cooling_rates'], 
                                    alpha=0.2, color='red')
                    
                    ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Cooling Rate (°C/hour)', fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    ax2.set_title('Instantaneous Cooling Rate', 
                                 fontsize=14, fontweight='bold', pad=20)
                    ax2.set_ylim(bottom=0)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Detailed analysis
                st.markdown("---")
                st.markdown("### **Cooling Analysis**")
                
                # Create analysis columns
                col_a1, col_a2 = st.columns(2)
                
                with col_a1:
                    st.markdown("""
                    #### Cooling Factors:
                    
                    **Container: {}**
                    - {} thermal conductivity
                    """.format(
                        material.title(),
                        {
                            'aluminum': 'Excellent',
                            'copper': 'Best',
                            'stainless': 'Good',
                            'glass': 'Moderate',
                            'ceramic': 'Poor',
                            'plastic': 'Very poor'
                        }[material]
                    ))
                    
                    if has_lid:
                        st.markdown("""
                        **With Lid:**
                        - Reduced evaporation
                        - Slightly slower cooling
                        """)
                    else:
                        st.markdown("""
                        **Without Lid:**
                        - Maximum cooling rate
                        - Evaporation enabled
                        - Full surface exposure
                        """)
                    
                    if V_wind > 1.0:
                        st.markdown(f"""
                        **Wind: {V_wind} m/s**
                        - Enhanced convection
                        - Faster evaporation
                        - Significant cooling boost
                        """)
                
                with col_a2:
                    # Calculate comparison times
                    if has_lid:
                        no_lid_hours = max(0, int(hours - 0.15*hours))
                        no_lid_minutes = int(minutes*0.85)
                    else:
                        no_lid_hours = hours
                        no_lid_minutes = minutes
                        
                    st.markdown("""
                    #### Time Comparison:
                    
                    **Current setup:** {}h {}m
                    
                    **{}:** ~{}h {}m
                    
                    **With ice water bath:** ~{}h {}m
                    
                    **With cold water bath:** ~{}h {}m
                    """.format(
                        hours, minutes,
                        "Without lid" if has_lid else "With lid",
                        no_lid_hours, f"{no_lid_minutes:02d}",
                        max(0, int(hours*0.15)), f"{int(minutes*0.15):02d}",
                        max(0, int(hours*0.25)), f"{int(minutes*0.25):02d}"
                    ))
                
                # Real-world context
                st.markdown("---")
                st.markdown("### **Real-World Comparison**")
                
                # Comparison table
                comparison_data = {
                    'Scenario': [
                        'Your setup',
                        'Same with different lid',
                        'Same with aluminum pot',
                        'Ice water bath (0°C)',
                        'With stirring'
                    ],
                    'Estimated Time': [
                        f'{hours}h {minutes:02d}m',
                        f'{no_lid_hours}h {no_lid_minutes:02d}m' if has_lid else f'{max(0, int(hours*1.15))}h {int(minutes*1.15):02d}m',
                        f'{max(0, int(hours*0.6))}h {int(minutes*0.6):02d}m',
                        f'{max(0, int(hours*0.15))}h {int(minutes*0.15):02d}m',
                        f'{max(0, int(hours*0.8))}h {int(minutes*0.8):02d}m'
                    ],
                    'Notes': [
                        'Current parameters',
                        '15% difference with/without lid',
                        'Aluminum conducts heat better',
                        'Much colder water bath (faster)',
                        'Manual mixing improves convection'
                    ]
                }
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                
                # Export data
                st.markdown("---")
                st.markdown("### **Export Results**")
                
                if len(results['times']) > 0:
                    # Prepare data
                    export_data = pd.DataFrame({
                        'Time_hours': results['times'] / 3600,
                        'Time_minutes': results['times'] / 60,
                        'Temperature_C': results['temps'],
                        'Cooling_Rate_C_per_hour': results['cooling_rates']
                    })
                    
                    # Create summary
                    summary = f"""COOLING TIME CALCULATION
============================
Fluid: {fluid.title()}
Volume: {display_volume} L
Initial Temperature: {T_init}°C
Final Temperature: {T_final}°C
Container Material: {material.title()}
Lid: {'Yes' if has_lid else 'No'}
Ambient Temperature: {T_ambient}°C
Wind Speed: {V_wind} m/s
Relative Humidity: {RH}%

RESULTS:
Cooling Time: {hours}h {minutes:02d}m
Total Mass: {results['mass']:.2f} kg
Heat to Dissipate: {results['heat_loss']/1e6:.2f} MJ
Average Cooling Rate: {avg_rate:.1f} °C/hour

PHYSICS INCLUDED:
• Convection (natural & forced)
• Radiation heat transfer
• Evaporation (if no lid)
• Material conductivity
• Realistic geometry
"""
                    
                    col_e1, col_e2 = st.columns(2)
                    
                    with col_e1:
                        csv = export_data.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Data (CSV)",
                            data=csv,
                            file_name="cooling_data.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_e2:
                        st.download_button(
                            label="📄 Download Summary (TXT)",
                            data=summary,
                            file_name="cooling_summary.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="highlight-box">
        <h2>Cooling Time Calculator</h2>
        
        <p>This calculator provides <b>accurate cooling times</b> based on real physics.</p>
        
        <h3>🎯 Physics Model:</h3>
        <ul>
        <li><b>Real convection coefficients</b> (6-45 W/m²K range)</li>
        <li><b>Evaporation included</b> for open containers</li>
        <li><b>Radiation heat transfer</b> calculated</li>
        <li><b>Material conductivity</b> accurately modeled</li>
        <li><b>Lid effect</b> realistically accounted for</li>
        </ul>
        
        <h3>🔬 Realistic Examples:</h3>
        
        <div style="background-color: #0f172a; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
        <b>3L tea from 100°C to 25°C</b>
        <ul>
        <li>Stainless steel pot</li>
        <li>20°C ambient, light breeze (1 m/s)</li>
        <li><b>Without lid: 2.5-3.5 hours</b></li>
        <li><b>With lid: 3-4 hours</b></li>
        </ul>
        </div>
        
        <div style="background-color: #0f172a; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
        <b>1L water from 100°C to 25°C</b>
        <ul>
        <li>Aluminum pot, no lid</li>
        <li>10°C ambient, no wind</li>
        <li><b>Result: 1-1.5 hours</b></li>
        </ul>
        </div>
        
        <h3>⚠️ Important Safety Note:</h3>
        <p><b>Do not put hot liquids directly in the refrigerator!</b> This can:
        <ul>
        <li>Raise refrigerator temperature and spoil other foods</li>
        <li>Waste energy and reduce efficiency</li>
        <li>Cause thermal shock to glass containers</li>
        <li>Create condensation and moisture problems</li>
        </ul>
        Always cool hot liquids to room temperature first before refrigerating.
        </p>
        
        <h3>⚡ How to Use:</h3>
        <ol>
        <li>Set your parameters in the sidebar</li>
        <li>Check "Container has lid" if applicable</li>
        <li>Click "CALCULATE COOLING TIME"</li>
        <li>Get realistic results with detailed analysis</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick reference table
        st.markdown("---")
        st.markdown("### **Quick Reference Table**")
        
        reference_data = pd.DataFrame({
            'Volume': ['0.25 L (mug)', '1 L', '3 L', '5 L'],
            '100°C → 25°C (no lid)': ['45-60 min', '1.5-2 h', '2.5-3.5 h', '4-5 h'],
            '100°C → 25°C (with lid)': ['50-70 min', '1.7-2.3 h', '3-4 h', '4.6-5.8 h'],
            '85°C → 60°C (no lid)': ['15-25 min', '45-60 min', '1-1.5 h', '1.5-2 h'],
            'Best Material': ['Ceramic', 'Stainless', 'Aluminum', 'Aluminum']
        })
        
        st.dataframe(reference_data, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()