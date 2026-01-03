import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Cooling Time Calculator",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { 
        background-color: #0e1117; 
        color: #fafafa; 
    }
    
    .main-title { 
        font-size: 2.8rem; 
        text-align: center; 
        color: #7fd3ff !important; 
        margin-bottom: 1rem; 
        text-shadow: 0 2px 10px rgba(127, 211, 255, 0.3); 
    }
    
    .result-box { 
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
        padding: 2rem; 
        border-radius: 1rem; 
        border: 2px solid #3B82F6;
        text-align: center; 
        margin: 2rem 0; 
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); 
    }
    
    .metric-box {
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #334155;
        margin: 0.5rem 0;
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
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.3);
    }
    
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

class CoolingCalculator:
    """Advanced cooling time calculator with realistic physics"""
    
    MATERIALS = {
        'stainless_steel': {
            'density': 8000, 'cp': 500, 'emissivity': 0.15, 'thickness': 0.0015
        },
        'aluminum': {
            'density': 2700, 'cp': 897, 'emissivity': 0.09, 'thickness': 0.002
        },
        'copper': {
            'density': 8960, 'cp': 385, 'emissivity': 0.04, 'thickness': 0.001
        },
        'glass': {
            'density': 2500, 'cp': 840, 'emissivity': 0.92, 'thickness': 0.003
        },
        'ceramic': {
            'density': 2400, 'cp': 1000, 'emissivity': 0.90, 'thickness': 0.005
        }
    }
    
    LIQUIDS = {
        'water': {'density': 998, 'cp': 4186},
        'tea': {'density': 1000, 'cp': 4200},
        'coffee': {'density': 1000, 'cp': 4180},
        'milk': {'density': 1030, 'cp': 3930},
        'soup': {'density': 1050, 'cp': 3800},
        'oil': {'density': 920, 'cp': 2000}
    }
    
    @staticmethod
    def calculate_geometry(diameter_cm, height_cm, volume_liters=None):
        """Calculate geometry based on diameter and height, with optional volume"""
        d = diameter_cm / 100  # meters
        h_container = height_cm / 100  # meters
        
        # Calculate maximum container volume
        max_volume = np.pi * (d/2)**2 * h_container * 1000  # liters
        
        if volume_liters is not None:
            if volume_liters > max_volume:
                raise ValueError(f"Volume {volume_liters}L exceeds container capacity {max_volume:.1f}L")
            # Calculate actual liquid height
            h_liquid = volume_liters / 1000 / (np.pi * (d/2)**2)  # meters
        else:
            h_liquid = h_container
            volume_liters = max_volume
        
        # Calculate areas
        A_side = np.pi * d * h_liquid
        A_top = np.pi * (d/2)**2
        A_bottom = np.pi * (d/2)**2
        
        # Container surface area for mass calculation
        A_container = np.pi * d * h_container + 2 * np.pi * (d/2)**2
        
        return {
            'diameter': d,
            'height_container': h_container,
            'height_liquid': h_liquid,
            'area_side': A_side,
            'area_top': A_top,
            'area_bottom': A_bottom,
            'area_container': A_container,
            'volume_liters': volume_liters,
            'max_volume_liters': max_volume
        }
    
    @staticmethod
    def calculate_convection_coefficient(delta_T, wind_speed, lid_type):
        """Calculate convective heat transfer coefficient"""
        # Natural convection coefficient (W/m²K)
        if delta_T > 80:
            h_nat = 20.0
        elif delta_T > 60:
            h_nat = 18.0
        elif delta_T > 40:
            h_nat = 16.0
        elif delta_T > 20:
            h_nat = 14.0
        elif delta_T > 10:
            h_nat = 12.0
        else:
            h_nat = 10.0
        
        # Wind effect
        if wind_speed > 0.1:
            h_forced = 5.6 + 3.8 * wind_speed**0.8
            h_conv = max(h_nat, h_forced)
        else:
            h_conv = h_nat
        
        # Lid effect
        lid_factors = {
            'tight': 0.4,      # 60% reduction
            'air_gap': 0.7,    # 30% reduction
            'none': 1.0        # No reduction
        }
        return h_conv * lid_factors.get(lid_type, 0.7)
    
    @staticmethod
    def calculate_cooling_time(diameter_cm, height_cm, material, liquid_type,
                              volume_liters, T_initial, T_final, T_ambient,
                              wind_speed=0.0, lid_type='air_gap', ceramic_base=True):
        """Main calculation function"""
        
        # Get properties
        mat_props = CoolingCalculator.MATERIALS.get(material, 
            CoolingCalculator.MATERIALS['stainless_steel'])
        liquid_props = CoolingCalculator.LIQUIDS.get(liquid_type,
            CoolingCalculator.LIQUIDS['water'])
        
        # Calculate geometry
        geometry = CoolingCalculator.calculate_geometry(diameter_cm, height_cm, volume_liters)
        
        # Calculate masses
        volume_m3 = volume_liters / 1000
        mass_liquid = volume_m3 * liquid_props['density']
        mass_container = geometry['area_container'] * mat_props['thickness'] * mat_props['density']
        
        # Thermal mass
        thermal_mass = (mass_liquid * liquid_props['cp'] + 
                       mass_container * mat_props['cp'])
        
        # Effective areas
        if ceramic_base:
            effective_bottom = geometry['area_bottom'] * 0.2  # 80% reduction
        else:
            effective_bottom = geometry['area_bottom']
        
        if lid_type == 'none':
            effective_top = geometry['area_top']
        else:
            effective_top = geometry['area_top'] * 0.5
        
        effective_area = geometry['area_side'] + effective_top + effective_bottom
        
        # Calculate average heat transfer coefficient
        delta_T_initial = T_initial - T_ambient
        T_mid = (T_initial + T_final) / 2
        delta_T_mid = T_mid - T_ambient
        
        h_conv_initial = CoolingCalculator.calculate_convection_coefficient(
            delta_T_initial, wind_speed, lid_type)
        h_conv_mid = CoolingCalculator.calculate_convection_coefficient(
            delta_T_mid, wind_speed, lid_type)
        h_conv_avg = (h_conv_initial + h_conv_mid) / 2
        
        # Radiation
        sigma = 5.67e-8
        emissivity = mat_props['emissivity']
        
        def h_rad(T):
            T_surf_K = T + 273.15
            T_amb_K = T_ambient + 273.15
            return emissivity * sigma * (T_surf_K**2 + T_amb_K**2) * (T_surf_K + T_amb_K)
        
        h_rad_avg = (h_rad(T_initial) + h_rad(T_mid)) / 2
        
        # Evaporation (only without lid)
        h_evap = 0
        if lid_type == 'none' and delta_T_initial > 30:
            h_evap = 4.0 * delta_T_initial**0.3
        
        h_total = h_conv_avg + h_rad_avg + h_evap
        
        # Cooling constant
        k = h_total * effective_area / thermal_mass  # 1/second
        
        # Cooling time
        if T_final <= T_ambient:
            cooling_time_seconds = float('inf')
        else:
            cooling_time_seconds = np.log((T_initial - T_ambient) / (T_final - T_ambient)) / k
        
        # Generate curve
        times = np.linspace(0, min(cooling_time_seconds * 1.2, 24 * 3600), 100)
        temps = T_ambient + (T_initial - T_ambient) * np.exp(-k * times)
        
        cooling_rate = (T_initial - T_final) / (cooling_time_seconds / 3600) if cooling_time_seconds > 0 else 0
        
        return {
            'cooling_time_seconds': cooling_time_seconds,
            'cooling_curve': {'times': times, 'temperatures': temps},
            'parameters': {
                'thermal_mass': thermal_mass,
                'heat_transfer_coefficient': h_total,
                'effective_area': effective_area,
                'cooling_constant': k,
                'cooling_rate': cooling_rate,
                'mass_liquid': mass_liquid,
                'mass_container': mass_container,
                'volume_liters': volume_liters
            },
            'geometry': geometry
        }

def format_time(seconds):
    """Format time in hours and minutes"""
    if seconds == float('inf'):
        return "∞"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes:02d}min"

def create_cooling_plot(times, temperatures, T_final, T_ambient):
    """Create cooling curve plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dark theme
    ax.set_facecolor('#1e293b')
    fig.patch.set_facecolor('#0e1117')
    
    ax.plot(times / 3600, temperatures, color='#3B82F6', linewidth=3, label='Temperature')
    ax.axhline(y=T_final, color='#EF4444', linestyle='--', alpha=0.7, label=f'Target: {T_final}°C')
    ax.axhline(y=T_ambient, color='#94A3B8', linestyle=':', alpha=0.5, label=f'Ambient: {T_ambient}°C')
    
    ax.set_xlabel('Time (hours)', color='#E2E8F0', fontsize=12)
    ax.set_ylabel('Temperature (°C)', color='#E2E8F0', fontsize=12)
    ax.set_title('Cooling Curve', color='#7FD3FF', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, color='#64748b')
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#E2E8F0')
    ax.tick_params(colors='#94A3B8')
    
    max_time_hours = min(times[-1] / 3600, 24)
    ax.set_xlim(0, max_time_hours)
    
    plt.tight_layout()
    return fig

def main():
    st.markdown('<h1 class="main-title">🌡️ Cooling Time Calculator</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🔧 Input Parameters")
        
        # Container dimensions
        st.markdown("#### 📏 Container Dimensions")
        col1, col2 = st.columns(2)
        with col1:
            diameter = st.number_input(
                "Diameter (cm)",
                min_value=5.0,
                max_value=100.0,
                value=17.0,
                step=0.5,
                help="Inner diameter of the container"
            )
        with col2:
            height = st.number_input(
                "Height (cm)",
                min_value=5.0,
                max_value=100.0,
                value=13.0,
                step=0.5,
                help="Inner height of the container"
            )
        
        # Liquid volume
        st.markdown("#### 💧 Liquid Properties")
        volume = st.number_input(
            "Liquid Volume (liters)",
            min_value=0.1,
            max_value=100.0,
            value=3.0,
            step=0.1
        )
        
        liquid = st.selectbox(
            "Liquid Type",
            options=list(CoolingCalculator.LIQUIDS.keys()),
            format_func=lambda x: x.title(),
            index=1
        )
        
        material = st.selectbox(
            "Container Material",
            options=list(CoolingCalculator.MATERIALS.keys()),
            format_func=lambda x: x.replace('_', ' ').title(),
            index=0
        )
        
        # Temperatures
        st.markdown("#### 🌡️ Temperatures")
        col3, col4 = st.columns(2)
        with col3:
            T_initial = st.number_input(
                "Initial Temp (°C)",
                min_value=0.0,
                max_value=200.0,
                value=100.0,
                step=1.0
            )
        with col4:
            T_final = st.number_input(
                "Final Temp (°C)",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=1.0
            )
        
        T_ambient = st.slider(
            "Ambient Temperature (°C)",
            min_value=-10,
            max_value=40,
            value=14,
            step=1
        )
        
        # Environment
        st.markdown("#### 🌬️ Environment")
        wind_speed = st.slider(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
            help="0 = calm air, 5 = light breeze, 10 = strong wind"
        )
        
        # Options
        st.markdown("#### 🎛️ Additional Options")
        lid_type = st.selectbox(
            "Lid Type",
            options=['tight', 'air_gap', 'none'],
            format_func=lambda x: {
                'tight': 'Tight Lid',
                'air_gap': 'Lid with Air Gap',
                'none': 'No Lid'
            }[x],
            index=1
        )
        
        ceramic_base = st.checkbox(
            "Ceramic/Insulated Base",
            value=True,
            help="Container placed on ceramic or insulated surface"
        )
        
        st.markdown("---")
        st.markdown("*Results update automatically*")
    
    # Always perform calculation with current parameters
    try:
        results = CoolingCalculator.calculate_cooling_time(
            diameter_cm=diameter,
            height_cm=height,
            material=material,
            liquid_type=liquid,
            volume_liters=volume,
            T_initial=T_initial,
            T_final=T_final,
            T_ambient=T_ambient,
            wind_speed=wind_speed,
            lid_type=lid_type,
            ceramic_base=ceramic_base
        )
        
        cooling_time = results['cooling_time_seconds']
        params = results['parameters']
        
        # Result display
        lid_text = {
            'tight': 'Tight Lid',
            'air_gap': 'Lid with Air Gap',
            'none': 'No Lid'
        }[lid_type]
        
        # Determine color based on cooling time
        hours = cooling_time / 3600
        if hours < 2:
            color = "#10b981"  # Green
        elif hours < 4:
            color = "#3B82F6"  # Blue
        elif hours < 6:
            color = "#f59e0b"  # Yellow
        else:
            color = "#ef4444"  # Red
        
        st.markdown(f"""
        <div class="result-box">
            <h2 style="color: #7FD3FF; margin-bottom: 1rem;">Cooling Time</h2>
            <h1 style="color: {color}; font-size: 3.5rem; margin: 0;">
                {format_time(cooling_time)}
            </h1>
            <p style="color: #94A3B8; font-size: 1.2rem; margin-top: 0.5rem;">
                {volume:.1f}L {liquid.title()} from {T_initial}°C to {T_final}°C
            </p>
            <p style="color: #64748b; font-size: 1rem;">
                {material.replace('_', ' ').title()} • {lid_text} • 
                {T_ambient}°C ambient • {'With' if ceramic_base else 'Without'} ceramic base
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Volume", f"{volume:.1f} L")
        with col2:
            st.metric("Cooling Rate", f"{params['cooling_rate']:.1f} °C/h")
        with col3:
            st.metric("ΔT", f"{T_initial - T_final}°C")
        with col4:
            heat_kj = params['thermal_mass'] * (T_initial - T_final) / 1000
            st.metric("Heat Loss", f"{heat_kj:.0f} kJ")
        
        # Cooling curve
        st.markdown("### 📈 Cooling Curve")
        fig = create_cooling_plot(
            results['cooling_curve']['times'],
            results['cooling_curve']['temperatures'],
            T_final,
            T_ambient
        )
        st.pyplot(fig)
        
        # Comparison with other conditions
        st.markdown("### 🔄 Comparison with Other Conditions")
        comparisons = []
        
        if lid_type != 'none':
            res_no_lid = CoolingCalculator.calculate_cooling_time(
                diameter, height, material, liquid, volume,
                T_initial, T_final, T_ambient, wind_speed,
                'none', ceramic_base
            )
            t_nl = res_no_lid['cooling_time_seconds']
            comparisons.append(("No Lid", format_time(t_nl),
                f"{(cooling_time - t_nl)/cooling_time*100:.0f}% faster"))
        
        if wind_speed < 2.0:
            res_wind = CoolingCalculator.calculate_cooling_time(
                diameter, height, material, liquid, volume,
                T_initial, T_final, T_ambient, 2.0,
                lid_type, ceramic_base
            )
            t_w = res_wind['cooling_time_seconds']
            comparisons.append(("Light Breeze (2 m/s)", format_time(t_w),
                f"{(cooling_time - t_w)/cooling_time*100:.0f}% faster"))
        
        if ceramic_base:
            res_no_ceramic = CoolingCalculator.calculate_cooling_time(
                diameter, height, material, liquid, volume,
                T_initial, T_final, T_ambient, wind_speed,
                lid_type, False
            )
            t_nc = res_no_ceramic['cooling_time_seconds']
            comparisons.append(("Without Ceramic Base", format_time(t_nc),
                f"{(cooling_time - t_nc)/cooling_time*100:.0f}% faster"))
        
        if material != 'aluminum':
            res_al = CoolingCalculator.calculate_cooling_time(
                diameter, height, 'aluminum', liquid, volume,
                T_initial, T_final, T_ambient, wind_speed,
                lid_type, ceramic_base
            )
            t_al = res_al['cooling_time_seconds']
            change = (t_al - cooling_time)/cooling_time*100
            comparisons.append(("Aluminum Container", format_time(t_al),
                f"{change:+.0f}%"))
        
        if comparisons:
            df = pd.DataFrame(comparisons, columns=['Condition', 'Time', 'Change'])
            st.dataframe(df, width='stretch', hide_index=True)
        
        # Detailed parameters in expander
        with st.expander("🔍 Show Detailed Parameters"):
            col_details1, col_details2 = st.columns(2)
            with col_details1:
                st.markdown("**Heat Transfer**")
                st.write(f"Heat transfer coefficient: {params['heat_transfer_coefficient']:.2f} W/m²K")
                st.write(f"Effective area: {params['effective_area']:.3f} m²")
                st.write(f"Cooling constant (k): {params['cooling_constant']:.6f} 1/s")
                
                st.markdown("**Container Geometry**")
                geometry = results['geometry']
                st.write(f"Diameter: {diameter} cm")
                st.write(f"Height: {height} cm")
                st.write(f"Liquid height: {geometry['height_liquid']*100:.1f} cm")
                st.write(f"Total surface area: {geometry['area_side'] + geometry['area_top'] + geometry['area_bottom']:.3f} m²")
            
            with col_details2:
                st.markdown("**Thermal Properties**")
                st.write(f"Liquid mass: {params['mass_liquid']:.2f} kg")
                st.write(f"Container mass: {params['mass_container']:.2f} kg")
                st.write(f"Total thermal mass: {params['thermal_mass']/1000:.1f} kJ/°C")
                
                st.markdown("**Material Properties**")
                mat_props = CoolingCalculator.MATERIALS[material]
                st.write(f"Density: {mat_props['density']} kg/m³")
                st.write(f"Specific heat: {mat_props['cp']} J/(kg·K)")
                st.write(f"Emissivity: {mat_props['emissivity']}")
    
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")

if __name__ == "__main__":
    main()