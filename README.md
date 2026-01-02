# ❄️ Cooling Time Calculator

A physics-based web application that calculates accurate cooling times for liquids based on real thermodynamic principles.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)

## 🌟 Features

- **Real Physics Model**: Accurate calculations based on thermodynamics principles
- **Multiple Cooling Mechanisms**: Convection, radiation, and evaporation
- **Material Properties**: Different container materials with realistic conductivity
- **Environmental Factors**: Ambient temperature, wind speed, humidity
- **Lid Effect**: Account for containers with/without lids
- **Interactive Visualization**: Real-time cooling curves and rates
- **Data Export**: Download results as CSV or summary text

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/andgrim/cooling-time-calculator.git
cd cooling-time-calculator

    Create virtual environment (optional but recommended)

bash

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

    Install dependencies

bash

pip install -r requirements.txt

    Run the application

bash

streamlit run app.py

    Open your browser and go to http://localhost:8501

📊 Physics Model

The calculator uses a comprehensive physics model including:
Heat Transfer Mechanisms

    Convection (Natural & Forced)

        Realistic convection coefficients (6-45 W/m²K)

        Wind effect on forced convection

        Material-dependent heat transfer

    Radiation

        Stefan-Boltzmann law

        Emissivity factor (ε = 0.9)

    Evaporation

        Only for open containers

        Humidity-dependent

        Wind-enhanced evaporation

Material Properties

    Excellent conductors: Copper, Aluminum

    Good conductors: Stainless Steel

    Poor conductors: Glass, Ceramic

    Insulators: Plastic

🎯 Usage Examples
Typical Scenarios

    Tea/Coffee Cooling: 1L from 100°C to 60°C (drinking temperature)

    Soup Storage: 3L from 90°C to 25°C (room temperature)

    Cooking Oil: 2L from 180°C to 40°C (safe handling)

Realistic Parameters

    Wind speed: 0 m/s (still air) to 10 m/s (strong wind)

    Humidity: 10% (dry) to 90% (humid)

    Ambient temperature: 0°C to 40°C

📈 Output Features

    Main Result: Total cooling time in hours and minutes

    Cooling Curve: Temperature vs. time graph

    Cooling Rate: Instantaneous cooling rate analysis

    Heat Transfer Metrics: Total heat dissipated, mass, ΔT

    Comparison Table: Alternative scenarios comparison

    Export Options: CSV data and text summary

⚠️ Safety Information
Important Notes

    Do NOT put hot liquids directly in the refrigerator

        Can raise refrigerator temperature

        May spoil other foods

        Wastes energy and reduces efficiency

        Can cause thermal shock to containers