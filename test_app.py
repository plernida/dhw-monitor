import streamlit as st

st.title("🌊 DHW Monitor - Test")
st.write("If you see this, deployment works!")

if st.button("Generate Demo"):
    import numpy as np
    import plotly.graph_objects as go
    
    # Simple test plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    fig = go.Figure(data=go.Scatter(x=x, y=y))
    st.plotly_chart(fig)
    st.success("✅ All dependencies working!")
