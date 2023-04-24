import streamlit as st
import plotly


fig = plotly.io.read_json("plots/json/channels_data_scatter_matrix.json")

st.title("Channels data dashboard ")
st.sidebar.title("navigation")
st.plotly_chart(fig)