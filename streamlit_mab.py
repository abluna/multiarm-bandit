import pandas as pd
import numpy as np
import streamlit as st
import mab
import sys
import itertools

st.write("""
# Multi-Arm Bandit Campaign Simulator
""")

test_message = mab.mab_test()
if st.button('Test App'):
    with st.spinner('Wait for it...'):
        st.write(test_message)
