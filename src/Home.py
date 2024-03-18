import streamlit as st

st.set_page_config(
    page_title="World Constitutions AI Assistant",
    page_icon="\U0001F4D6",
)

background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.unsplash.com/photo-1562504208-03d85cc8c23e?q=80&w=2340&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: 85vw 110vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)
st.header("World Constitutions AI Assistant")
st.markdown("#### Explore the Constitutions of the World!")

with st.sidebar:
    all_countries = ["Italy", "France"]

    selected_country = st.selectbox("Select country:", all_countries)



