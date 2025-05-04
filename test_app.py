import streamlit as st

st.title("Ứng dụng Streamlit đơn giản")
st.write("Xin chào từ Streamlit!")

name = st.text_input("Nhập tên của bạn:")
if name:
    st.write(f"Chào bạn, {name}!")