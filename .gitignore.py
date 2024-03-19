import streamlit as st

def main():
    st.set_page_config(page_title='Chat with multiple PDFs',page_icon=":books:")
    st.header("Chat with multiple PDFS :books:")
    st.text_input("Ask a question about your documents:")
    with st.sidebar:
        st.subheader("your documents")
        st.text_input("Upload your pdfs here and click on 'Process'")
if __name__ == '__main__':
    main()