import streamlit as st
import os
import tempfile
from frontend.utils.api_client import upload_pdf, get_papers, delete_paper

def render_sidebar():
    with st.sidebar:
        st.title("📚 RAG Assistant")
        st.markdown("Upload your research papers and start chatting.")
        
        st.divider()
        
        # Upload Section
        st.subheader("Upload Paper")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Index Paper", use_container_width=True):
                with st.spinner("Processing PDF (Parsing, Chunking, Embedding)..."):
                    # Save to temp file to send via requests
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                        
                    try:
                        result = upload_pdf(tmp_path)
                        st.success(f"Successfully indexed: {result['title']}")
                        st.session_state["papers_updated"] = True
                    except Exception as e:
                        st.error(f"Error uploading paper: {e}")
                    finally:
                        os.unlink(tmp_path)
        
        st.divider()
        
        # Papers List
        st.subheader("Indexed Papers")
        
        try:
            papers = get_papers()
            if not papers:
                st.info("No papers indexed yet.")
            else:
                for paper in papers:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.caption(f"📄 {paper['title']}")
                    with col2:
                        if st.button("🗑️", key=f"del_{paper['paper_id']}", help="Delete Paper"):
                            try:
                                delete_paper(paper['paper_id'])
                                st.session_state["papers_updated"] = True
                                st.rerun()
                            except Exception as e:
                                st.error("Failed to delete.")
        except Exception as e:
            st.error(f"Could not load papers: {e}")
