import streamlit as st
from frontend.utils.api_client import query_assistant, get_papers

def render_chat_interface():
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Filter section
    col1, col2 = st.columns([1, 1])
    with col1:
        mode = st.selectbox(
            "Research Mode",
            options=["General", "Beginner", "Methodology", "Limitations", "Comparison"],
            index=0
        )
    with col2:
        try:
            papers = get_papers()
            paper_options = {"All Papers": None}
            for p in papers:
                paper_options[p['title']] = p['paper_id']
                
            selected_paper_title = st.selectbox("Filter by Paper", options=list(paper_options.keys()))
            selected_paper_id = paper_options[selected_paper_title]
        except Exception:
            selected_paper_id = None
            st.selectbox("Filter by Paper", options=["All Papers"])

    st.divider()

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("View Retrieved Sources"):
                    for idx, source in enumerate(msg["sources"]):
                        meta = source.get("metadata", {})
                        title = meta.get("paper_title", "Unknown")
                        page = meta.get("page_number", "?")
                        score = source.get("score", source.get("hybrid_score", 0.0))
                        
                        st.markdown(f"""
                        <div class="source-chunk-box">
                            <strong>Source {idx + 1}: {title} (Page {page})</strong><br>
                            <em>Relevance Score: {score:.4f}</em><br>
                            {source.get('text', '')[:300]}...
                        </div>
                        """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about your papers..."):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response with loading spinner
        with st.chat_message("assistant"):
            with st.spinner("Analyzing papers..."):
                try:
                    filter_ids = [selected_paper_id] if selected_paper_id else None
                    response = query_assistant(prompt, mode=mode.lower(), filter_paper_ids=filter_ids)
                    
                    answer = response.get("answer", "No answer generated.")
                    sources = response.get("source_chunks", [])
                    metrics = response.get("metrics", {})
                    
                    st.markdown(answer)
                    
                    st.caption(f"⏱️ Retrieval: {metrics.get('retrieval_latency_ms', 0)}ms | Generation: {metrics.get('generation_latency_ms', 0)}ms")
                    
                    if sources:
                        with st.expander("View Retrieved Sources"):
                            for idx, source in enumerate(sources):
                                meta = source.get("metadata", {})
                                title = meta.get("paper_title", "Unknown")
                                page = meta.get("page_number", "?")
                                score = source.get("score", source.get("hybrid_score", 0.0))
                                
                                st.markdown(f"""
                                <div class="source-chunk-box">
                                    <strong>Source {idx + 1}: {title} (Page {page})</strong><br>
                                    <em>Relevance Score: {score:.4f}</em><br>
                                    {source.get('text', '')[:300]}...
                                </div>
                                """, unsafe_allow_html=True)
                                
                    # Add assistant message to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error querying assistant: {e}")
