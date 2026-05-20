import streamlit as st
from frontend.utils.api_client import get_metrics

def render_evaluation_page():
    st.title("📊 Evaluation Dashboard")
    st.markdown("Monitor the performance, latency, and token usage of the RAG pipeline.")
    
    try:
        metrics = get_metrics()
        
        st.subheader("High-Level Metrics")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Total Queries", f"{metrics.get('total_queries', 0)}")
        col2.metric("Total Input Tokens", f"{metrics.get('total_input_tokens', 0)}")
        col3.metric("Total Output Tokens", f"{metrics.get('total_output_tokens', 0)}")
        
        st.divider()
        
        st.subheader("Average Latency (seconds)")
        col4, col5, col6 = st.columns(3)
        
        avg_total = metrics.get('avg_total_latency', 0)
        avg_retrieval = metrics.get('avg_retrieval_latency', 0)
        avg_gen = metrics.get('avg_generation_latency', 0)
        
        col4.metric("Total Latency", f"{avg_total:.2f} s")
        col5.metric("Retrieval Latency", f"{avg_retrieval:.2f} s")
        col6.metric("Generation Latency", f"{avg_gen:.2f} s")
        
        if avg_total > 0:
            # Simple bar chart for latency breakdown
            st.markdown("### Latency Breakdown")
            chart_data = {
                "Stage": ["Retrieval", "Generation"],
                "Latency (s)": [avg_retrieval, avg_gen]
            }
            st.bar_chart(data=chart_data, x="Stage", y="Latency (s)", use_container_width=True)
            
    except Exception as e:
        st.error(f"Could not load metrics: {e}")
