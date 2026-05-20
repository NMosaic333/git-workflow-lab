import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class QueryLog(Base):
    __tablename__ = 'query_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    query = Column(String, nullable=False)
    mode = Column(String, nullable=False)
    retrieval_latency = Column(Float, default=0.0)
    generation_latency = Column(Float, default=0.0)
    total_latency = Column(Float, default=0.0)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    num_chunks_retrieved = Column(Integer, default=0)

class EvaluationTracker:
    def __init__(self, db_path: str = "./data/evaluation/metrics.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # SQLite URL
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(db_url)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        # Session factory
        self.Session = sessionmaker(bind=self.engine)
        
    def log_query(self, query: str, mode: str, retrieval_latency: float, generation_latency: float, input_tokens: int, output_tokens: int, num_chunks: int):
        total_latency = retrieval_latency + generation_latency
        
        session = self.Session()
        try:
            log_entry = QueryLog(
                query=query,
                mode=mode,
                retrieval_latency=retrieval_latency,
                generation_latency=generation_latency,
                total_latency=total_latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                num_chunks_retrieved=num_chunks
            )
            session.add(log_entry)
            session.commit()
        finally:
            session.close()

    def get_metrics(self):
        """Retrieve aggregated metrics for the dashboard."""
        session = self.Session()
        try:
            logs = session.query(QueryLog).all()
            if not logs:
                return {
                    "total_queries": 0,
                    "avg_total_latency": 0.0,
                    "avg_retrieval_latency": 0.0,
                    "avg_generation_latency": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0
                }
                
            total_queries = len(logs)
            avg_total_latency = sum(log.total_latency for log in logs) / total_queries
            avg_retrieval_latency = sum(log.retrieval_latency for log in logs) / total_queries
            avg_generation_latency = sum(log.generation_latency for log in logs) / total_queries
            total_input_tokens = sum(log.input_tokens for log in logs)
            total_output_tokens = sum(log.output_tokens for log in logs)
            
            return {
                "total_queries": total_queries,
                "avg_total_latency": avg_total_latency,
                "avg_retrieval_latency": avg_retrieval_latency,
                "avg_generation_latency": avg_generation_latency,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens
            }
        finally:
            session.close()
