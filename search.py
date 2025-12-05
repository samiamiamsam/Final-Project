import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss


def chunk_text(text: str, size: int = 200, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


class Index:
    
    def __init__(self):
        self.docs: List[Dict] = []
        self.chunks: List[str] = []
        self.doc_map: List[tuple] = []
        self.bm25: BM25Okapi | None = None
        try:
            self.embedder = SentenceTransformer("all-mpnet-base-v2")
            print("Using all-mpnet-base-v2 BERT model for enhanced semantic search")
        except Exception as e:
            print(f"Could not load all-mpnet-base-v2, using all-MiniLM-L6-v2: {e}")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index: faiss.IndexFlatIP | None = None
        self.valid_chunk_indices: List[int] = []
        self.bert_embeddings: np.ndarray | None = None

    def add_document(self, doc_id: str, name: str, text: str) -> bool:
        if not text or not text.strip():
            return False
        chunks = chunk_text(text)
        if not chunks:
            return False
        start = len(self.chunks)
        self.chunks.extend(chunks)
        self.docs.append({"id": doc_id, "name": name, "text": text, "chunks": chunks})
        self.doc_map.append((len(self.docs) - 1, start, start + len(chunks)))
        return True
    
    def add_documents_batch(self, documents: List[Dict[str, str]]) -> int:
        added_count = 0
        for doc in documents:
            if self.add_document(doc["id"], doc["name"], doc["text"]):
                added_count += 1
        if added_count > 0:
            self._rebuild()
        return added_count

    def _rebuild(self):
        if not self.chunks:
            self.bm25 = None
            self.index = None
            return
        
        tokenized = []
        valid_chunk_indices = []
        for i, c in enumerate(self.chunks):
            if c and c.strip():
                tokens = c.lower().split()
                if tokens:
                    tokenized.append(tokens)
                    valid_chunk_indices.append(i)
        
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)
            self.valid_chunk_indices = valid_chunk_indices
        else:
            self.bm25 = None
            self.valid_chunk_indices = []

        if valid_chunk_indices:
            try:
                valid_chunks = [self.chunks[i] for i in valid_chunk_indices]
                emb = self.embedder.encode(
                    valid_chunks, 
                    normalize_embeddings=True, 
                    show_progress_bar=True,
                    batch_size=32,
                    convert_to_numpy=True
                )
                emb = np.array(emb).astype('float32')
                self.bert_embeddings = emb
                
                if emb.size > 0:
                    dim = emb.shape[1]
                    idx = faiss.IndexFlatIP(dim)
                    idx.add(emb)
                    self.index = idx
                    print(f"BERT index built: {len(valid_chunks)} chunks, {dim} dimensions")
                else:
                    self.index = None
                    self.bert_embeddings = None
            except Exception as e:
                print(f"Error building BERT/FAISS index: {e}")
                import traceback
                traceback.print_exc()
                self.index = None
                self.bert_embeddings = None
        else:
            self.index = None
            self.bert_embeddings = None

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not query or not query.strip():
            return []
        
        if not self.chunks or len(self.chunks) == 0:
            return []
        
        if not self.bm25 or self.index is None:
            return []

        try:
            q_tokens = query.lower().split()
            if not q_tokens:
                return []
            
            bm25_scores = self.bm25.get_scores(q_tokens)
            bm25_top = np.argsort(bm25_scores)[::-1][:min(100, len(bm25_scores))]

            q_emb = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False).astype('float32')
            search_k = min(top_k * 5, self.index.ntotal) if self.index.ntotal > 0 else 0
            bert_candidates = []
            bert_similarities = {}
            
            if search_k > 0:
                D, I = self.index.search(q_emb, search_k)
                for dist, idx in zip(D[0], I[0]):
                    if idx < len(self.valid_chunk_indices):
                        chunk_idx = self.valid_chunk_indices[idx]
                        bert_candidates.append(chunk_idx)
                        bert_similarities[chunk_idx] = float(dist)

            candidate_chunks = set()
            for bm25_idx in bm25_top[:50]:
                if bm25_idx < len(self.valid_chunk_indices):
                    candidate_chunks.add(self.valid_chunk_indices[bm25_idx])
            for chunk_idx in bert_candidates[:50]:
                candidate_chunks.add(chunk_idx)
            
            if candidate_chunks and self.bert_embeddings is not None:
                candidate_list = list(candidate_chunks)
                chunk_to_emb_idx = {chunk_idx: i for i, chunk_idx in enumerate(self.valid_chunk_indices)}
                
                candidate_embeddings_list = []
                valid_candidate_list = []
                for chunk_idx in candidate_list:
                    if chunk_idx in chunk_to_emb_idx:
                        emb_idx = chunk_to_emb_idx[chunk_idx]
                        if emb_idx < len(self.bert_embeddings):
                            candidate_embeddings_list.append(self.bert_embeddings[emb_idx])
                            valid_candidate_list.append(chunk_idx)
                
                if len(candidate_embeddings_list) > 0:
                    candidate_embeddings = np.array(candidate_embeddings_list)
                    bert_similarities_explicit = np.dot(candidate_embeddings, q_emb[0])
                    for chunk_idx, sim_score in zip(valid_candidate_list, bert_similarities_explicit):
                        if chunk_idx not in bert_similarities or sim_score > bert_similarities[chunk_idx]:
                            bert_similarities[chunk_idx] = float(sim_score)

            fused = {}
            
            for rank, bm25_idx in enumerate(bm25_top):
                if bm25_idx < len(self.valid_chunk_indices):
                    chunk_idx = self.valid_chunk_indices[bm25_idx]
                    normalized_bm25 = bm25_scores[bm25_idx] / (max(bm25_scores) + 1e-8) if max(bm25_scores) > 0 else 0
                    fused[chunk_idx] = fused.get(chunk_idx, 0) + 0.4 * normalized_bm25 + 0.1 / (rank + 1)
            
            for chunk_idx, bert_score in bert_similarities.items():
                fused[chunk_idx] = fused.get(chunk_idx, 0) + 0.6 * max(0, bert_score)

            if not fused:
                return []

            ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k * 3]

            results = []
            for chunk_idx, score in ranked:
                if chunk_idx >= len(self.chunks):
                    continue
                chunk = self.chunks[chunk_idx]
                doc = None
                for doc_idx, s, e in self.doc_map:
                    if s <= chunk_idx < e:
                        doc = self.docs[doc_idx]
                        break
                if doc:
                    results.append({
                        "filename": doc["name"],
                        "score": round(score, 4),
                        "snippet": chunk[:250] + ("..." if len(chunk) > 250 else "")
                    })  #for HTML display
            
            filename_to_best_result = {}
            for result in results:
                filename = result["filename"]
                if filename not in filename_to_best_result:
                    filename_to_best_result[filename] = result
                else:
                    if result["score"] > filename_to_best_result[filename]["score"]:
                        filename_to_best_result[filename] = result
            
            deduplicated_results = list(filename_to_best_result.values())
            deduplicated_results.sort(key=lambda x: x["score"], reverse=True)
            return deduplicated_results[:top_k]
        except Exception as e:
            print(f"Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

