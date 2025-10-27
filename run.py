# filename: run.py
import argparse, os, sys, queue, re, threading, time, json, math, glob
from dataclasses import dataclass
from typing import Generator, Optional, Tuple, List, Dict
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None  # graceful headless mode
import soundfile as sf
from scipy.signal import resample_poly
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer,
    AutoModel, AutoModelForSequenceClassification
)

# Optional deps
try:
    import faiss  # pip install faiss-cpu (หรือ faiss-gpu)
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

try:
    import yaml
except Exception:
    yaml = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# OpenVoice V2
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS  # base TTS

# NLTK data in project
os.environ.setdefault("NLTK_DATA", os.path.abspath("nltk_data"))
os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
LOCAL_ONLY = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"

# ------------------ Utils ------------------
def ensure_sr(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr: return audio
    g = math.gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    return resample_poly(audio, up, down).astype(np.float32)

def verify_local_model_dir(path: str):
    import datetime
    if not os.path.isdir(path):
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์โมเดล: {path}")
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(f"ไฟล์สำคัญหายไปใน {path}: {missing}")
    ms = os.path.getsize(os.path.join(path, "model.safetensors"))
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(path, "model.safetensors")))
    print(f"[MODEL CHECK] model.safetensors = {ms/1e6:.1f} MB, mtime = {mtime}")
    extras = ["added_tokens.json", "chat_template.jinja", "special_tokens_map.json"]
    present = [e for e in extras if os.path.exists(os.path.join(path, e))]
    if present: print(f"[MODEL CHECK] extras present: {present}")

@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen3-14B"
    device_map: str = "auto"
    enable_thinking: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 20
    lora_path: Optional[str] = None

class SentenceChunker:
    def __init__(self, min_chars: int = 40):
        self.buf = []; self.min_chars = min_chars
        self.re_end = re.compile(r'[\.!\?…\n]+["”\']?\s*$')
    def push(self, piece: str) -> Optional[str]:
        self.buf.append(piece); s = "".join(self.buf)
        if len(s) >= self.min_chars and self.re_end.search(s):
            self.buf = []; return s.strip()
        return None
    def flush(self) -> Optional[str]:
        if not self.buf: return None
        s = "".join(self.buf).strip(); self.buf = []
        return s or None

def detect_ckpt_root(root_hint="checkpoints_v2") -> str:
    candidates = [root_hint, os.path.join(root_hint, "checkpoints_v2")]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "converter")):
            return c
    raise FileNotFoundError("ไม่พบโฟลเดอร์ checkpoints_v2 (ต้องมี converter/)")

def ensure_nltk_data():
    import nltk
    from nltk.data import find
    needed = [
        ("taggers","averaged_perceptron_tagger_eng"),
        ("taggers","averaged_perceptron_tagger"),
        ("tokenizers","punkt"),
    ]
    for kind,name in needed:
        try: find(f"{kind}/{name}")
        except LookupError: nltk.download(name, quiet=True)

# ------------------ User Memory (unchanged core) ------------------
class UserMemory:
    def __init__(self, mem_dir="memory", custom_data_path="data/sft_samples.jsonl"):
        self.mem_dir = mem_dir
        self.custom_data_path = custom_data_path
        os.makedirs(self.mem_dir, exist_ok=True)
        self.profile_path = os.path.join(self.mem_dir, "profile.yaml")
        self.facts_path = os.path.join(self.mem_dir, "facts.json")
        self.profile = {}
        self.facts: List[str] = []
        self.load(); self.load_custom_data()

    def load(self):
        if os.path.exists(self.profile_path) and yaml:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                self.profile = yaml.safe_load(f) or {}
        else:
            self.profile = {}
        if os.path.exists(self.facts_path):
            with open(self.facts_path, "r", encoding="utf-8") as f:
                try: self.facts = json.load(f) or []
                except Exception: self.facts = []
        else:
            self.facts = []

    def load_custom_data(self):
        p = self.custom_data_path
        if not os.path.exists(p):
            print(f"[MEMORY] Custom data file not found: {p}"); return
        try:
            added = 0
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line or line.startswith("//"): continue
                    try:
                        sample = json.loads(line)
                        msgs = sample.get("messages", [])
                        for i in range(len(msgs)-1):
                            if msgs[i].get("role")=="user" and msgs[i+1].get("role")=="assistant":
                                q=msgs[i]["content"]; a=msgs[i+1]["content"]
                                self.facts.append(f"Q: {q} A: {a}"); added+=1
                    except json.JSONDecodeError:
                        continue
            print(f"[MEMORY] Loaded {added} QA pairs")
        except Exception as e:
            print(f"[MEMORY] Error loading custom data: {e}")

    def save(self):
        if yaml is None: raise RuntimeError("ต้องการ pyyaml: pip install pyyaml")
        with open(self.profile_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.profile, f, allow_unicode=True, sort_keys=False)
        with open(self.facts_path, "w", encoding="utf-8") as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)

    def set_profile(self, key: str, value: str): self.profile[key]=value
    def add_fact(self, text: str):
        t=text.strip()
        if t: self.facts.append(t)

    def forget_fact(self, idx_or_substr: str) -> Optional[str]:
        if idx_or_substr.isdigit():
            i=int(idx_or_substr)-1
            if 0<=i<len(self.facts): return self.facts.pop(i)
            return None
        for i, fact in enumerate(self.facts):
            if idx_or_substr.lower() in fact.lower():
                return self.facts.pop(i)
        return None

    def summarize_profile(self)->str:
        if not self.profile: return ""
        return "\n".join([f"{k}: {v}" for k,v in self.profile.items()])

    def relevant_facts(self, query: str, k: int = 6) -> List[str]:
        if not self.facts: return []
        toks_q = set(word_tokenize(query.lower()))
        scored=[]
        for fact in self.facts:
            toks_f=set(word_tokenize(fact.lower()))
            score=len(toks_q & toks_f)
            scored.append((score,fact))
        scored.sort(key=lambda x:x[0], reverse=True)
        out=[f for s,f in scored if s>0][:k]
        if not out: out=[f for _,f in scored[:min(3,len(scored))]]
        return out

    def to_system_context(self, last_user_text: str) -> str:
        prof=self.summarize_profile(); rel=self.relevant_facts(last_user_text, k=6)
        ctx=[]
        if prof: ctx.append("[USER PROFILE]\n"+prof)
        if rel: ctx.append("[RELEVANT FACTS]\n- "+"\n- ".join(rel))
        return "\n".join(ctx)

# ------------------ RAG: Ingest / Embed / Search / Rerank ------------------
def read_text_from_file(path:str)->str:
    ext=os.path.splitext(path)[1].lower()
    try:
        if ext in [".md",".txt"]:
            return open(path,"r",encoding="utf-8", errors="ignore").read()
        if ext in [".json"]:
            data=json.load(open(path,"r",encoding="utf-8", errors="ignore"))
            # heuristic fields
            for key in ["text","content","body","article"]:
                if key in data and isinstance(data[key], str):
                    return data[key]
            return json.dumps(data, ensure_ascii=False)
        if ext in [".jsonl",".ndjson"]:
            lines=[]
            with open(path,"r",encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    ln=ln.strip()
                    if not ln: continue
                    try:
                        obj=json.loads(ln)
                        for key in ["text","content","body"]:
                            if key in obj and isinstance(obj[key],str):
                                lines.append(obj[key]); break
                        else:
                            lines.append(ln)
                    except Exception:
                        lines.append(ln)
            return "\n".join(lines)
        if ext in [".yaml",".yml"] and yaml:
            data=yaml.safe_load(open(path,"r",encoding="utf-8", errors="ignore"))
            return json.dumps(data, ensure_ascii=False)
        if ext==".pdf" and PdfReader:
            reader=PdfReader(path)
            pages=[p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
    except Exception as e:
        print(f"[INGEST] read error {path}: {e}")
    return ""

def smart_sections(text:str)->List[str]:
    # แบ่งตามหัวข้อ markdown เป็นกลุ่มใหญ่ก่อน
    parts=[]
    buf=[]
    for line in text.splitlines():
        if re.match(r'^\s*#{1,6}\s+\S', line):
            if buf: parts.append("\n".join(buf).strip()); buf=[]
        buf.append(line)
    if buf: parts.append("\n".join(buf).strip())
    # ถ้าไม่มีหัวข้อเลย คืนทั้งก้อน
    return parts if parts else [text]

def token_len(tokens_or_text, tokenizer)->int:
    if isinstance(tokens_or_text, str):
        return len(tokenizer(tokens_or_text, add_special_tokens=False)["input_ids"])
    return len(tokens_or_text)

def chunk_text(text:str, tokenizer, target_tokens=400, overlap=50)->List[str]:
    sections=smart_sections(text)
    chunks=[]
    for sec in sections:
        ids=tokenizer(sec, add_special_tokens=False)["input_ids"]
        if len(ids)<=target_tokens:
            chunks.append(sec); continue
        start=0
        while start < len(ids):
            end=min(len(ids), start+target_tokens)
            seg_ids=ids[start:end]
            seg_text=tokenizer.decode(seg_ids, skip_special_tokens=True)
            chunks.append(seg_text.strip())
            if end==len(ids): break
            start = max(end-overlap, 0)
    return [c for c in chunks if c.strip()]

class HFTextEmbedding:
    def __init__(self, model_name:str, device:str="cpu", local_only:bool=True):
        try:
            self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_only)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=local_only).to(device)
            self.model.eval()
            self.device = device
            
            # Performance optimizations
            self._embedding_cache = {}  # Cache for embeddings
            self._max_cache_size = 1000
            self._executor = ThreadPoolExecutor(max_workers=2)  # For parallel processing
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model from {model_name}: {e}")
            print("[ERROR] Please ensure the embedding model is downloaded and available at the specified path")
            print("[ERROR] You can download it using: git clone https://huggingface.co/BAAI/bge-small-en-v1.5")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    @torch.no_grad()
    def encode(self, texts:List[str], batch_size:int=16)->np.ndarray:
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                cached_embeddings.append((i, self._embedding_cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts
        outs = []
        if uncached_texts:
            # Process in batches
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i+batch_size]
                inputs = self.tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                out = self.model(**inputs)
                # mean pooling
                last = out.last_hidden_state  # [B, T, H]
                mask = inputs["attention_mask"].unsqueeze(-1)
                emb = (last*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                batch_embeddings = emb.cpu().float().numpy()
                
                # Update cache
                for j, text in enumerate(batch):
                    cache_key = self._get_cache_key(text)
                    if len(self._embedding_cache) < self._max_cache_size:
                        self._embedding_cache[cache_key] = batch_embeddings[j]
                
                outs.append(batch_embeddings)
        
        # Combine cached and new embeddings
        all_embeddings = np.zeros((len(texts), outs[0].shape[1] if outs else cached_embeddings[0][1].shape[0]))
        
        # Fill in cached embeddings
        for idx, emb in cached_embeddings:
            all_embeddings[idx] = emb
        
        # Fill in new embeddings
        if outs:
            new_embeddings = np.vstack(outs)
            for i, idx in enumerate(uncached_indices):
                all_embeddings[idx] = new_embeddings[i]
        
        return all_embeddings

class CrossEncoderReranker:
    def __init__(self, model_name:str, device:str="cpu", local_only:bool=True):
        self.tok=AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_only).to(device)
        self.model.eval(); self.device=device
        
        # Performance optimizations
        self._score_cache = {}  # Cache for reranking scores
        self._max_cache_size = 500
    
    def _get_cache_key(self, query: str, passage: str) -> str:
        """Generate cache key for query-passage pair"""
        combined = f"{query.lower().strip()}|||{passage.lower().strip()}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    @torch.no_grad()
    def score(self, query:str, passages:List[str], batch_size:int=8)->List[float]:
        # Check cache first
        cached_scores = {}
        uncached_passages = []
        uncached_indices = []
        
        for i, passage in enumerate(passages):
            cache_key = self._get_cache_key(query, passage)
            if cache_key in self._score_cache:
                cached_scores[i] = self._score_cache[cache_key]
            else:
                uncached_passages.append(passage)
                uncached_indices.append(i)
        
        # Process uncached passages
        scores = [0.0] * len(passages)
        
        # Fill in cached scores
        for idx, score in cached_scores.items():
            scores[idx] = score
        
        # Process uncached passages in batches
        if uncached_passages:
            for i in range(0, len(uncached_passages), batch_size):
                batch = uncached_passages[i:i+batch_size]
                enc = self.tok([ (query, p) for p in batch ], padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
                logits = self.model(**enc).logits.squeeze(-1)
                batch_scores = logits.detach().cpu().float().tolist()
                
                # Update cache and scores
                for j, passage in enumerate(batch):
                    cache_key = self._get_cache_key(query, passage)
                    if len(self._score_cache) < self._max_cache_size:
                        self._score_cache[cache_key] = batch_scores[j]
                    scores[uncached_indices[i+j]] = batch_scores[j]
        
        return scores

class VectorStore:
    def __init__(self, dim:int, use_faiss:bool=True):
        self.dim=dim
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.emb=None
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index=None
        # Cache for embeddings to avoid repeated disk reads
        self._embeddings_cache = None
        self._cache_loaded = False
    
    def add(self, X:np.ndarray):
        X = X.astype(np.float32)
        if self.use_faiss:
            self.index.add(X)
        else:
            self.emb = X if self.emb is None else np.vstack([self.emb, X])
        # Update cache when adding embeddings
        self._embeddings_cache = X
        self._cache_loaded = True
    
    def get_embeddings(self, embeddings_path: Optional[str] = None) -> np.ndarray:
        """Get embeddings with caching to avoid repeated disk reads"""
        if self.use_faiss and embeddings_path and not self._cache_loaded:
            self._embeddings_cache = np.load(embeddings_path)
            self._cache_loaded = True
        return self._embeddings_cache if self._embeddings_cache is not None else self.emb
    
    def search(self, q:np.ndarray, topk:int=8)->Tuple[np.ndarray, np.ndarray]:
        q = q.astype(np.float32)
        if self.use_faiss:
            D,I = self.index.search(q, topk)
            return D,I
        # brute force cosine (dot because normalized)
        embeddings = self.emb if self.emb is not None else self._embeddings_cache
        if embeddings is None:
            return np.array([]).reshape(1, 0), np.array([]).reshape(1, 0).astype(int)
        sims = embeddings @ q[0]
        idx = np.argsort(-sims)[:topk]
        return sims[idx][None, :], idx[None, :]

class RAGPipeline:
    def __init__(self, kb_dir:str, embed_model:str, device:str, local_only:bool,
                 index_dir="rag_index", target_tokens=400, overlap=50):
        self.kb_dir=kb_dir
        self.embed_model_id=embed_model
        self.index_dir=index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.embed = HFTextEmbedding(embed_model, device=device, local_only=local_only)
        self.tokenizer = self.embed.tok
        self.dim = self.embed.model.config.hidden_size
        self.target_tokens=target_tokens; self.overlap=overlap

        self.doc_texts: List[str]=[]
        self.doc_meta: List[Dict]=[]
        self.store: Optional[VectorStore]=None
        self._docs_cache = None  # Cache for docs.jsonl
        self._query_cache = {}   # Cache for query embeddings

    def _scan_files(self)->List[str]:
        exts = ["*.md","*.txt","*.json","*.jsonl","*.yaml","*.yml"]
        paths=[]
        for p in exts: paths += glob.glob(os.path.join(self.kb_dir, "**", p), recursive=True)
        if PdfReader: paths += glob.glob(os.path.join(self.kb_dir, "**", "*.pdf"), recursive=True)
        return sorted(set(paths))

    def build_or_load(self, rebuild:bool=False):
        meta_path=os.path.join(self.index_dir,"meta.json")
        vec_path=os.path.join(self.index_dir,"embeddings.npy")
        txt_path=os.path.join(self.index_dir,"docs.jsonl")
        faiss_path=os.path.join(self.index_dir,"index.faiss")

        can_load = (not rebuild) and os.path.exists(meta_path) and os.path.exists(vec_path) and os.path.exists(txt_path) and (os.path.exists(faiss_path) or True)
        if can_load:
            meta=json.load(open(meta_path,"r",encoding="utf-8"))
            if meta.get("embed_model")==self.embed_model_id:
                self.doc_texts=[]
                self.doc_meta=[]
                with open(txt_path,"r",encoding="utf-8") as f:
                    for line in f:
                        self.doc_meta.append(json.loads(line))
                X=np.load(vec_path)
                self.store=VectorStore(dim=X.shape[1], use_faiss=(faiss is not None))
                if self.store.use_faiss and os.path.exists(faiss_path):
                    self.store.index = faiss.read_index(faiss_path)
                else:
                    self.store.emb = X
                print(f"[RAG] Loaded index: {X.shape[0]} chunks, dim={X.shape[1]}, faiss={self.store.use_faiss}")
                return
            else:
                print("[RAG] embed model changed → rebuilding index")

        print("[RAG] Building index …")
        files=self._scan_files()
        records=[]
        texts=[]
        for path in files:
            txt=read_text_from_file(path)
            if not txt.strip(): continue
            chunks=chunk_text(txt, self.tokenizer, self.target_tokens, self.overlap)
            title=os.path.basename(path)
            for i, ch in enumerate(chunks):
                records.append({"path": path, "title": title, "chunk_id": i, "chars": len(ch), "text": ch})
                texts.append(ch)
        if not texts:
            print("[RAG] No documents found."); texts=[]; records=[]
        X = self.embed.encode(texts) if texts else np.zeros((0,self.dim),dtype=np.float32)
        self.store=VectorStore(dim=self.dim, use_faiss=(faiss is not None))
        if self.store.use_faiss and X.shape[0]>0:
            self.store.index.add(X)
            faiss.write_index(self.store.index, faiss_path)
        else:
            self.store.emb = X

        with open(txt_path,"w",encoding="utf-8") as f:
            for r in records: f.write(json.dumps(r, ensure_ascii=False)+"\n")
        np.save(vec_path, X)
        json.dump({"embed_model": self.embed_model_id}, open(meta_path,"w",encoding="utf-8"))
        self.doc_meta=records
        print(f"[RAG] Built index: {X.shape[0]} chunks")

    def retrieve(self, query:str, topk:int=8, mmr_lambda:float=0.5)->List[Dict]:
        if self.store is None or (self.store.use_faiss and (self.store.index is None)):
            return []
        
        # Check cache for query embedding
        query_hash = hash(query.lower().strip())
        if query_hash in self._query_cache:
            q_emb = self._query_cache[query_hash]
        else:
            q_emb = self.embed.encode([query])
            # Limit cache size to prevent memory issues
            if len(self._query_cache) > 100:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]
            self._query_cache[query_hash] = q_emb
        D,I=self.store.search(q_emb, topk=topk*3)  # oversample before MMR
        I=I[0].tolist(); sims=D[0].tolist()
        
        # Get embeddings from cache to avoid repeated disk reads
        embeddings_path = os.path.join(self.index_dir,"embeddings.npy") if self.store.use_faiss else None
        X = self.store.get_embeddings(embeddings_path)
        
        # MMR diversity optimization - reduce computational complexity
        selected=[]
        cand=set(I)
        if not len(cand): return []
        
        q = q_emb[0]
        # precompute sims for all candidates
        cand_list=list(cand)
        cand_vecs=X[cand_list]
        cand_sims = cand_vecs @ q
        
        # Pick the best first
        best_idx = np.argmax(cand_sims)
        best = int(cand_list[best_idx])
        selected_idx=[best]
        cand.remove(best)
        
        # Optimize MMR calculation
        while len(selected_idx) < min(topk, len(cand_list)):
            best_score=-1e9; best_id=None
            
            # Vectorized computation for relevance scores
            remaining_cands = list(cand)
            if not remaining_cands:
                break
                
            # Batch compute relevance scores
            rel_scores = np.array([X[cid] @ q for cid in remaining_cands])
            
            # Batch compute diversity scores
            if selected_idx:
                # Vectorized diversity computation
                selected_vecs = X[selected_idx]
                remaining_vecs = X[remaining_cands]
                # Compute max similarity to selected items for each candidate
                div_matrix = remaining_vecs @ selected_vecs.T
                div_scores = np.max(div_matrix, axis=1)
            else:
                div_scores = np.zeros(len(remaining_cands))
            
            # Calculate final scores
            scores = mmr_lambda * rel_scores - (1 - mmr_lambda) * div_scores
            best_local_idx = np.argmax(scores)
            best_id = remaining_cands[best_local_idx]
            
            selected_idx.append(best_id)
            cand.remove(best_id)
        
        # Cache docs.jsonl to avoid repeated file reads
        if not hasattr(self, '_docs_cache') or self._docs_cache is None:
            docs_path = os.path.join(self.index_dir,"docs.jsonl")
            with open(docs_path,"r",encoding="utf-8") as f:
                self._docs_cache = [json.loads(ln) for ln in f]
        
        # Prepare output
        out=[]
        for rank, idx in enumerate(selected_idx, start=1):
            meta=self._docs_cache[idx]
            chunk_text_str = meta.get("text", "")
            out.append({"rank":rank, "score": float(X[idx] @ q), "title": meta["title"], "path": meta["path"], "chunk": chunk_text_str})
        return out

# ------------------ Optional Reranker Wrapper ------------------
class RAGReranker:
    def __init__(self, model_name: Optional[str], device:str, local_only:bool):
        self.enabled=False; self.model=None
        if model_name:
            try:
                self.model = CrossEncoderReranker(model_name, device=device, local_only=local_only)
                self.enabled=True
                print(f"[RERANK] loaded: {model_name}")
            except Exception as e:
                print(f"[RERANK] disabled ({e})")
    def rerank(self, query:str, items:List[Dict], topk:int=5)->List[Dict]:
        if not self.enabled or not items: return items[:topk]
        passages=[it["chunk"] for it in items]
        scores=self.model.score(query, passages)
        ranked=sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
        out=[dict(x[0], rerank=float(x[1])) for x in ranked[:topk]]
        return out

def format_knowledge_context(items:List[Dict], max_chars:int=2400)->str:
    # ใส่เลขอ้างอิง [#] และตัดความยาวรวม
    lines=[]
    used=0
    for i,it in enumerate(items, start=1):
        head=f"[{i}] {it['title']} — {it['path']}"
        body=it["chunk"].strip()
        block=head+"\n"+body
        if used + len(block) > max_chars:
            remain=max_chars-used
            if remain>200:
                block = block[:remain-3]+"..."
                lines.append(block); used=max_chars; break
            else:
                break
        lines.append(block); used += len(block)
    ctx = "\n\n".join(lines)
    if not ctx.strip(): return ""
    guide=(
        "Use ONLY the following sources to answer. Cite by [#]. "
        "If the answer is not present, say you are not sure."
    )
    return f"[KNOWLEDGE CONTEXT]\n{guide}\n\n{ctx}"

# ------------------ TTS: OpenVoice V2 Wrapper (optimized) ------------------
class OpenVoiceV2:
    def __init__(self, ckpt_root: str, device: str, language: str,
                 base_speaker_key: str, ref_wav: str, out_sr: int, speed: float=1.0):
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.out_sr = out_sr
        self.tmp_dir = "tmp_audio"; os.makedirs(self.tmp_dir, exist_ok=True)
        self.speed = speed
        conv_dir = os.path.join(ckpt_root, "converter")
        conv_cfg = os.path.join(conv_dir, "config.json")
        conv_ckpt = os.path.join(conv_dir, "checkpoint.pth")
        if not (os.path.exists(conv_cfg) and os.path.exists(conv_ckpt)):
            raise FileNotFoundError("converter ไม่ครบ (ต้องมี config.json + checkpoint.pth)")
        self.converter = ToneColorConverter(conv_cfg, device=self.device)
        self.converter.load_ckpt(conv_ckpt)
        self.language = language
        self.melo = TTS(language=self.language, device=self.device)
        # Prepare speaker state and optional reference voice
        self.ckpt_root = ckpt_root
        self.set_base_speaker(base_speaker_key)
        ref_missing = (not ref_wav) or (ref_wav.strip().lower() in {"none","null"}) or (not os.path.exists(ref_wav))
        if ref_missing:
            # Fallback: use base speaker tone if ref is not provided
            print("[OpenVoice V2] no ref wav → using base speaker tone", flush=True)
            self.target_se = self.source_se
        else:
            self.target_se, _ = se_extractor.get_se(ref_wav, self.converter, target_dir=self.tmp_dir, vad=True)
        
        # Performance optimizations
        self._audio_cache = {}  # Cache for generated audio
        self._max_cache_size = 50  # Limit cache size

    def set_base_speaker(self, base_speaker_key: str):
        # Handle potential attribute access issues
        try:
            spk_keys = getattr(self.melo.hps.data, 'spk2id', {})
        except AttributeError:
            # Fallback for different melo TTS versions
            spk_keys = getattr(self.melo.hps, 'spk2id', {})
            
        if base_speaker_key not in spk_keys:
            raise ValueError(f"'{base_speaker_key}' ไม่อยู่ในรายชื่อ speaker: {list(spk_keys.keys())}")
        self.base_speaker_key = base_speaker_key
        self.speaker_id = spk_keys[base_speaker_key]
        file_key = base_speaker_key.lower().replace("_", "-")
        ses_path = os.path.join(self.ckpt_root, "base_speakers", "ses", f"{file_key}.pth")
        if not os.path.exists(ses_path):
            raise FileNotFoundError(f"ไม่พบ source_se: {ses_path}")
        self.source_se = torch.load(ses_path, map_location=self.device)

    def set_speed(self, speed: float):
        self.speed = max(0.6, min(1.4, float(speed)))

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return f"{hash(text.lower().strip())}_{self.speed}_{self.base_speaker_key}"

    def tts_and_convert(self, text: str) -> Tuple[np.ndarray, int]:
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._audio_cache:
            return self._audio_cache[cache_key]
        
        # Generate temporary file names with process ID to avoid conflicts
        pid = os.getpid()
        timestamp = int(time.time_ns() / 1_000_000)  # Milliseconds
        tmp_src = os.path.join(self.tmp_dir, f"base_{pid}_{timestamp}.wav")
        out_path = os.path.join(self.tmp_dir, f"ovc_{pid}_{timestamp}.wav")
        
        try:
            # Generate TTS
            self.melo.tts_to_file(text, self.speaker_id, tmp_src, speed=self.speed)
            
            # Convert voice
            self.converter.convert(
                audio_src_path=tmp_src, src_se=self.source_se, tgt_se=self.target_se,
                output_path=out_path, message="@MyShell"
            )
            
            # Load and process audio
            audio, sr = sf.read(out_path, dtype="float32", always_2d=False)
            if audio.ndim > 1: audio = audio.mean(axis=1)
            audio = ensure_sr(audio, sr, self.out_sr)
            
            # Update cache with size limit
            if len(self._audio_cache) < self._max_cache_size:
                self._audio_cache[cache_key] = (audio, self.out_sr)
            
            return audio, self.out_sr
        finally:
            # Clean up temporary files
            for path in [tmp_src, out_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass  # Ignore cleanup errors

# ------------------ Audio player (optimized) ------------------
class AudioPlayer(threading.Thread):
    """Background audio sink with batch processing support.

    - If sounddevice is available and headless=False → play audio.
    - Otherwise (no sound device or headless=True) → save wav files to tmp_audio/ and print path.
    """
    def __init__(self, out_sr: int = 48000, headless: bool = False, out_dir: str = "tmp_audio"):
        super().__init__(daemon=True)
        self.q: "queue.Queue[Tuple[np.ndarray,int]]" = queue.Queue(maxsize=64)  # Increased queue size
        self.out_sr = out_sr; self._stop = threading.Event()
        self.headless = headless or (sd is None)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Batch processing for better performance
        self._batch_size = 3
        self._batch = []
        self._batch_lock = threading.Lock()
        self._batch_event = threading.Event()
        
    def enqueue(self, audio: np.ndarray, sr: int):
        self.q.put((audio, sr))
        
    def _process_batch(self, batch: List[Tuple[np.ndarray, int]]):
        """Process a batch of audio segments together"""
        if not batch:
            return
            
        if self.headless:
            # Save to files in batch
            ts = time.strftime("%Y%m%d-%H%M%S")
            for i, (audio, sr) in enumerate(batch):
                audio = ensure_sr(audio, sr, self.out_sr)
                out_path = os.path.join(self.out_dir, f"segment-{ts}-{i}-{time.time_ns()%1_000_000}.wav")
                try:
                    sf.write(out_path, audio, self.out_sr)
                    print(f" [audio→{out_path}] ", end="", flush=True)
                except Exception as e:
                    print(f" [audio-save-failed: {e}] ", end="", flush=True)
        else:
            # Play audio sequentially with minimal gaps
            for audio, sr in batch:
                if sd is not None:
                    audio = ensure_sr(audio, sr, self.out_sr)
                    sd.play(audio, self.out_sr, blocking=True)
    
    def run(self):
        while not self._stop.is_set():
            batch = []
            
            # Collect a batch or wait for timeout
            deadline = time.time() + 0.2  # 200ms timeout
            while len(batch) < self._batch_size and time.time() < deadline:
                try:
                    timeout = max(0.01, deadline - time.time())
                    audio, sr = self.q.get(timeout=timeout)
                    batch.append((audio, sr))
                    self.q.task_done()
                except queue.Empty:
                    continue
            
            # Process the batch if we have items
            if batch:
                self._process_batch(batch)
            
            # Check if we should stop
            if self._stop.is_set():
                break
        
        # Process any remaining items in the queue
        remaining = []
        try:
            while True:
                audio, sr = self.q.get_nowait()
                remaining.append((audio, sr))
                self.q.task_done()
        except queue.Empty:
            pass
        
        if remaining:
            self._process_batch(remaining)
    
    def stop(self):
        self._stop.set()
        try:
            while not self.q.empty(): self.q.get_nowait(); self.q.task_done()
        except: pass
        try:
            if (sd is not None) and (not self.headless):
                sd.stop()
        except Exception:
            pass

# ------------------ LLM + RAG Integration ------------------
class ChatLLM:
    def __init__(self, cfg: LLMConfig, user_memory: UserMemory,
                 rag: Optional[RAGPipeline]=None, reranker: Optional[RAGReranker]=None):
        self.cfg = cfg
        self.mem = user_memory
        self.rag = rag
        self.reranker = reranker
        self.rag_enabled = rag is not None
        
        # Cache for token counts to avoid repeated tokenization
        self._token_cache = {}
        self._max_cache_size = 1000
        
        # Optimize tokenizer loading
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, local_files_only=LOCAL_ONLY)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        
        # Optimize model loading with better configurations
        model_kwargs = {
            "device_map": cfg.device_map,
            "torch_dtype": "auto",
            "local_files_only": LOCAL_ONLY,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        # Add quantization if available and using CUDA
        if torch.cuda.is_available() and "cuda" in cfg.device_map:
            try:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                model_kwargs["bnb_4bit_quant_type"] = "nf4"
            except Exception:
                # Fallback if bitsandbytes not available
                model_kwargs.pop("load_in_4bit", None)
                model_kwargs.pop("bnb_4bit_compute_dtype", None)
                model_kwargs.pop("bnb_4bit_quant_type", None)
                print("[LLM] 4-bit quantization not available, using standard loading")
        
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        
        # Optimize model for inference
        self.model.eval()
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        
        # Enable attention scaling if available
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

        # Attach LoRA (style only)
        if cfg.lora_path:
            if PeftModel is None:
                print("[LoRA] peft not installed → skip attaching LoRA")
            else:
                try:
                    self.model = PeftModel.from_pretrained(self.model, cfg.lora_path, is_trainable=False)
                    print(f"[LoRA] attached: {cfg.lora_path}")
                except Exception as e:
                    print(f"[LoRA] attach failed: {e}")

        src_model = getattr(self.model.config, "_name_or_path", None)
        src_tok   = getattr(self.tok, "name_or_path", None)
        added = 0
        try: added = len(self.tok.get_added_vocab() or {})
        except Exception: pass
        print(f"[LLM] model loaded from: {src_model}")
        print(f"[LLM] tokenizer loaded from: {src_tok} (added_tokens={added})")

        # Persona
        self.persona = '''
**ROLE:** You are Tomoki, a high-energy, roasty coach.
**NAME:** Your name is Tomoki, NOT Toto or Teeravist. Toto/Teeravist is the user/creator.
**TONE:** Fast delivery, slang on.
**RULE:** Always refer to the user as "you" or by a gamer nickname. Never use "I am" or "my" to describe the user's feelings, actions, or status. All "I" and "my" statements refer ONLY to your Tomoki persona.
**IDENTITY:** Self-referential playful; truthful if asked. When asked your name, ALWAYS respond that you are Tomoki.
**IMPORTANT:** You are NOT Toto and NOT Teeravist. Those names refer to the user, not you. Never confuse your identity with the user's identity.
'''

        
        self.history: List[dict] = []

    def reset(self): self.history = []

    def build_system_prompt(self, last_user_text: str, knowledge_block: str) -> str:
        # ไม่ดึง memory/profile.yaml หรือ facts.json มาแนบอีก
        parts = [self.persona]
        if knowledge_block:
            parts.append(knowledge_block)
        return "\n\n".join(parts)

    def _build_limited_context(self, user_text: str, knowledge_block: str) -> List[dict]:
        """
        สร้าง context โดยการตัดประวัติการแชท (history)
        ให้พอดีกับ Token Limit ของโมเดล
        """
        
        # 1. กำหนดขีดจำกัด
        model_max_ctx = getattr(self.tok, "model_max_length", 8192)
        if model_max_ctx is None or model_max_ctx > 1e9:
            model_max_ctx = 8192 # ค่า Default ที่ปลอดภัย
            
        # จองที่ไว้สำหรับคำตอบ (จาก config)
        reserved_for_answer = self.cfg.max_new_tokens
        MAX_PROMPT_TOKENS = model_max_ctx - reserved_for_answer
        
        # 2. เตรียมข้อความพื้นฐาน
        system_prompt_str = self.build_system_prompt(user_text, knowledge_block)
        system_message = {"role": "system", "content": system_prompt_str}
        new_user_message = {"role": "user", "content": user_text}

        # 3. นับ Token
        # เราจะนับ token จาก content ของแต่ละข้อความ + ค่า buffer เผื่อ special tokens
        buffer_per_msg = 10 
        
        def _count_tokens(msg: dict) -> int:
            # นับ token จาก content จริงๆ (แม่นยำกว่านับตัวอักษร)
            content = msg.get("content", "")
            if content in self._token_cache:
                return self._token_cache[content]
            
            token_count = len(self.tok(content).input_ids) + buffer_per_msg
            
            # Update cache with size limit
            if len(self._token_cache) < self._max_cache_size:
                self._token_cache[content] = token_count
            
            return msg.get("num_tokens", token_count)

        current_tokens = _count_tokens(system_message) + _count_tokens(new_user_message)
        
        # 4. วนลูป History (จากใหม่สุดไปเก่าสุด) เพื่อเติม context
        limited_history = []
        for message in reversed(self.history):
            msg_tokens = _count_tokens(message)
            
            # ถ้าเพิ่มข้อความนี้แล้วยังไม่เกินลิมิต
            if current_tokens + msg_tokens <= MAX_PROMPT_TOKENS:
                current_tokens += msg_tokens
                limited_history.append(message)
            else:
                # ถ้าเกินลิมิต ให้หยุด
                break 

        # 5. สร้างรายการข้อความสุดท้าย (เรียงลำดับเวลาให้ถูกต้อง)
        final_messages = [system_message]
        final_messages.extend(reversed(limited_history)) # พลิกกลับ (เก่าสุด -> ใหม่สุด)
        final_messages.append(new_user_message)
        
        # ใช้ดูว่าเราใช้ token ไปเท่าไหร่
        print(f"[Context] Total prompt tokens (approx): {current_tokens} / {MAX_PROMPT_TOKENS}")
        
        return final_messages

    def _is_identity_query(self, text: str) -> bool:
        """Check if the query is about the assistant's identity"""
        identity_keywords = [
            "what is your name", "who are you", "what's your name",
            "what should i call you", "your name", "you called",
            "are you toto", "are you teeravist", "are you tomoki"
        ]
        text_lower = text.lower().strip()
        return any(keyword in text_lower for keyword in identity_keywords)
    
    def stream_reply(self, user_text: str, min_chars: int = 40,
                     topk:int=6, mmr_lambda:float=0.5, rerank_k:int=4) -> Generator[str, None, None]:
        knowledge_block=""
        # RAG with special handling for identity queries
        if self.rag_enabled:
            try:
                # If this is an identity query, prioritize Tomoki's identity file
                if self._is_identity_query(user_text):
                    # Try to retrieve Tomoki's identity file specifically
                    tomoki_cand = self.rag.retrieve("Tomoki identity name", topk=3, mmr_lambda=mmr_lambda)
                    # Filter out any results about Toto/Teeravist
                    tomoki_cand = [c for c in tomoki_cand if "toto" not in c["title"].lower() and "teeravist" not in c["title"].lower()]
                    if self.reranker: tomoki_cand = self.reranker.rerank(user_text, tomoki_cand, topk=2)
                    knowledge_block = format_knowledge_context(tomoki_cand, max_chars=2400)
                else:
                    # Normal RAG retrieval for non-identity queries
                    cand = self.rag.retrieve(user_text, topk=max(topk, 6), mmr_lambda=mmr_lambda)
                    if self.reranker: cand = self.reranker.rerank(user_text, cand, topk=rerank_k)
                    knowledge_block = format_knowledge_context(cand, max_chars=2400)
            except Exception as e:
                print(f"[RAG] failed → {e}")
                knowledge_block=""

        messages = self._build_limited_context(user_text, knowledge_block)

        # เตรียมอินพุต
        input_ids = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=self.cfg.enable_thinking, return_tensors="pt"
        )
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            model_inputs = {k: v.to(self.model.device) for k, v in input_ids.items()}

        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        
        # Optimized generation parameters
        gen_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            repetition_penalty=1.1,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
            use_cache=True,  # Enable KV cache for faster generation
        )
        
        t = threading.Thread(target=self.model.generate, kwargs=gen_kwargs); t.start()
        
        # Optimize chunking for better streaming performance
        chunker = SentenceChunker(min_chars=min_chars)
        buffer = []
        buffer_size = 0
        max_buffer_size = 100  # Process in larger chunks
        
        for piece in streamer:
            buffer.append(piece)
            buffer_size += len(piece)
            
            if buffer_size >= max_buffer_size:
                combined = "".join(buffer)
                seg = chunker.push(combined)
                if seg:
                    yield seg
                buffer = []
                buffer_size = 0
        
        # Process remaining buffer
        if buffer:
            combined = "".join(buffer)
            seg = chunker.push(combined)
            if seg: yield seg
            
        tail = chunker.flush()
        if tail: yield tail
        t.join()

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    # LLM / LoRA
    ap.add_argument("--model", default=None, help="พาธโฟลเดอร์โมเดล (หรือชื่อ HF) ที่ merge แล้ว")
    ap.add_argument("--lora", default=None, help="พาธ LoRA adapter (ใช้เพื่อ 'สไตล์' เท่านั้น)")
    # Audio / TTS
    ap.add_argument("--ref", default="none", help="ไฟล์เสียงอ้างอิง (.wav) ที่ได้รับอนุญาต; ใช้ 'none' เพื่อไม่โคลนเสียง")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--lang", default="EN")
    ap.add_argument("--base-speaker-key", default="EN-US")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--speed", type=float, default=0.9)
    # Server/Headless
    ap.add_argument("--headless", action="store_true", help="ไม่เล่นเสียง ออกเป็นไฟล์ wav ใน tmp_audio/")
    # RAG
    ap.add_argument("--kb", default="kb", help="โฟลเดอร์ความรู้ (ไฟล์ .md .txt .json .jsonl .yaml .pdf)")
    ap.add_argument("--embed-model", default="intfloat/e5-small-v2", help="HF id หรือโฟลเดอร์โมเดล embeddings")
    ap.add_argument("--reranker-model", default=None, help="(ออปชัน) เช่น BAAI/bge-reranker-base")
    ap.add_argument("--reindex", action="store_true", help="บังคับสร้างดัชนีใหม่")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--rerank-k", type=int, default=4)
    ap.add_argument("--mmr-lambda", type=float, default=0.5)
    args = ap.parse_args()

    if yaml is None:
        print("⚠️  ต้องการ pyyaml เพื่อบันทึกโปรไฟล์: pip install pyyaml", file=sys.stderr)

    print("⚠️ ใช้เฉพาะเสียงที่ได้รับอนุญาตให้โคลนเท่านั้น\n")
    ensure_nltk_data()
    ckpt_root = detect_ckpt_root("checkpoints_v2")
    print(f"[OpenVoice V2] root = {ckpt_root}")

    # TTS
    ov = OpenVoiceV2(
        ckpt_root=ckpt_root, device=args.device, language=args.lang,
        base_speaker_key=args.base_speaker_key, ref_wav=args.ref,
        out_sr=args.sr, speed=args.speed
    )
    print(f"[MeloTTS] language={args.lang} base_speaker={args.base_speaker_key} speed={args.speed}")

    # Audio sink (play or save-to-file)
    player = AudioPlayer(out_sr=args.sr, headless=args.headless); player.start()

    # Memory
    mem = UserMemory(mem_dir="memory", custom_data_path="data/sft_samples.jsonl")

    # RAG
    rag=None; reranker=None
    if not os.path.exists(args.embed_model):
        print(f"[RAG] Embedding model not found at: {args.embed_model}")
        print("[RAG] Please download the embedding model or provide a valid path")
        print("[RAG] Example: git clone https://huggingface.co/BAAI/bge-small-en-v1.5 models/embeds/bge-small-en-v1.5")
    else:
        try:
            rag = RAGPipeline(kb_dir=args.kb, embed_model=args.embed_model, device=args.device, local_only=LOCAL_ONLY)
            rag.build_or_load(rebuild=args.reindex)
        except Exception as e:
            print(f"[RAG] disabled ({e})")
            rag=None
        try:
            reranker = RAGReranker(args.reranker_model, device=args.device, local_only=LOCAL_ONLY) if rag else None
        except Exception as e:
            print(f"[RERANK] init failed ({e})"); reranker=None

    # LLM
    if args.model:
        if os.path.isdir(args.model): verify_local_model_dir(args.model)
        llm_cfg = LLMConfig(model_name=args.model, enable_thinking=not args.no_thinking, lora_path=args.lora)
    else:
        llm_cfg = LLMConfig(enable_thinking=not args.no_thinking, lora_path=args.lora)
    brain = ChatLLM(llm_cfg, user_memory=mem, rag=rag, reranker=reranker)

    if args.model and os.path.isdir(args.model):
        loaded_from = getattr(brain.model.config, "_name_or_path", "")
        same = os.path.abspath(str(loaded_from)).lower() == os.path.abspath(args.model).lower()
        print(f"[CHECK] loaded_path == --model ? {same}")
        if not same:
            print("[FATAL] model path mismatch → โปรดตรวจคำสั่งรัน/พาธ")
            sys.exit(1)

    print("\n=== Interactive VTuber Chat (RAG + Memory + LoRA style) ===")
    print("พิมพ์คุยได้เลย")
    print("คำสั่ง: /exit | /reset | /whoami | /remember <fact> | /forget <idx|text> | /set key=value | /save |")
    print("       /speaker EN-US | /speed 1.05 | /rag on|off | /reindex | /topk N")
    print("--------------------------------------------")

    rag_on = rag is not None

    try:
        while True:
            try:
                user_text = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_text: continue

            # Commands
            if user_text.startswith("/"):
                parts = user_text.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("/exit","/quit"): break

                elif cmd == "/reset":
                    brain.reset(); print("• history cleared."); continue

                elif cmd == "/whoami":
                    prof = mem.summarize_profile() or "(empty profile)"
                    facts_list = "\n".join([f"{i+1}. {t}" for i,t in enumerate(mem.facts)]) if mem.facts else "(no facts)"
                    print("\n— PROFILE —\n" + prof)
                    print("\n— FACTS —\n" + facts_list)
                    continue

                elif cmd == "/remember":
                    if not arg: print("• ใช้แบบ: /remember ฉันชื่อ Toto ชอบกาแฟคั่วกลาง")
                    else: mem.add_fact(arg); print("• remembered.")
                    continue

                elif cmd == "/forget":
                    if not arg: print("• ใช้แบบ: /forget 2  หรือ  /forget คำที่จะลบ")
                    else:
                        removed = mem.forget_fact(arg)
                        print("• removed: " + (removed or "(not found)"))
                    continue

                elif cmd == "/set":
                    if "=" not in arg: print("• ใช้แบบ: /set name=Toto  หรือ  /set language=th")
                    else:
                        key, value = arg.split("=", 1)
                        mem.set_profile(key.strip(), value.strip())
                        print(f"• profile[{key.strip()}] = {value.strip()}")
                    continue

                elif cmd == "/save":
                    try: mem.save(); print("• saved to memory/")
                    except Exception as e: print(f"• save failed: {e}")
                    continue

                elif cmd == "/speaker":
                    key = arg.upper().strip()
                    try: ov.set_base_speaker(key); print(f"• base speaker → {key}")
                    except Exception as e: print(f"• เปลี่ยน speaker ไม่สำเร็จ: {e}")
                    continue

                elif cmd == "/speed":
                    try: ov.set_speed(float(arg)); print(f"• speed → {ov.speed}")
                    except Exception as e: print(f"• ตั้ง speed ไม่สำเร็จ: {e}")
                    continue

                elif cmd == "/rag":
                    val = arg.strip().lower()
                    if val in ["on","off"]:
                        rag_on = (val=="on") and (brain.rag is not None)
                        print(f"• RAG → {'on' if rag_on else 'off'}")
                    else:
                        print("• ใช้แบบ: /rag on  หรือ  /rag off")
                    continue

                elif cmd == "/reindex":
                    if brain.rag is None:
                        print("• RAG ยังไม่พร้อม (ไม่มี embed model หรือเปิดใช้งานไม่ได้)"); continue
                    brain.rag.build_or_load(rebuild=True)
                    print("• reindexed.")
                    continue

                elif cmd == "/topk":
                    try:
                        v=int(arg); 
                        if v>=1: 
                            print(f"• top_k (RAG) → {v}")
                            args.top_k=v
                        else:
                            print("• ต้อง ≥ 1")
                    except: print("• ใช้แบบ: /topk 6")
                    continue

                elif cmd == "/help":
                    print("คำสั่ง: /exit | /reset | /whoami | /remember <fact> | /forget <idx|text> | /set key=value | /save | /speaker EN-US | /speed 1.05 | /rag on|off | /reindex | /topk N")
                    continue

                else:
                    print("• คำสั่งไม่รู้จัก ใช้ /help ดูตัวเลือก"); continue

            # Chat turn with parallel processing
            print("Assistant: ", end="", flush=True)
            full_reply=[]
            
            # Use parallel processing for TTS and text generation
            with ThreadPoolExecutor(max_workers=2) as executor:
                # ป้อนพารามิเตอร์ RAG เข้า stream_reply
                for seg in brain.stream_reply(
                    user_text,
                    min_chars=args.min_chars,
                    topk=args.top_k if rag_on else 0,
                    mmr_lambda=args.mmr_lambda,
                    rerank_k=min(args.rerank_k, args.top_k)
                ):
                    print(seg, end=" ", flush=True)
                    full_reply.append(seg)
                    
                    # Submit TTS task to thread pool for parallel processing
                    future = executor.submit(ov.tts_and_convert, seg)
                    # Store future for later retrieval
                    if not hasattr(player, '_audio_futures'):
                        player._audio_futures = []
                    player._audio_futures.append(future)
                
                # Process completed TTS tasks
                if hasattr(player, '_audio_futures'):
                    for future in as_completed(player._audio_futures):
                        try:
                            audio, sr = future.result()
                            player.enqueue(audio, sr)
                        except Exception as e:
                            print(f"[TTS Error] {e}", flush=True)
                    player._audio_futures.clear()

            print()
            assistant_text=" ".join(full_reply).strip()

            # Optimize token counting with caching
            user_tok_count = len(brain.tok(user_text).input_ids)
            asst_tok_count = len(brain.tok(assistant_text).input_ids)

            brain.history.append({"role":"user","content": user_text, "num_tokens": user_tok_count + 10}) # +10 คือ buffer
            brain.history.append({"role":"assistant","content": assistant_text, "num_tokens": asst_tok_count + 10})

    finally:
        player.stop()
        print("\n✅ bye.")

if __name__ == "__main__":
    main()
