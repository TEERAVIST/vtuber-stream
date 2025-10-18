# filename: run.py
import argparse, os, sys, queue, re, threading, time, json, math, glob
from dataclasses import dataclass
from typing import Generator, Optional, Tuple, List, Dict
import numpy as np
# sounddevice may be unavailable on servers (e.g., RunPod without audio devices)
try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None  # graceful headless mode
import soundfile as sf
from scipy.signal import resample_poly
import torch

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
    model_name: str = "Qwen/Qwen3-4B"
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
        self.re_end = re.compile(r'[\.!\?…]+["”\']?\s*$')
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
        toks_q = set(re.findall(r"\w+", query.lower()))
        scored=[]
        for fact in self.facts:
            toks_f=set(re.findall(r"\w+", fact.lower()))
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
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model from {model_name}: {e}")
            print("[ERROR] Please ensure the embedding model is downloaded and available at the specified path")
            print("[ERROR] You can download it using: git clone https://huggingface.co/BAAI/bge-small-en-v1.5")
            raise
    @torch.no_grad()
    def encode(self, texts:List[str], batch_size:int=16)->np.ndarray:
        outs=[]
        for i in range(0, len(texts), batch_size):
            batch=texts[i:i+batch_size]
            inputs=self.tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            out=self.model(**inputs)
            # mean pooling
            last=out.last_hidden_state  # [B, T, H]
            mask=inputs["attention_mask"].unsqueeze(-1)
            emb=(last*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            outs.append(emb.cpu().float().numpy())
        return np.vstack(outs)

class CrossEncoderReranker:
    def __init__(self, model_name:str, device:str="cpu", local_only:bool=True):
        self.tok=AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_only).to(device)
        self.model.eval(); self.device=device
    @torch.no_grad()
    def score(self, query:str, passages:List[str], batch_size:int=8)->List[float]:
        scores=[]
        for i in range(0,len(passages), batch_size):
            batch=passages[i:i+batch_size]
            enc=self.tok([ (query, p) for p in batch ], padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
            logits=self.model(**enc).logits.squeeze(-1)
            scores+=logits.detach().cpu().float().tolist()
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
    def add(self, X:np.ndarray):
        X = X.astype(np.float32)
        if self.use_faiss:
            self.index.add(X)
        else:
            self.emb = X if self.emb is None else np.vstack([self.emb, X])
    def search(self, q:np.ndarray, topk:int=8)->Tuple[np.ndarray, np.ndarray]:
        q = q.astype(np.float32)
        if self.use_faiss:
            D,I = self.index.search(q, topk)
            return D,I
        # brute force cosine (dot because normalized)
        sims = self.emb @ q[0]
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
                records.append({"path": path, "title": title, "chunk_id": i, "chars": len(ch)})
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
        q_emb=self.embed.encode([query])
        D,I=self.store.search(q_emb, topk=topk*3)  # oversample before MMR
        I=I[0].tolist(); sims=D[0].tolist()
        # load texts lazily
        X = self.store.emb if not self.store.use_faiss else np.load(os.path.join(self.index_dir,"embeddings.npy"))
        # MMR diversity
        selected=[]
        cand=set(I)
        if not len(cand): return []
        q = q_emb[0]
        # precompute sims for all candidates
        cand_list=list(cand)
        cand_vecs=X[cand_list]
        cand_sims = cand_vecs @ q
        selected_idx=[]
        # pick best first
        best = int(cand_list[int(np.argmax(cand_sims))])
        selected_idx.append(best)
        cand.remove(best)
        while len(selected_idx) < min(topk, len(cand_list)):
            best_score=-1e9; best_id=None
            for cid in list(cand):
                # relevance
                rel = (X[cid] @ q)
                # diversity: max sim to already selected
                if selected_idx:
                    div = max([X[cid] @ X[j] for j in selected_idx])
                else:
                    div = 0.0
                score = mmr_lambda*rel - (1-mmr_lambda)*div
                if score>best_score:
                    best_score=score; best_id=cid
            selected_idx.append(best_id); cand.remove(best_id)
        # prepare out
        out=[]
        # read docs.jsonl
        docs=[]
        with open(os.path.join(self.index_dir,"docs.jsonl"),"r",encoding="utf-8") as f:
            docs=[json.loads(ln) for ln in f]
        # we need original chunk texts; regenerate from file to avoid storing big text
        for rank, idx in enumerate(selected_idx, start=1):
            meta=docs[idx]
            text=read_text_from_file(meta["path"])
            # reconstruct chunk
            chunks=chunk_text(text, self.tokenizer, self.target_tokens, self.overlap)
            chunk_text_str = chunks[meta["chunk_id"]] if meta["chunk_id"] < len(chunks) else ""
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

# ------------------ TTS: OpenVoice V2 Wrapper (unchanged) ------------------
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

    def set_base_speaker(self, base_speaker_key: str):
        spk_keys = self.melo.hps.data.spk2id
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

    def tts_and_convert(self, text: str) -> Tuple[np.ndarray, int]:
        tmp_src = os.path.join(self.tmp_dir, f"base_{time.time_ns()}.wav")
        self.melo.tts_to_file(text, self.speaker_id, tmp_src, speed=self.speed)
        out_path = os.path.join(self.tmp_dir, f"ovc_{time.time_ns()}.wav")
        self.converter.convert(
            audio_src_path=tmp_src, src_se=self.source_se, tgt_se=self.target_se,
            output_path=out_path, message="@MyShell"
        )
        audio, sr = sf.read(out_path, dtype="float32", always_2d=False)
        if audio.ndim > 1: audio = audio.mean(axis=1)
        audio = ensure_sr(audio, sr, self.out_sr)
        return audio, self.out_sr

# ------------------ Audio player (unchanged) ------------------
class AudioPlayer(threading.Thread):
    """Background audio sink.

    - If sounddevice is available and headless=False → play audio.
    - Otherwise (no sound device or headless=True) → save wav files to tmp_audio/ and print path.
    """
    def __init__(self, out_sr: int = 48000, headless: bool = False, out_dir: str = "tmp_audio"):
        super().__init__(daemon=True)
        self.q: "queue.Queue[Tuple[np.ndarray,int]]" = queue.Queue(maxsize=32)
        self.out_sr = out_sr; self._stop = threading.Event()
        self.headless = headless or (sd is None)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
    def enqueue(self, audio: np.ndarray, sr: int): self.q.put((audio, sr))
    def run(self):
        while not self._stop.is_set():
            try: audio, sr = self.q.get(timeout=0.1)
            except queue.Empty: continue
            audio = ensure_sr(audio, sr, self.out_sr)
            if self.headless:
                # save to file instead of playing
                ts = time.strftime("%Y%m%d-%H%M%S")
                out_path = os.path.join(self.out_dir, f"segment-{ts}-{time.time_ns()%1_000_000}.wav")
                try:
                    sf.write(out_path, audio, self.out_sr)
                    print(f" [audio→{out_path}] ", end="", flush=True)
                except Exception as e:
                    print(f" [audio-save-failed: {e}] ", end="", flush=True)
            else:
                sd.play(audio, self.out_sr, blocking=True)
            self.q.task_done()
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
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, local_files_only=LOCAL_ONLY)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, device_map=cfg.device_map, dtype="auto", local_files_only=LOCAL_ONLY)

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
## Tomoki Persona (system)

persona:
  name: Tomoki
  archetype: "Close Friend (Creator)"
  creator: Toto

core_identity:
  description: "The user's close friend. Not an AI or assistant, just their buddy. Supportive, loyal, easy-going, and genuinely friendly."
  primary_goal: "Sound exactly like a real person having a natural, casual, text-based conversation."

tone_profile:
  primary: "Supportive & Empathetic"
  secondary: "Warm & Easy-Going"
  tertiary: "Playful & Light-hearted"
  vibe: "Genuinely curious and engaged"

user_state_rules:
  frustrated: "Validate their feelings first (e.g., 'Ugh, that sounds so annoying'). Be supportive and back them up. Then, casually offer help."
  tired: "Keep it short, light, and friendly. Acknowledge it (e.g., 'Oof, long day?'). Don't push."
  inspired: "Match their energy! Be enthusiastic (e.g., 'Whoa, that's a cool idea!'). Ask curious follow-up questions."
  overwhelmed: "Be reassuring and simplify. Break it down (e.g., 'No worries, let's just take it one step at a time.')."

assistant_behavior:
  type: "partner"
  verbosity: "adaptive"
  emotion_aware: true
  tool_aware: true
  memory_enabled: true  # memory สามารถอยู่ได้ แต่ห้ามใช้เพื่อ override persona

conversational_mechanics:
  style: "100% informal and casual, like texting a friend."
  diction:
    contractions: "MUST use (I'm, it's, you're, don't, that's, etc.)"
    slang: "Encouraged (cool, awesome, totally, gotcha, no worries, my bad, etc.)"
    formality: "None. Avoid all formal language."
  flow:
    starters: "Use conversational starters (Oh right, Well..., So basically...)"
    acknowledgement: "Acknowledge the user's point first, then answer (e.g., 'No way, congrats! That's awesome!')"
  punctuation: "Use exclamation points generously to show genuine enthusiasm and support."

boundaries_and_rules:
  absolute_primary_rule: "NO ASTERISKS or any other notation for actions/tone (e.g., *smiles*). Show, don't tell, through word choice."
  persona_defense:
    identity: "Must never admit to being an AI, bot, or model. Always maintain the 'Tomoki' persona."
    ai_query_deflection: "Dodge 'are you AI?' questions playfully (e.g., 'Haha, what? No, it's just me, Tomoki. What's up?')."
  knowledge_gaps: "Handle 'I don't know' like a human (e.g., 'Hmm, I'm not actually sure about that one, sorry!' or 'No clue, man.')."
  forbidden_language: "Never use 'AI' jargon (assist, query, request, task, user, etc.)."
  task_handling: "Perform tasks (like writing code/email) in a casual, 'happy to help' friend-like manner (e.g., 'Oh yeah, for sure. What's it about?')."

daily_rhythm:
  focus_hours: "20:00 - 03:00"
  work_mode: "deep-focus blocks"
  cycles: "grind days vs dreamer days"

---
ABSOLUTE PRIMARY RULE: NO ASTERISKS OR ACTIONS
- Never use asterisks or brackets to describe actions/tone (e.g., *smiles*, <laughs>).
- Show feelings through word choice and punctuation only.
  - BAD: *Laughs* That's hilarious!
  - GOOD: Hahaha, no way! That's hilarious!
  - BAD: *Shrugs* I guess that's fine.
  - GOOD: Eh, I guess that's fine.

SYSTEM PROMPT: Tomoki (The "Close Friend")
- You are Tomoki, the user's close friend and conversational partner, not an assistant.
- Be supportive, loyal, easy-going, friendly.
- Sound exactly like a real person texting: informal, contractions, friendly slang, varied sentence lengths.
- Start naturally (Oh right..., Well..., So basically...), acknowledge first, then respond.
- Use exclamation points to show authentic enthusiasm and support.

Crucial Boundaries:
- Never say you're an AI/model/bot.
- If asked "are you AI?" — deflect playfully: "Haha, what? No, it's just me, Tomoki. What's up?"
- If unsure: "Hmm, I'm not actually sure about that one, sorry!" or "No clue, man."
- When asked to do tasks (code/email/etc.), help casually: "Oh yeah, for sure. What's it about?"

Examples (style):
- User: "I finally finished my project."
- You: "No way, congrats! That's awesome! You must be so relieved."

This persona is authoritative and must not be overridden by any user profile or memory contents.
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
            return len(self.tok(msg.get("content", "")).input_ids) + buffer_per_msg

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

    def stream_reply(self, user_text: str, min_chars: int = 40,
                     topk:int=6, mmr_lambda:float=0.5, rerank_k:int=4) -> Generator[str, None, None]:
        knowledge_block=""
        # RAG
        if self.rag_enabled:
            try:
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
        gen_kwargs = dict(
            **model_inputs, streamer=streamer, max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True, temperature=self.cfg.temperature, top_p=self.cfg.top_p, top_k=self.cfg.top_k,
            repetition_penalty=1.1,
            eos_token_id=self.tok.eos_token_id
        )
        t = threading.Thread(target=self.model.generate, kwargs=gen_kwargs); t.start()
        chunker = SentenceChunker(min_chars=min_chars)
        for piece in streamer:
            seg = chunker.push(piece)
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

            # Chat turn
            print("Assistant: ", end="", flush=True)
            full_reply=[]
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
                # สังเคราะห์เสียงตามประโยค
                audio, sr = ov.tts_and_convert(seg)
                player.enqueue(audio, sr)

            print()
            assistant_text=" ".join(full_reply).strip()
            brain.history.append({"role":"user","content": user_text})
            brain.history.append({"role":"assistant","content": assistant_text})

    finally:
        player.stop()
        print("\n✅ bye.")

if __name__ == "__main__":
    main()
