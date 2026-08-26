#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_match_light.py —— 轻量级语义匹配器（纯 numpy，无 torch 依赖）

原理：
  1. 用 TF-IDF 向量化维度定义+锚点，建 FAISS 索引
  2. 对 JD 短语做同样的向量化
  3. cosine similarity 匹配最佳维度
  4. 关键词兜底（与原 semantic_match.py 一致）

优势：
  - 纯 numpy + faiss-cpu，无需 torch/sentence-transformers
  - 启动快（<1秒），内存占用小（~50MB）
  - 精度虽不如神经网络 embedding，但远优于纯关键词

用法：
  from semantic_match_light import LightSemanticMatcher
  matcher = LightSemanticMatcher("taxonomy.yaml")
  result = matcher.classify("负责薪酬核算与薪资发放")
"""

import os
import re
import yaml
import numpy as np
from collections import Counter
from dataclasses import dataclass


@dataclass
class Classification:
    phrase: str
    dimension_id: str | None
    dimension_name: str | None
    confidence: float
    method: str  # tfidf / lexicon / tfidf+lexicon / unmatched
    jd_depth: int = 0
    review: bool = False
    note: str = ""


class Taxonomy:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.semantic = cfg.get("semantic", {})
        self.dims = {}
        for d in cfg["dimensions"]:
            self.dims[d["id"]] = type("Dim", (), {
                "id": d["id"],
                "name": d["name"],
                "weight": d.get("weight", 10),
                "definition": d.get("definition", ""),
                "anchors": d.get("anchors", []),
                "lexicon": d.get("lexicon", []),
                "depth_verbs": d.get("depth_verbs", {}),
                "negatives": d.get("negatives", []),
            })()
        self._lex_index = self._build_lexicon_index()

    def _build_lexicon_index(self):
        idx = {}
        for did, d in self.dims.items():
            for kw in d.lexicon:
                idx.setdefault(kw, []).append(did)
        return idx

    def lexical_match(self, phrase):
        hits = {}
        for kw, dids in self._lex_index.items():
            if kw in phrase:
                for did in dids:
                    hits[did] = hits.get(did, 0) + 1
        if not hits:
            return None, 0
        best = max(hits, key=hits.get)
        return best, hits[best]

    def judge_depth(self, phrase, dim_id):
        d = self.dims.get(dim_id)
        if not d:
            return 0
        depth = 0
        for lvl in ["L4", "L3", "L2", "L1"]:
            n = int(lvl[1])
            if any(v in phrase for v in d.depth_verbs.get(lvl, [])):
                depth = max(depth, n)
        return depth


class LightSemanticMatcher:
    """轻量级语义匹配器：TF-IDF + FAISS，无需 torch。"""

    def __init__(self, taxonomy_path):
        self.tax = Taxonomy(taxonomy_path)
        self.s = self.tax.semantic
        self._vocab = None
        self._idf = None
        self._dim_vectors = None  # (n_dims, vocab_size)
        self._dim_ids = []
        self._index = None
        self._built = False

    # ---- 分词（简单按字符+常见词） ----
    @staticmethod
    def _tokenize(text):
        """简单中文分词：按字符 bigram + 已知关键词。"""
        text = text.lower()
        tokens = []
        # 字符 bigram
        for i in range(len(text) - 1):
            tokens.append(text[i:i+2])
        # 单字符
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                tokens.append(ch)
        return tokens

    # ---- TF-IDF 向量化 ----
    def _build_vocab(self, corpus):
        """从语料构建词汇表和 IDF。"""
        doc_freq = Counter()
        n_docs = len(corpus)
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                doc_freq[t] += 1
        # 过滤低频和高频
        vocab = {}
        for t, df in doc_freq.items():
            if 1 <= df <= n_docs * 0.8:
                idf = np.log((n_docs + 1) / (df + 1)) + 1
                vocab[t] = idf
        self._vocab = vocab
        self._idf = {t: vocab[t] for t in vocab}

    def _vectorize(self, text):
        """将文本转为 TF-IDF 向量。"""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for i, (word, idf) in enumerate(self._vocab.items()):
            if word in tf:
                vec[i] = tf[word] * idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ---- 建索引 ----
    def ensure_built(self):
        if self._built:
            return True
        try:
            import faiss
        except ImportError:
            print("[warn] faiss 未安装，降级为纯词典规则")
            return False

        # 构建语料：每个维度的 definition + anchors
        corpus = []
        dim_map = []  # (dim_id, text_type, text)
        for did, d in self.tax.dims.items():
            rep = d.definition + "。" + "；".join(d.anchors)
            corpus.append(rep)
            dim_map.append((did, "definition", rep))
            for a in d.anchors:
                corpus.append(a)
                dim_map.append((did, "anchor", a))

        # 构建词汇表和 IDF
        self._build_vocab(corpus)

        # 向量化所有维度表征
        vecs = np.array([self._vectorize(doc) for doc in corpus], dtype=np.float32)

        # 建 FAISS 索引
        dim_size = vecs.shape[1]
        index = faiss.IndexFlatIP(dim_size)
        index.add(vecs)

        self._index = index
        self._dim_map = dim_map
        self._dim_ids = list(self.tax.dims.keys())
        self._built = True
        return True

    # ---- 检索 ----
    def _retrieve(self, phrase):
        """返回 (dim_id, sim) 的 top 结果。"""
        if not self._built:
            return None, 0.0
        q = self._vectorize(phrase).reshape(1, -1)
        k = min(self.s.get("top_k", 5), len(self._dim_map))
        sims, ids = self._index.search(q, k)
        # 取同维度最高 sim
        best = {}
        for sim, i in zip(sims[0], ids[0]):
            if i < 0:
                continue
            did, _, _ = self._dim_map[i]
            if did not in best or sim > best[did]:
                best[did] = float(sim)
        if not best:
            return None, 0.0
        did = max(best, key=best.get)
        return did, best[did]

    # ---- 核心：融合分类 ----
    def classify(self, phrase, use_embed=True):
        """
        四规则闸门：
        1. tfidf ≥ theta_high → 直接采纳
        2. theta_low ≤ tfidf < high + 词典命中 → 采纳+加成
        3. theta_low ≤ tfidf < high + 无词典 → 采纳但 review
        4. tfidf < theta_low → 词典兜底；都不中 → unmatched
        """
        theta_high = self.s.get("theta_high", 0.62)
        theta_low = self.s.get("theta_low", 0.45)
        boost = self.s.get("lexicon_boost", 0.12)

        if not phrase or len(phrase) < 3:
            return Classification(phrase, None, None, 0.0, "unmatched", 0)

        lex_dim, lex_hit = self.tax.lexical_match(phrase)
        tfidf_dim, tfidf_sim = (None, 0.0)
        if use_embed and self.ensure_built():
            tfidf_dim, tfidf_sim = self._retrieve(phrase)

        # 规则 1：tfidf 高置信 → 直接采纳
        if tfidf_dim and tfidf_sim >= theta_high:
            d = self.tax.dims[tfidf_dim]
            return Classification(phrase, tfidf_dim, d.name,
                                  round(min(1.0, tfidf_sim), 3), "tfidf",
                                  self.tax.judge_depth(phrase, tfidf_dim))

        # 规则 2：tfidf 中置信 + 词典一致 → 采纳+加成
        if tfidf_dim and theta_low <= tfidf_sim < theta_high:
            d = self.tax.dims[tfidf_dim]
            if lex_hit and lex_dim == tfidf_dim:
                conf = min(1.0, tfidf_sim + boost)
                return Classification(phrase, tfidf_dim, d.name, round(conf, 3),
                                      "tfidf+lexicon", self.tax.judge_depth(phrase, tfidf_dim))
            if lex_hit and lex_dim != tfidf_dim:
                ld = self.tax.dims[lex_dim]
                return Classification(phrase, lex_dim, ld.name, 0.60,
                                      "lexicon-override-review",
                                      self.tax.judge_depth(phrase, lex_dim),
                                      review=True,
                                      note=f"tfidf指向{d.name}({tfidf_sim:.2f})，词典指向{ld.name}")
            # 仅 tfidf 命中且中置信 → 采纳但标复核
            return Classification(phrase, tfidf_dim, d.name, round(tfidf_sim, 3),
                                  "tfidf-lowconf-review",
                                  self.tax.judge_depth(phrase, tfidf_dim),
                                  review=True, note="tfidf中置信，无词典佐证")

        # 规则 3：tfidf 低置信/未启用 → 词典兜底
        if lex_hit:
            ld = self.tax.dims[lex_dim]
            return Classification(phrase, lex_dim, ld.name, 0.70, "lexicon",
                                  self.tax.judge_depth(phrase, lex_dim))

        # 规则 4：都不中
        return Classification(phrase, None, None, 0.0, "unmatched", 0)


# ---- CLI 测试 ----
if __name__ == "__main__":
    import sys
    tx = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/taxonomy.yaml")
    m = LightSemanticMatcher(tx)
    sample = [
        "负责薪酬核算与薪资发放",
        "设计并推行绩效考核方案",
        "办理社保公积金开户与增减员",
        "搭建人力数据看板与报表",
        "负责全渠道招聘与面试甄选",
        "处理员工关系与劳动争议",
        "制定人力资源战略与规划",
        "安排会议日程与差旅",
        "英文邮件与会议沟通",
        "协助高管处理日常行政事务",
        "协助CEO制定公司战略拆解框架",
        "运用飞书多维表格搭建数据看板",
        "负责OKR/KPI绩效管理体系搭建",
        "主导组织架构与职级体系设计",
        "推动AI在HR流程中的落地应用",
    ]
    use_emb = "--no-embed" not in sys.argv
    print(f"{'JD 短语':<30} | {'维度':<12} | {'置信':>5} | {'深度':>3} | {'方法':<22} | 复核")
    print("-" * 110)
    for p in sample:
        c = m.classify(p, use_embed=use_emb)
        name = c.dimension_name or "-"
        flag = "⚠需复核" if c.review else ""
        print(f"{p:<28} | {name:<12} | {c.confidence:>5} | {c.jd_depth:>3} | {c.method:<22} | {flag}")
