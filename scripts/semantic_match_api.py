#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_match_api.py —— 基于 SiliconFlow API 的语义匹配器

用 SiliconFlow 的 bge-large-zh-v1.5 模型做 embedding，
无需本地 torch，精度高。

用法：
  from semantic_match_api import APISemanticMatcher
  matcher = APISemanticMatcher("taxonomy.yaml", api_key="your-api-key")
  result = matcher.classify("负责薪酬核算与薪资发放")
"""

import os
import yaml
import numpy as np
import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class Classification:
    phrase: str
    dimension_id: Optional[str]
    dimension_name: Optional[str]
    confidence: float
    method: str  # api / lexicon / api+lexicon / unmatched
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


class APISemanticMatcher:
    """基于 SiliconFlow API 的语义匹配器（支持批量 embedding）。"""

    API_URL = "https://api.siliconflow.cn/v1/embeddings"
    MODEL = "BAAI/bge-large-zh-v1.5"

    def __init__(self, taxonomy_path, api_key=None):
        self.tax = Taxonomy(taxonomy_path)
        self.s = self.tax.semantic
        self.api_key = api_key or os.environ.get("SILICONFLOW_API_KEY", "")
        self._dim_embeddings = None
        self._dim_map = []
        self._dim_ids = []
        self._built = False
        self._cache = {}  # 缓存已 embedding 的句子

    def _get_embeddings_batch(self, texts, batch_size=32):
        """批量获取 embedding，自动分批。"""
        if not self.api_key:
            raise ValueError("需要提供 SiliconFlow API Key")
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # 过滤已缓存的
            uncached = [(j, t) for j, t in enumerate(batch) if t not in self._cache]
            if uncached:
                resp = requests.post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.MODEL, "input": [t for _, t in uncached]},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                if "data" not in data:
                    raise RuntimeError(f"API 错误: {data}")
                for item in data["data"]:
                    orig_text = uncached[item["index"]][1]
                    self._cache[orig_text] = item["embedding"]
            # 组装这批结果
            for t in batch:
                all_embeddings.append(self._cache[t])
        return np.array(all_embeddings, dtype=np.float32)

    def _get_embedding(self, texts):
        """获取 embedding（支持单条或多条）。"""
        return self._get_embeddings_batch(texts)

    def _cosine_similarity(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def ensure_built(self):
        if self._built:
            return True
        if not self.api_key:
            print("[warn] 无 API Key，降级为词典规则")
            return False
        print("[api-embed] 正在构建维度索引...")
        texts = []
        dim_map = []
        for did, d in self.tax.dims.items():
            rep = d.definition + "。" + "；".join(d.anchors)
            texts.append(rep)
            dim_map.append((did, "definition", rep))
            for a in d.anchors:
                texts.append(a)
                dim_map.append((did, "anchor", a))
        embeddings = self._get_embeddings_batch(texts)
        self._dim_embeddings = embeddings
        self._dim_map = dim_map
        self._dim_ids = list(self.tax.dims.keys())
        self._built = True
        print(f"[api-embed] 索引构建完成，{len(texts)} 条向量")
        return True

    def _retrieve(self, phrase):
        if not self._built:
            return None, 0.0
        q = self._get_embedding([phrase])[0]
        best = {}
        for i, (did, _, _) in enumerate(self._dim_map):
            sim = self._cosine_similarity(q, self._dim_embeddings[i])
            if did not in best or sim > best[did]:
                best[did] = sim
        if not best:
            return None, 0.0
        did = max(best, key=best.get)
        return did, best[did]

    def pre_embed_sentences(self, sentences):
        """预批量 embedding 一批句子，填入缓存。"""
        unique = list(set(s for s in sentences if s and len(s) >= 4 and s not in self._cache))
        if unique:
            print(f"[api-embed] 预嵌入 {len(unique)} 条新句子...")
            self._get_embeddings_batch(unique)
        return len(unique)

    def classify(self, phrase, use_embed=True):
        theta_high = self.s.get("theta_high", 0.62)
        theta_low = self.s.get("theta_low", 0.45)
        boost = self.s.get("lexicon_boost", 0.12)
        if not phrase or len(phrase) < 3:
            return Classification(phrase, None, None, 0.0, "unmatched", 0)
        lex_dim, lex_hit = self.tax.lexical_match(phrase)
        api_dim, api_sim = (None, 0.0)
        if use_embed and self.ensure_built():
            api_dim, api_sim = self._retrieve(phrase)
        if api_dim and api_sim >= theta_high:
            d = self.tax.dims[api_dim]
            return Classification(phrase, api_dim, d.name,
                                  round(min(1.0, api_sim), 3), "api",
                                  self.tax.judge_depth(phrase, api_dim))
        if api_dim and theta_low <= api_sim < theta_high:
            d = self.tax.dims[api_dim]
            if lex_hit and lex_dim == api_dim:
                conf = min(1.0, api_sim + boost)
                return Classification(phrase, api_dim, d.name, round(conf, 3),
                                      "api+lexicon", self.tax.judge_depth(phrase, api_dim))
            if lex_hit and lex_dim != api_dim:
                ld = self.tax.dims[lex_dim]
                return Classification(phrase, lex_dim, ld.name, 0.60,
                                      "lexicon-override-review",
                                      self.tax.judge_depth(phrase, lex_dim),
                                      review=True,
                                      note=f"API指向{d.name}({api_sim:.2f})，词典指向{ld.name}")
            return Classification(phrase, api_dim, d.name, round(api_sim, 3),
                                  "api-lowconf-review",
                                  self.tax.judge_depth(phrase, api_dim),
                                  review=True, note="API中置信，无词典佐证")
        if lex_hit:
            ld = self.tax.dims[lex_dim]
            return Classification(phrase, lex_dim, ld.name, 0.70, "lexicon",
                                  self.tax.judge_depth(phrase, lex_dim))
        return Classification(phrase, None, None, 0.0, "unmatched", 0)


# ---- CLI 测试 ----
if __name__ == "__main__":
    import sys
    tx = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/taxonomy.yaml")
    key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        print("用法: python semantic_match_api.py <API_KEY>")
        sys.exit(1)

    m = APISemanticMatcher(tx, api_key=key)
    sample = [
        "负责薪酬核算与薪资发放",
        "设计并推行绩效考核方案",
        "办理社保公积金开户与增减员",
        "搭建人力数据看板与报表",
        "负责全渠道招聘与面试甄选",
        "处理员工关系与劳动争议",
        "制定人力资源战略与规划",
        "安排会议日程与差旅",
        "协助CEO制定公司战略拆解框架",
        "运用飞书多维表格搭建数据看板",
        "负责OKR/KPI绩效管理体系搭建",
        "主导组织架构与职级体系设计",
        "推动AI在HR流程中的落地应用",
        "协助管理层处理日常事务",
    ]
    print(f"{'JD 短语':<30} | {'维度':<12} | {'置信':>5} | {'深度':>3} | {'方法':<18} | 复核")
    print("-" * 100)
    for p in sample:
        c = m.classify(p, use_embed=True)
        name = c.dimension_name or "-"
        flag = "⚠需复核" if c.review else ""
        print(f"{p:<28} | {name:<12} | {c.confidence:>5} | {c.jd_depth:>3} | {c.method:<18} | {flag}")
