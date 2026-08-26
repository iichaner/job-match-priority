#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
job-match-priority —— 确定性匹配引擎（零 LLM 调用）

设计目标：把原 skill「全链路 Agent 推理」中对每条 JD 的语义判断，
编译成纯 Python 可复用的规则引擎。Agent 只在「首次使用」时生成
candidate_profile.json + match_config.yaml（见 SKILL.optimized.md），
之后任意数量的 JD、任意候选人，都只跑本脚本，token 消耗降至接近 0。

依赖：仅标准库 + PyYAML。
可选：sentence-transformers + faiss-cpu（本地 embedding 分类器，离线、无 API）。
"""

import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("缺少依赖 PyYAML：pip install pyyaml\n")
    raise

LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}
RANK_LEVEL = {v: k for k, v in LEVEL_RANK.items()}

# ── 解析工具 ──────────────────────────────────────────────

def parse_salary(text):
    """返回 (min_k, max_k) 浮点；无法解析返回 (None, None)。"""
    if not text:
        return (None, None)
    s = str(text).replace(",", "").replace("，", "")
    if "面议" in s or "薪资面议" in s:
        return (None, None)
    # 形如 15-17K / 10-15k·24薪 / 8K-12K / 20K
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*[kK]", s)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[kK]", s)
    if m:
        v = float(m.group(1))
        return (v, v)
    return (None, None)


def parse_experience(text):
    """返回 (min_years, max_years)；缺失为 None。"""
    if not text:
        return (None, None)
    s = str(text)
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*年", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(\d+)\s*年以上", s)
    if m:
        return (int(m.group(1)), None)
    m = re.search(r"(\d+)\s*年以下", s)
    if m:
        return (None, int(m.group(1)))
    return (None, None)


def contains_any(text, keywords):
    if not text:
        return False
    t = str(text).lower()
    return any(str(k).lower() in t for k in keywords)


def count_hits(text, keywords):
    if not text:
        return 0
    t = str(text).lower()
    return sum(1 for k in keywords if str(k).lower() in t)


# ── 分类器：关键词兜底 + 可选本地 embedding ───────────────

class Classifier:
    """将 JD 能力句归类到 KSAO 子维度。
    
    优先级: 关键词 → API embedding (SiliconFlow) → light-semantic (tfidf) → 原 embedding → 无匹配。
    """

    def __init__(self, config, embed=None, light_embed=None, api_embed=None):
        self.taxonomy = config.get("taxonomy", [])
        self.embed = embed  # 可选：原 LocalEmbeddingClassifier 实例
        self.light_embed = light_embed  # 可选：LightSemanticMatcher 实例
        self.api_embed = api_embed  # 可选：APISemanticMatcher 实例

    def classify(self, sentence):
        if not sentence or len(sentence) < 4:
            return None
        # 1) 关键词优先
        for entry in self.taxonomy:
            if contains_any(sentence, entry.get("keywords", [])):
                return entry["dimension"]
        # 2) API embedding (SiliconFlow bge-large-zh-v1.5)
        if self.api_embed and self.api_embed._built:
            from semantic_match_api import Classification
            c = self.api_embed.classify(sentence, use_embed=True)
            if c.dimension_id and c.confidence >= 0.45:
                dim_name = self._map_dim_id(c.dimension_id)
                if dim_name:
                    return dim_name
        # 3) light-semantic (tfidf+faiss, 无需 torch)
        if self.light_embed and self.light_embed._built:
            from semantic_match_light import Classification
            c = self.light_embed.classify(sentence, use_embed=True)
            if c.dimension_id and c.confidence >= 0.45:
                dim_name = self._map_dim_id(c.dimension_id)
                if dim_name:
                    return dim_name
        # 4) 原 embedding 兜底
        if self.embed and self.embed.is_available:
            dim, sim = self.embed.classify(sentence)
            if dim and sim >= 0.35:
                return dim
        return None

    def _map_dim_id(self, dim_id):
        """将 taxonomy.yaml 的英文 dimension_id 映射到 match_config.yaml 的中文 dimension name。"""
        ID_TO_NAME = {
            "compensation": "薪酬福利",
            "performance": "目标与绩效管理",
            "social_insurance": "薪酬福利",
            "people_analytics": "数据能力",
            "recruitment": "招聘与配置",
            "er": "员工关系",
            "strategy": "战略能力",
            "admin": "助理职能",
            "english": "英语能力",
        }
        # 也支持直接匹配中文名
        for entry in self.taxonomy:
            if entry["dimension"] == dim_id:
                return dim_id
        return ID_TO_NAME.get(dim_id)


# ── 深度判定 ──────────────────────────────────────────────

VERB_LEVELS = [
    (4, ["战略", "决策", "规划", "组织变革", "经营分析", "方法论", "体系输出"]),
    (3, ["搭建", "统筹", "体系化", "管理", "主导", "设计", "体系", "优化体系", "制度建设"]),
    (2, ["负责", "独立完成", "推动", "跟进", "执行", "实施"]),
    (1, ["协助", "配合", "参与", "了解", "支持", "跟进"]),
]

def judge_jd_depth(text, jd_ctx=None):
    """按「能力条目的职责动词」判定 JD 要求的深度等级 (1-5)。

    设计要点（与原 skill 一致）：深度是**逐条**信号，主要看该条目自身的
    动词（战略/搭建/负责/协助…），而非整份 JD 的年限/薪资。整岗资深程度
    由独立的 overqualify 叠加信号处理，避免把「薪酬核算」误判为 L4。
    """
    base = 2
    for lvl, words in VERB_LEVELS:
        if contains_any(text, words):
            base = max(base, lvl)
    return max(1, min(5, base))


# ── 主引擎 ────────────────────────────────────────────────

class Matcher:
    def __init__(self, profile, config, embed=None, light_embed=None, api_embed=None):
        self.p = profile
        self.c = config
        self.meta = profile.get("meta", {})
        self.dims = profile.get("dimensions", {})
        self.classifier = Classifier(config, embed=embed, light_embed=light_embed, api_embed=api_embed)

    # ---- 硬过滤 ----
    def hard_filter(self, jd):
        m = self.meta
        expect_min = m.get("expected_salary_min_k")
        expect_max = m.get("expected_salary_max_k")
        sal_min, sal_max = jd.get("salary_min"), jd.get("salary_max")
        title = jd.get("title", "")
        company = jd.get("company", "")
        city = jd.get("city", "")
        desc = jd.get("description", "")
        jd_len = len(desc or "")

        hf = self.c.get("hard_filters", {})

        # 岗位方向过滤：标题必须含目标角色关键词之一
        target_roles = m.get("target_roles", [])
        if target_roles:
            title_lower = title.lower()
            has_role = any(str(r).lower() in title_lower for r in target_roles)
            if not has_role:
                return (False, "岗位方向不匹配(非目标角色)")
            # 排除非 HR 类助理岗位
            exclude_types = m.get("exclude_assistant_types", [])
            if exclude_types and contains_any(title, exclude_types):
                return (False, "岗位方向不匹配(非HR类助理)")

        # 排除岗位方向
        if hf.get("exclude_titles") and contains_any(title, m.get("exclude_titles_keywords", [])):
            return (False, "岗位方向不匹配(排除项)")

        # 薪资完全不重叠
        if hf.get("salary_no_overlap") and sal_min is not None and sal_max is not None \
                and expect_min is not None and expect_max is not None:
            if not (sal_max >= expect_min and sal_min <= expect_max):
                return (False, "薪资完全不覆盖")

        # 城市（子串匹配：上海浦东陆家嘴 视为 上海）
        if hf.get("city_mismatch") and city and city.strip():
            c = city.strip()
            ok = m.get("accept_remote", False)
            for pref in m.get("preferred_cities", []):
                if pref in c or c in pref:
                    ok = True
                    break
            if not ok:
                return (False, "城市不符")

        # 英语要求
        eng = hf.get("english_required_phrases", {})
        req_en = 0
        if contains_any(desc + title, eng.get("L3", [])):
            req_en = 3
        elif contains_any(desc + title, eng.get("L2", [])):
            req_en = 2
        if req_en:
            cand_en = m.get("language_levels", {}).get("英语", 0)
            if req_en - cand_en >= 2:
                return (False, "英语要求不达标")

        # 其他语种
        for pat in hf.get("other_language", {}).get("patterns", []):
            lang = pat["lang"]
            if contains_any(desc + title, [lang]) and contains_any(desc + title, pat.get("strong", [])):
                cand = m.get("language_levels", {}).get(lang, 0)
                if pat["level"] - cand >= 2:
                    return (False, f"{lang}要求不达标")

        # 保险营销疑似
        im = hf.get("insurance_marketing", {})
        if im and jd_len <= im.get("jd_len_max", 160) and sal_max and sal_max > im.get("salary_min_k", 18):
            region = jd.get("region", "")
            if region and contains_any(region, im.get("city_contains", [])):
                return (False, "保险营销疑似岗")

        # 僵尸标题
        ztk = hf.get("zombie_title_keywords", [])
        if ztk and count_hits(title, ztk) >= hf.get("zombie_title_min_hits", 2):
            return (False, "僵尸岗位(标题特征)")

        # 空泛+高薪
        zeh = hf.get("zombie_empty_highpay", {})
        if zeh and jd_len < zeh.get("jd_len_max", 200) and sal_max and sal_max > zeh.get("salary_min_k", 18):
            return (False, "僵尸岗位(空泛高薪)")

        # 壳公司复合
        zs = hf.get("zombie_shell", {})
        if zs and contains_any(company, zs.get("company_keywords", [])) \
                and contains_any(title, zs.get("title_keywords", [])) and jd_len < zs.get("jd_len_max", 300):
            return (False, "僵尸岗位(壳公司复合)")

        # 学历
        edu_levels = hf.get("education_required_levels", [])
        education = jd.get("education", "")
        if edu_levels and m.get("education_max") in ("本科", "大专"):
            # 检查学历要求字段
            if education and contains_any(education, ["硕士", "MBA", "博士", "研究生"]):
                return (False, "学历不符(要求硕士/博士)")
            # 也检查 JD 描述中的学历要求
            if "本科及以上" not in desc and re.search(hf.get("education_required_regex", r"(硕士|MBA|博士)及以上"), desc):
                return (False, "学历不符(要求硕士/博士)")

        # 实习生/日薪岗
        intern_pats = hf.get("intern_patterns", [])
        if intern_pats and contains_any(title + desc, intern_pats):
            return (False, "实习生/日薪岗")

        # 英语要求
        en_kw = hf.get("english_required_keywords", [])
        if en_kw and contains_any(title + desc, en_kw):
            cand_en = m.get("language_levels", {}).get("英语", 0)
            if cand_en < 2:
                return (False, "英语能力要求")

        # 小语种要求
        lang_pats = hf.get("other_language_patterns", [])
        if lang_pats and contains_any(title + desc, lang_pats):
            return (False, "小语种要求")

        # 专业资质
        cert_pats = hf.get("professional_cert_patterns", [])
        if cert_pats and contains_any(desc, cert_pats):
            return (False, "专业资质不符")

        return (True, None)

    # ---- 能力条目抽取 ----
    def extract_items(self, jd):
        desc = jd.get("description", "") or ""
        sentences = re.split(r"[。；;\n]+", desc)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 4]

        # 预批量 embedding（API模式下一次性嵌入所有句子，避免逐条调用）
        if self.classifier.api_embed and self.classifier.api_embed._built:
            all_phrases = list(sentences)
            for s in sentences:
                for sub in re.split(r"[、，,]", s):
                    sub = sub.strip()
                    if len(sub) >= 4:
                        all_phrases.append(sub)
            self.classifier.api_embed.pre_embed_sentences(all_phrases)

        items = []
        total_sentences = len(sentences)
        for s in sentences:
            dim = self.classifier.classify(s)
            if dim:
                items.append({"text": s, "dimension": dim})
        # 上限：取前 max_items
        max_items = self.c.get("scoring", {}).get("max_items", 7)
        if len(items) > max_items:
            items = items[:max_items]
        # 下限补齐：把未命中的句子按 、， 拆细再试
        min_items = self.c.get("scoring", {}).get("min_items", 5)
        confidence = "high"
        if len(items) < min_items:
            for s in sentences:
                if len(items) >= min_items:
                    break
                if s in [it["text"] for it in items]:
                    continue
                for sub in re.split(r"[、，,]", s):
                    sub = sub.strip()
                    if len(items) >= min_items:
                        break
                    if len(sub) < 4:
                        continue
                    dim = self.classifier.classify(sub)
                    if dim:
                        items.append({"text": sub, "dimension": dim})
            if len(items) < min_items:
                confidence = "low"
        # 判定每条 JD 深度
        for it in items:
            it["jd_depth"] = judge_jd_depth(it["text"], jd)
        return items, confidence, total_sentences

    # ---- 评分 ----
    def score(self, items):
        cfg = self.c.get("scoring", {})
        w1 = cfg.get("weight_first", 15)
        w2 = cfg.get("weight_second", 15)
        wr = cfg.get("weight_rest", 10)
        dc = self.c.get("depth_coefficient", {})
        blacklist = self.p.get("domain_blacklist", [])

        n = len(items)
        if n == 0:
            return {"match_pct": 0.0, "total": 0, "scored": 0,
                    "details": [], "avg_cand": 0, "avg_jd": 0,
                    "depth_label": "严重不足"}

        details = []
        total = 0
        cand_sum = 0
        jd_sum = 0
        for i, it in enumerate(items):
            dim = it["dimension"]
            jd_d = it["jd_depth"]
            weight = w1 if i == 0 else (w2 if i == 1 else wr)
            total += weight
            # 未匹配维度 -> 0 分
            if dim is None:
                details.append({
                    "dimension": "未匹配", "weight": weight, "jd_depth": jd_d,
                    "cand_depth": 0, "coeff": 0.0, "score": 0.0,
                    "domain_mismatch": False,
                })
                jd_sum += jd_d
                continue
            cand_d = self.dims.get(dim, 0)
            # 领域黑名单 -> 直接 0
            domain_mismatch = False
            for b in blacklist:
                if b.get("keyword") and b["keyword"] in it["text"]:
                    domain_mismatch = True
                    break
            if domain_mismatch:
                coeff = 0.0
            elif cand_d >= jd_d:
                coeff = dc.get("ge", 1.0)
            elif cand_d == jd_d - 1:
                coeff = dc.get("minus1", 0.5)
            else:
                coeff = dc.get("minus2plus", 0.0)
            sc = weight * coeff
            details.append({
                "dimension": dim, "weight": weight, "jd_depth": jd_d,
                "cand_depth": cand_d, "coeff": coeff, "score": sc,
                "domain_mismatch": domain_mismatch,
            })
            cand_sum += cand_d
            jd_sum += jd_d

        match_pct = (sum(d["score"] for d in details) / total * 100) if total else 0.0
        n_matched = sum(1 for d in details if d["dimension"] != "未匹配")
        avg_cand = cand_sum / n_matched if n_matched > 0 else 0
        avg_jd = jd_sum / len(details) if details else 0
        diff = round(avg_cand - avg_jd)
        if diff >= 2:
            label = "严重超配"
        elif diff == 1:
            label = "轻度超配"
        elif diff == 0:
            label = "精准匹配"
        elif diff == -1:
            label = "轻度不足"
        else:
            label = "严重不足"
        return {"match_pct": round(match_pct, 1), "total": total,
                "scored": sum(d["score"] for d in details),
                "details": details, "avg_cand": avg_cand, "avg_jd": avg_jd,
                "depth_label": label}

    # ---- 叠加信号 ----
    def override_signals(self, jd, sc):
        sig = {}
        m = self.meta
        expect_min = m.get("expected_salary_min_k")
        sal_max = jd.get("salary_max")
        ov = self.c.get("override_signals", {})
        if ov.get("salary_severe") and sal_max is not None and expect_min is not None:
            if sal_max < expect_min * 0.9:
                sig["salary_severe"] = True
        if ov.get("overqualify"):
            if sc["details"] and (sc["avg_cand"] - sc["avg_jd"]) >= 2:
                sig["overqualify"] = True
        if ov.get("exp_mismatch"):
            exp_max = jd.get("exp_max")
            if exp_max is not None and m.get("total_years") and exp_max * 2 < m["total_years"]:
                sig["exp_mismatch"] = True
        if ov.get("ai_boost", {}).get("enabled") and ov["ai_boost"].get("scope") == "title":
            if contains_any(jd.get("title", ""), self.c.get("ai_keywords", [])):
                sig["ai_boost"] = True
        return sig

    # ---- 二次复核 ----
    def review_pass(self, jd, level, reason_extra):
        if level not in ("high", "medium"):
            return level, reason_extra
        rr = self.c.get("review_rules", {})
        title = jd.get("title", "")
        desc = jd.get("description", "")
        jd_len = len(desc or "")

        # A1 岗位级别错配
        a1 = rr.get("A1", {})
        if contains_any(title, a1.get("titles", [])) and not contains_any(title, a1.get("exclude", [])):
            return "low", (reason_extra + "；[复核降级]岗位级别错配：非助理岗")
        # B3 JD 空泛
        b3 = rr.get("B3", {})
        if b3 and jd_len < b3.get("jd_len_max", 200):
            hits = count_hits(desc, b3.get("universal_phrases", []))
            if hits >= b3.get("min_universal_hits", 3):
                return "low", (reason_extra + "；[复核降级]JD内容空泛")
        # F1 专业资质
        f1 = rr.get("F1", {})
        if f1 and contains_any(desc, f1.get("certs", [])):
            return "low", (reason_extra + "；[复核降级]专业资质不符")
        # F3 行业经验
        f3 = rr.get("F3", {})
        if f3:
            for ind in f3.get("industries", []):
                if ind in desc and ind not in self.p.get("domain_coverage", []):
                    return "low", (reason_extra + f"；[复核降级]行业经验不符：{ind}")
        return level, reason_extra

    # ---- 理由生成 ----
    def build_reason(self, jd, level, sc, sig, hard_reason):
        if hard_reason:
            return f"硬过滤：{hard_reason}"
        pct = sc["match_pct"]
        # core：取最高分与最低分维度做概括
        if sc["details"]:
            best = max(sc["details"], key=lambda d: d["coeff"])
            worst = min(sc["details"], key=lambda d: d["coeff"])
            core = f"{best['dimension']}匹配" if best["coeff"] >= 1.0 else f"{worst['dimension']}不足"
        else:
            core = "能力无匹配项"
        extra = []
        if sig.get("salary_severe"):
            extra.append("薪资不足降级")
        if sig.get("overqualify"):
            extra.append("严重超配降级")
        if sig.get("exp_mismatch"):
            extra.append("经验错配降级")
        if sig.get("ai_boost"):
            extra.append("AI加分")
        extra_s = "，".join(extra)
        reason = f"匹配度{pct}%，{core}"
        if extra_s:
            reason += f"，{extra_s}"
        return reason

    # ---- 总入口 ----
    def evaluate(self, jd):
        passed, hard_reason = self.hard_filter(jd)
        if not passed:
            return {
                "优先级判断": "low",
                "匹配度": 0.0,
                "优先级判断理由": f"硬过滤：{hard_reason}",
                "能力条目得分": "",
                "叠加信号": "",
                "平均JD深度": "",
                "平均候选人深度": "",
                "hard_filter": hard_reason,
            }
        items, confidence, total_sentences = self.extract_items(jd)
        sc = self.score(items)
        # 基础判定
        thr = self.c.get("scoring", {}).get("thresholds", {})
        if sc["match_pct"] >= thr.get("high", 80):
            base = "high"
        elif sc["match_pct"] >= thr.get("medium", 60):
            base = "medium"
        else:
            base = "low"
        # 匹配质量检查：如果匹配条目太少，降级
        if len(items) < 4 and base == "high":
            base = "medium"
        elif len(items) < 3 and base == "medium":
            base = "low"
        # 维度覆盖检查：如果匹配的维度太少，降级
        if sc["details"]:
            unique_dims = set(d["dimension"] for d in sc["details"])
            if len(unique_dims) <= 1 and base == "high":
                base = "medium"
        # 标题相关性检查：标题不含明确 HR/总助 关键词时，降级
        title = jd.get("title", "")
        title_lower = title.lower()
        strong_hr_keywords = ["总助", "ceo助理", "总裁助理", "董事长助理", "总经理助理", "hrbp", "人事", "行政", "人事行政", "人力资源", "人事专员", "人事主管", "人事经理", "行政专员", "行政主管", "行政经理", "招聘专员", "培训专员", "薪酬专员", "人事总监", "行政总监", "hrd", "hrm", "cho", "秘书", "hr", "ea", "executive assistant"]
        has_strong_hr = any(str(k).lower() in title_lower for k in strong_hr_keywords)
        if not has_strong_hr and base in ("high", "medium"):
            base = "low"
        # 总经理助理/总裁助理 类岗位：需要有战略/组织/数据等核心能力才能评为 high
        gm_keywords = ["总经理助理", "总裁助理", "董事长助理", "总助"]
        is_gm_assistant = any(k in title for k in gm_keywords)
        if is_gm_assistant and base == "high":
            core_dims = {"战略能力", "组织能力_体系搭建", "目标与绩效管理", "数据能力", "AI与自动化", "飞书生态"}
            matched_dims = set(d["dimension"] for d in sc["details"])
            core_matched = matched_dims & core_dims
            if len(core_matched) < 2:
                base = "medium"
        # 叠加信号
        sig = self.override_signals(jd, sc)
        rank = LEVEL_RANK[base]
        for _ in range(sum(1 for k in ("salary_severe", "overqualify", "exp_mismatch") if sig.get(k))):
            rank = max(0, rank - 1)
        if sig.get("ai_boost"):
            rank = min(2, rank + 1)
        level = RANK_LEVEL[rank]
        # 二次复核
        level, extra = self.review_pass(jd, level, "")
        reason = self.build_reason(jd, level, sc, sig, hard_reason=None)
        if extra:
            reason = reason + extra
        # 能力条目得分串
        item_str = " ".join(
            f"{d['dimension']}{d['score']:.0f}/{d['weight']}" for d in sc["details"]
        )
        # 叠加信号文本
        sig_parts = []
        if sig.get("salary_severe"):
            sig_parts.append("薪资不足")
        if sig.get("overqualify"):
            sig_parts.append("能力超配")
        if sig.get("exp_mismatch"):
            sig_parts.append("经验错配")
        if sig.get("ai_boost"):
            sig_parts.append("AI提升")
        sig_text = "、".join(sig_parts) if sig_parts else "无"
        # 平均深度
        avg_jd = sc["avg_jd"]
        avg_cand = sc["avg_cand"]
        avg_jd_str = f"L{int(round(avg_jd))}" if avg_jd > 0 else ""
        avg_cand_str = f"L{int(round(avg_cand))}" if avg_cand > 0 else ""
        return {
            "优先级判断": level,
            "匹配度": sc["match_pct"],
            "优先级判断理由": reason[:30],
            "能力条目得分": item_str,
            "叠加信号": sig_text,
            "平均JD深度": avg_jd_str,
            "平均候选人深度": avg_cand_str,
            "_confidence": confidence,
            "_items": len(items),
        }


# ── 配置加载 ──────────────────────────────────────────────

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_profile(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def enrich_jd(row, profile_meta=None):
    """把一行 CSV/表格记录标准化为引擎可用的 jd dict（含解析后的薪资/经验）。
    
    支持两种列名格式：
    1. 原型格式：岗位名称, 薪资范围, 工作地点, JD描述
    2. Boss直聘实际格式：职位名称, 薪资, 城市, 职位描述
    """
    def pick(keys):
        for k in keys:
            if k in row and row[k] not in (None, ""):
                return str(row[k])
        return ""
    title = pick(["职位名称", "岗位名称", "title", "job_title"])
    company = pick(["公司名称", "company"])
    city = pick(["城市", "工作地点", "city", "location"])
    salary = pick(["薪资", "薪资范围", "salary"])
    description = pick(["职位描述", "JD描述", "description", "job_desc"])
    job_id = pick(["job_id", "岗位id", "id"])
    region = pick(["区域", "region"])
    education = pick(["学历要求", "education"])
    sal_min, sal_max = parse_salary(salary)
    exp_min, exp_max = parse_experience(pick(["经验要求", "experience"]))
    return {
        "job_id": job_id,
        "title": title,
        "company": company,
        "city": city,
        "region": region,
        "salary_raw": salary,
        "salary_min": sal_min,
        "salary_max": sal_max,
        "exp_min": exp_min,
        "exp_max": exp_max,
        "description": description,
        "education": education,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--embed", action="store_true", help="启用本地 embedding 分类器")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    config = load_config(args.config)
    embed = None
    if args.embed:
        try:
            from embed_classify import LocalEmbeddingClassifier
            embed = LocalEmbeddingClassifier(config)
            print("[embed] 本地 embedding 分类器已加载" if embed.is_available
                  else "[embed] sentence-transformers 不可用，回退关键词")
        except Exception as e:
            print(f"[embed] 加载失败，回退关键词：{e}")
    matcher = Matcher(profile, config, embed=embed)

    # 简单自测：直接传 JSON 行
    print("引擎加载成功；用 run_match.py 跑 CSV，或用 matcher.evaluate(jd) 编程调用。")
