# Job Match Priority — Boss直聘 JD 匹配优先级判定

> 🦞 基于 KSAO 框架的加权评分匹配引擎。Agent 仅一次编译候选人画像，纯 Python 确定性引擎批量复用，匹配阶段零 LLM 调用。

---

## 目录

- [功能简介](#功能简介)
- [快速开始](#快速开始)
- [架构设计](#架构设计)
- [配置说明](#配置说明)
- [运行模式](#运行模式)
- [输入输出](#输入输出)
- [判定逻辑](#判定逻辑)
- [候选人画像配置](#候选人画像配置)
- [常见问题](#常见问题)
- [目录结构](#目录结构)

---

## 功能简介

**解决的问题**：Boss 直聘爬取的大量 JD，人工逐条筛选效率低，需要自动判定与候选人的匹配优先级。

**核心能力**：
- 自动判定每条 JD 的优先级：`high` / `medium` / `low`
- 输出量化匹配度（0-100%）和判定理由
- 支持硬过滤、能力匹配评分、叠加信号修正、二次复核
- 候选人画像可复用，换人只换 profile 文件

**技术特点**：
- **零 LLM 调用**：匹配阶段不消耗任何 API token
- **确定性可复现**：同一输入必定得到同一输出
- **本地运行**：数据不出机，隐私安全

---

## 快速开始

### 方式一：引导脚本（推荐新手）

```bash
cd job-match-priority
python scripts/setup_wizard.py
```

引导脚本会自动检测环境、配置 API Key、验证配置、运行示例测试。

### 方式二：手动配置

#### 1. 安装依赖

```bash
pip install pyyaml requests  # 必需
pip install faiss-cpu numpy   # 可选：TF-IDF 模式
```

#### 2. 配置 API Key

```bash
# 方式1：编辑配置文件
cp config/config.example.yaml config/config.local.yaml
# 编辑 config/config.local.yaml，填入 API Key

# 方式2：环境变量
export SILICONFLOW_API_KEY="***"
```

#### 3. 运行匹配

```bash
python scripts/run_match.py \
  --profile config/candidate_profile.json \
  --config config/match_config.yaml \
  --input your_jds.csv \
  --out result.csv \
  --api-embed
```

---

## 架构设计

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Agent 编译（仅一次）                         │
│                                                     │
│  候选人简历 + 偏好 ──→ candidate_profile.json          │
│  业务规则 + 阈值   ──→ match_config.yaml              │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Layer 2: Python 确定性引擎（零 LLM）                  │
│                                                     │
│  输入: JD CSV + profile + config                     │
│  输出: 源数据 + 优先级判断 + 匹配度 + 理由             │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Layer 3: 跨候选人复用                                │
│                                                     │
│  换人 = 换 candidate_profile.json                     │
└─────────────────────────────────────────────────────┘
```

---

## 配置说明

### API Key 配置优先级

| 优先级 | 来源 | 说明 |
|--------|------|------|
| 1 | `--api-key` 命令行参数 | 最高优先 |
| 2 | `SILICONFLOW_API_KEY` 环境变量 | CI/CD 推荐 |
| 3 | `config/config.local.yaml` | 本地开发推荐 |

### 替换向量供应商

编辑 `config/config.local.yaml`：

```yaml
# SiliconFlow（默认，免费 tier 每月 100 万 token）
siliconflow:
  api_key: "***"
  model: "BAAI/bge-large-zh-v1.5"

# 或 OpenAI 兼容 API
# custom_api:
#   api_key: "***"
#   base_url: "https://api.openai.com/v1"
#   model: "text-embedding-3-small"
```

---

## 运行模式

| 模式 | 参数 | 精度 | 速度 |
|------|------|------|------|
| 关键词 | 默认 | ⭐⭐⭐ | 最快 |
| TF-IDF | `--light-embed` | ⭐⭐⭐⭐ | 快 |
| **API Embedding** | `--api-embed` | ⭐⭐⭐⭐⭐ | 中等 |

---

## 输入输出

### 输入 CSV 列要求

```csv
职位名称,薪资,经验要求,学历要求,公司名称,城市,区域,job_id,职位描述,创建日期
```

### 输出列

| 列 | 含义 |
|----|------|
| `优先级判断` | `high` / `medium` / `low` |
| `匹配度` | 0-100% |
| `优先级判断理由` | 简短原因 |
| `能力条目得分` | 每条能力的分数 |
| `叠加信号` | 降级/加级信号 |

---

## 判定逻辑

```
Phase 1: 硬过滤（一票否决）
  ├─ 岗位方向 / 薪资 / 城市 / 学历 / 英语
  ├─ 保险营销 / 僵尸标题 / 空泛高薪 / 壳公司
  ├─ 行业经验不符（硬性要求）
  └─ 证书不符（硬性要求）
  ↓

Phase 2: 能力条目抽取 + 归类 + 深度判定
  ├─ 关键词 / TF-IDF / API Embedding
  └─ 得到: [{维度, 深度}, ...]
  ↓

Phase 3: 加权评分 → 匹配度 0-100%
  ↓

Phase 4: 基础判定（≥95% high / ≥90% medium）
  ↓

Phase 5: 叠加信号修正（薪资不足降级 / AI加分）
  ↓

Phase 6: 二次复核（空泛JD / 岗位错配 / 资质不符 → LOW）
```

---

## 候选人画像配置

详见 `config/candidate_profile.json`，关键字段：

- `meta.target_roles`：目标岗位类型
- `meta.industry_experience`：行业经验（行业 + 年限 + 岗位类型）
- `meta.role_accumulation`：岗位跨行业累计年限
- `dimensions`：各能力维度深度（L1-L4）
- `meta.hard_filter_industries`：无经验行业（硬性要求时 LOW）

---

## 常见问题

| 问题 | 解决 |
|------|------|
| 报错 "需要 API Key" | 配置 `config/config.local.yaml` 或设环境变量 |
| CSV 只读到1行 | JD 描述含换行，本工具已处理（用 csv.DictReader） |
| 匹配度100%但判定 LOW | 触发二次复核，检查 `优先级判断理由` 列 |
| 如何调整灵敏度 | 修改 `config/match_config.yaml` 的 `thresholds` |

---

## 目录结构

```
job-match-priority/
├── SKILL.md                    # Agent 入口文档
├── README.md                   # 本文档
├── scripts/
│   ├── run_match.py            # CLI 入口
│   ├── matcher.py              # 核心引擎
│   ├── semantic_match_api.py   # API Embedding
│   ├── semantic_match_light.py # TF-IDF 本地
│   └── setup_wizard.py         # 首次使用引导
├── config/
│   ├── candidate_profile.json  # 候选人画像
│   ├── match_config.yaml       # 评分规则
│   ├── taxonomy.yaml           # 语义维度词表
│   ├── config.example.yaml     # 配置模板
│   └── config.local.yaml       # 本地配置（不提交）
├── sample/
│   ├── jds_sample.csv          # 测试样本
│   └── jds_sample_判定结果.csv  # 期望结果
└── .gitignore
```

---

## License

MIT
