# job-match-priority — JD 匹配优先级判定

> 🦞 Boss 直聘 JD 自动匹配候选人优先级：high / medium / low

---

## 触发词

当用户提到以下关键词时激活本 skill：

> 匹配度 / 优先级判定 / JD 筛选 / 职位匹配 / 投递优先 / 岗位评估 / 匹配分析

---

## 一句话定位

把 Boss 直聘爬取的 JD，自动判定与候选人的匹配优先级。Agent 仅一次编译候选人画像，纯 Python 引擎批量处理，零 LLM 调用。

---

## 前置条件

1. 已准备 JD 数据（CSV 格式，含 `职位名称` `薪资` `职位描述` 等列）
2. 已配置候选人画像（`config/candidate_profile.json`）
3. （可选）已配置 API Key 用于语义 embedding

---

## 工作流程

### Step 0：首次使用 — 配置环境 + 编译候选人画像

```bash
cd <skill目录>
python scripts/setup_wizard.py
```

引导脚本会自动：
1. 检测环境依赖
2. 配置 API Key（SiliconFlow / 自定义 / 跳过）
3. **交互式编译候选人画像**（行业经验/能力深度/排除项）
4. 验证配置
5. 运行示例测试

> 💡 也可跳过引导，直接编辑 `config/candidate_profile.json`

### Step 1：编译候选人画像（仅首次 / 换人时）

当用户首次使用或更换候选人时：

1. 收集候选人信息：
   - 目标岗位类型
   - 期望薪资范围
   - 期望城市
   - 学历 / 语言能力
   - 行业经验（行业 + 年限 + 岗位类型）
   - 专业资质（有 / 无）

2. **产出 `config/candidate_profile.json`**

3. **产出 `config/match_config.yaml`**（如需调整规则）

### Step 2：批量匹配（日常使用）

```bash
python scripts/run_match.py \
  --profile config/candidate_profile.json \
  --config config/match_config.yaml \
  --input <JD文件.csv> \
  --out <结果文件.csv> \
  --api-embed
```

### Step 3：查看结果

输出文件包含原始数据 + 新增列：
- `优先级判断`：high / medium / low
- `匹配度`：0-100%
- `优先级判断理由`：简短原因
- `能力条目得分`：每条能力的分数
- `叠加信号`：降级/加级信号

---

## 运行模式

| 模式 | 参数 | 精度 | 适用场景 |
|------|------|------|---------|
| 关键词 | 默认 | ⭐⭐⭐ | 快速验证、无网络 |
| TF-IDF | `--light-embed` | ⭐⭐⭐⭐ | 离线环境 |
| **API Embedding** | `--api-embed` | ⭐⭐⭐⭐⭐ | **日常推荐** |

---

## 配置文件说明

| 文件 | 位置 | 用途 |
|------|------|------|
| `candidate_profile.json` | `config/` | 候选人画像（能力深度/薪资/行业经验） |
| `match_config.yaml` | `config/` | 评分规则（阈值/硬过滤/叠加信号） |
| `taxonomy.yaml` | `config/` | 语义维度词表（能力分类锚点） |
| `config.local.yaml` | `config/` | API Key 等本地配置（不提交 Git） |

---

## 关键规则（Agent 必读）

1. **CSV 行数必须用 `csv.DictReader` 统计**，禁止 `wc -l`（JD 描述含换行）
2. **深度系数严格按差距**：低1级=50%，低2级+=0%
3. **AI 提升仅检测 title**（见 `match_config.yaml` 的 `ai_boost.scope`）
4. **能力条目按条目计分**，不按关键词；权重仅按位置分配
5. **结果文件基于源文件追加字段**，不丢弃原始数据

---

## 输出示例

```csv
职位名称,薪资,优先级判断,匹配度,优先级判断理由
CEO助理,15-18K,high,93.8%,战略能力匹配
销售代表,15-20K,low,0.0%,硬过滤：岗位方向不匹配
行政助理,16-20K,low,0.0%,硬过滤：僵尸岗位(标题特征)
```

---

## 依赖

```
# 必需
pyyaml

# API Embedding 模式（推荐）
requests

# TF-IDF 本地模式
faiss-cpu
numpy
```

---

## 目录结构

```
job-match-priority/
├── SKILL.md                    # 本文档（Agent 入口）
├── README.md                   # 用户文档
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
