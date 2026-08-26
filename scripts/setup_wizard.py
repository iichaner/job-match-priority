#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_wizard.py —— 首次使用引导脚本

帮助新用户完成初始配置：
1. 检测环境依赖
2. 配置 API Key（SiliconFlow 或自定义）
3. 编译候选人画像（交互式问答）
4. 验证配置
5. 运行示例测试

用法：
    python setup_wizard.py
"""

import os
import sys
import json
import yaml
from pathlib import Path

# 颜色输出
def green(text): return f"\033[92m{text}\033[0m"
def red(text): return f"\033[91m{text}\033[0m"
def yellow(text): return f"\033[93m{text}\033[0m"
def bold(text): return f"\033[1m{text}\033[0m"
def dim(text): return f"\033[2m{text}\033[0m"

BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
SAMPLE_DIR = BASE_DIR / "sample"

def check_env():
    """Step 1: 检测环境依赖"""
    print(bold("\n📋 Step 1: 检测环境依赖\n"))
    issues = []

    py_ver = sys.version_info
    if py_ver >= (3, 10):
        print(f"  {green('✓')} Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    else:
        print(f"  {red('✗')} Python {py_ver.major}.{py_ver.minor}（需要 3.10+）")
        issues.append("python")

    try:
        import yaml
        print(f"  {green('✓')} PyYAML")
    except ImportError:
        print(f"  {red('✗')} PyYAML 未安装 → pip install pyyaml")
        issues.append("pyyaml")

    try:
        import requests
        print(f"  {green('✓')} requests（API 模式）")
    except ImportError:
        print(f"  {yellow('⚠')} requests 未安装 → pip install requests")

    try:
        import faiss
        print(f"  {green('✓')} faiss-cpu（TF-IDF 模式）")
    except ImportError:
        print(f"  {yellow('⚠')} faiss-cpu 未安装 → pip install faiss-cpu")

    return issues

def setup_api_key():
    """Step 2: 配置 API Key"""
    print(bold("\n🔑 Step 2: 配置 API Key\n"))
    print("  支持的 Embedding 供应商：")
    print("    1. SiliconFlow（推荐，免费 tier 每月 100 万 token）")
    print("    2. 自定义 OpenAI 兼容 API")
    print("    3. 跳过（使用关键词模式，无需 API）\n")

    choice = input("  请选择 [1/2/3]（默认 1）: ").strip() or "1"

    config_path = CONFIG_DIR / "config.local.yaml"

    if choice == "1":
        print(f"\n  {bold('SiliconFlow 配置指南：')}")
        print("  1. 访问 https://cloud.siliconflow.cn/ 注册账号")
        print("  2. 进入「API 密钥」页面，创建新密钥")
        print("  3. 复制密钥（sk-xxx 格式）\n")

        api_key = input("  请粘贴你的 SiliconFlow API Key: ").strip()
        if not api_key:
            print(f"  {yellow('跳过，稍后可手动编辑 config/config.local.yaml')}")
            return "keyword"

        model = input("  模型名称（默认 BAAI/bge-large-zh-v1.5）: ").strip() or "BAAI/bge-large-zh-v1.5"

        config = {
            "siliconflow": {"api_key": api_key, "model": model},
            "default_mode": "api-embed"
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        print(f"\n  {green('✓')} API Key 已保存到 config/config.local.yaml")
        return "api-embed"

    elif choice == "2":
        print(f"\n  {bold('自定义 API 配置：')}\n")
        api_key = input("  API Key: ").strip()
        base_url = input("  Base URL（默认 https://api.siliconflow.cn/v1）: ").strip() or "https://api.siliconflow.cn/v1"
        model = input("  模型名称（默认 BAAI/bge-large-zh-v1.5）: ").strip() or "BAAI/bge-large-zh-v1.5"

        config = {
            "custom_api": {"api_key": api_key, "base_url": base_url, "model": model},
            "default_mode": "api-embed"
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        print(f"\n  {green('✓')} 配置已保存")
        return "api-embed"

    else:
        print(f"\n  {yellow('跳过，使用关键词模式')}")
        return "keyword"

def build_candidate_profile():
    """Step 3: 交互式编译候选人画像"""
    print(bold("\n👤 Step 3: 编译候选人画像\n"))
    print("  回答以下问题，生成你的候选人配置文件。")
    print("  直接回车跳过可选项，稍后可手动编辑 config/candidate_profile.json\n")

    profile = {"meta": {}, "dimensions": {}}

    # ── 基础信息 ──
    print(bold("  【基础信息】"))
    name = input("  候选人名称（默认：候选人）: ").strip() or "候选人"
    profile["meta"]["name"] = name

    target = input("  目标岗位（逗号分隔，如：总助,CEO助理,HRBP）: ").strip()
    profile["meta"]["target_roles"] = [x.strip() for x in target.split(",") if x.strip()] if target else ["总助", "CEO助理"]

    salary_min = input("  期望最低薪资（K，如 15）: ").strip()
    salary_max = input("  期望最高薪资（K，如 20）: ").strip()
    profile["meta"]["expected_salary_min_k"] = int(salary_min) if salary_min else 15
    profile["meta"]["expected_salary_max_k"] = int(salary_max) if salary_max else 20

    cities = input("  期望城市（逗号分隔，如：上海,杭州）: ").strip()
    profile["meta"]["preferred_cities"] = [x.strip() for x in cities.split(",") if x.strip()] if cities else ["上海"]

    edu = input("  最高学历（博士/硕士/本科/大专，默认 本科）: ").strip() or "本科"
    profile["meta"]["education_max"] = edu

    years = input("  总工作年限（如 8）: ").strip()
    profile["meta"]["total_years"] = int(years) if years else 5

    # ── 语言能力 ──
    print(bold("\n  【语言能力】"))
    print("  等级：0=无 1=基础 2=良好 3=流利 4=母语")
    en = input("  英语等级（默认 1）: ").strip()
    profile["meta"]["language_levels"] = {"英语": int(en) if en else 1}

    # ── 行业经验 ──
    print(bold("\n  【行业经验】"))
    print("  格式：行业名 年限 岗位1,岗位2,...（每行一条，空行结束）")
    print("  示例：咨询 3 助教,培训,项目管理")
    print("  示例：消费品/电商 3 总助,销售助理,人力")

    industry_exp = []
    role_accum = {}
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split()
        if len(parts) >= 3:
            industry = parts[0]
            years = int(parts[1])
            roles = [r.strip() for r in parts[2].split(",")]
            industry_exp.append({"industry": industry, "years": years, "roles": roles})
            for r in roles:
                role_accum[r] = role_accum.get(r, 0) + years
        elif len(parts) == 2:
            industry = parts[0]
            years = int(parts[1])
            industry_exp.append({"industry": industry, "years": years, "roles": []})

    profile["meta"]["industry_experience"] = industry_exp
    profile["meta"]["role_accumulation"] = role_accum

    # ── 无经验行业 ──
    print(bold("\n  【无经验行业（硬性要求时判 LOW）】"))
    no_ind = input("  逗号分隔（如：金融,芯片,半导体）: ").strip()
    profile["meta"]["hard_filter_industries"] = [x.strip() for x in no_ind.split(",") if x.strip()] if no_ind else []

    # ── 能力维度深度 ──
    print(bold("\n  【能力维度深度评分】"))
    print("  等级：1=了解 2=执行 3=负责 4=主导 5=专家")
    print("  直接回车使用默认值\n")

    default_dims = {
        "战略能力": 3, "目标与绩效管理": 3, "组织能力_体系搭建": 3,
        "招聘与配置": 3, "薪酬福利": 3, "组织能力_人才发展": 3,
        "数据能力": 3, "AI与自动化": 3, "飞书生态": 3,
        "项目与流程管理": 3, "助理职能": 3, "员工关系": 3,
    }

    for dim, default in default_dims.items():
        while True:
            val = input(f"  {dim}（默认 {default}）: ").strip()
            if not val:
                profile["dimensions"][dim] = default
                break
            try:
                profile["dimensions"][dim] = int(val)
                break
            except ValueError:
                print(f"  {red('请输入数字 1-5')}")

    # ── 排除项 ──
    print(bold("\n  【排除岗位（标题含这些词时判 LOW）】"))
    exclude = input("  逗号分隔（如：纯销售,电话销售）: ").strip()
    profile["meta"]["exclude_titles_keywords"] = [x.strip() for x in exclude.split(",") if x.strip()] if exclude else ["纯销售", "电话销售"]

    # ── 写入文件 ──
    profile_path = CONFIG_DIR / "candidate_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"\n  {green('✓')} 候选人画像已保存到 config/candidate_profile.json")
    print(f"  {dim('  稍后可手动编辑调整')}")
    return True

def verify_config():
    """Step 4: 验证配置"""
    print(bold("\n🔍 Step 4: 验证配置\n"))

    files = [
        ("candidate_profile.json", CONFIG_DIR / "candidate_profile.json"),
        ("match_config.yaml", CONFIG_DIR / "match_config.yaml"),
        ("taxonomy.yaml", CONFIG_DIR / "taxonomy.yaml"),
    ]

    ok = True
    for name, path in files:
        if path.exists():
            print(f"  {green('✓')} {name}")
        else:
            print(f"  {red('✗')} {name} 不存在")
            ok = False

    return ok

def run_sample(mode):
    """Step 5: 运行示例测试"""
    print(bold("\n🧪 Step 5: 运行示例测试\n"))

    sample_path = SAMPLE_DIR / "jds_sample.csv"
    if not sample_path.exists():
        print(f"  {red('✗')} 测试样本不存在")
        return

    # 读取 API Key
    api_key = ""
    config_path = CONFIG_DIR / "config.local.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        api_key = cfg.get("siliconflow", {}).get("api_key", "") or \
                  cfg.get("custom_api", {}).get("api_key", "")
    api_key = api_key or os.environ.get("SILICONFLOW_API_KEY", "")

    cmd_parts = [
        sys.executable, str(BASE_DIR / "scripts" / "run_match.py"),
        "--profile", str(CONFIG_DIR / "candidate_profile.json"),
        "--config", str(CONFIG_DIR / "match_config.yaml"),
        "--input", str(sample_path),
        "--out", "/tmp/wizard_test.csv",
    ]

    if mode == "api-embed" and api_key:
        cmd_parts.extend(["--api-embed", "--api-key", api_key])
    elif mode == "light-embed":
        cmd_parts.append("--light-embed")

    print(f"  模式: {mode}")
    ret = os.system(" ".join(cmd_parts))
    if ret == 0:
        import csv
        with open("/tmp/wizard_test.csv", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        from collections import Counter
        dist = Counter(r.get("优先级判断", "") for r in rows)
        print(f"\n  {green('✓')} 测试通过！结果: {dict(dist)}")
    else:
        print(f"\n  {red('✗')} 测试失败")

def main():
    print(bold("=" * 50))
    print(bold("  Job Match Priority — 首次使用引导"))
    print(bold("=" * 50))

    # Step 1: 环境检测
    issues = check_env()
    if "python" in issues or "pyyaml" in issues:
        print(f"\n{red('环境依赖不满足，请先安装后重新运行')}")
        sys.exit(1)

    # Step 2: API Key
    mode = setup_api_key()

    # Step 3: 候选人画像
    print()
    build_choice = input("  是否现在编译候选人画像？[Y/n]（默认 Y）: ").strip().lower()
    if build_choice != "n":
        build_candidate_profile()

    # Step 4: 验证
    if not verify_config():
        print(f"\n{red('配置验证失败')}")
        sys.exit(1)

    # Step 5: 测试
    run_sample(mode)

    # 完成
    print(bold("\n" + "=" * 50))
    print(green("  🎉 配置完成！"))
    print(bold("=" * 50))
    print(f"""
  现在可以运行匹配了：

    python scripts/run_match.py \\
      --profile config/candidate_profile.json \\
      --config config/match_config.yaml \\
      --input your_jds.csv \\
      --out result.csv \\
      --api-embed

  详细文档请查看 README.md
""")

if __name__ == "__main__":
    main()
