"""
评测脚本：对指定 prompt 版本跑评测集，输出每题得分与总分。
支持两种评分方式：
  1. 关键词命中率（快速、免费）
  2. LLM-as-Judge（更准确，需要额外 API 调用）

用法:
  uv run eval/run_eval.py              # 评测最新版本 prompt
  uv run eval/run_eval.py --version v1 # 评测指定版本
  uv run eval/run_eval.py --compare v1 v2  # 对比两个版本
  uv run eval/run_eval.py --list      # 列出所有可用版本
  uv run eval/run_eval.py --version v1 --llm-judge  # 使用 LLM 裁判评分
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime

# 将项目根目录加入 path，以便导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent

from knowledge import load_or_rebuild_vectorstore
from prompt_manager import load_prompt, load_latest_prompt, list_versions

load_dotenv()

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CASES_PATH = os.path.join(EVAL_DIR, "test_cases.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")


def init_llm():
    return ChatOpenAI(
        model="qwen3.6-plus",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def init_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-v4",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        check_embedding_ctx_length=False,
    )


def keyword_score(answer: str, expected_keywords: list[str]) -> float:
    """关键词命中率评分：命中比例 * 5，满分 5 分"""
    if not expected_keywords:
        return -1  # 无关键词要求（如天气类），跳过
    hits = sum(1 for kw in expected_keywords if kw in answer)
    return round(hits / len(expected_keywords) * 5, 2)


async def llm_judge_score(llm, question: str, expected_keywords: list[str], answer: str) -> float:
    """LLM 裁判评分：让大模型给回答打 1-5 分"""
    expected_text = "、".join(expected_keywords) if expected_keywords else "（无特定关键词要求）"
    response = await llm.ainvoke([
        {"role": "system", "content": "你是一个严格的评测裁判。根据用户问题和期望要点，给 AI 回答打分（1-5分）。只输出一个数字。"},
        {"role": "user", "content": f"问题: {question}\n期望要点: {expected_text}\nAI回答: {answer}\n\n请打分（1-5）:"},
    ])
    try:
        # 从回复中提取数字
        score_text = response.content.strip()
        for ch in score_text:
            if ch.isdigit():
                return min(5, max(1, int(ch)))
        return 3  # 解析失败给中间分
    except Exception:
        return 3


async def run_eval_for_version(version: str, use_llm_judge: bool = False):
    """对指定 prompt 版本运行完整评测"""
    print(f"\n{'='*60}")
    print(f"  评测 Prompt 版本: {version}")
    print(f"{'='*60}")

    # 加载评测集
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    llm = init_llm()
    embeddings = init_embeddings()
    vectorstore = load_or_rebuild_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_knowledge_base",
        description="用于搜索公司内部知识库，查找产品信息、公司政策、常见问题等。",
    )

    system_prompt = load_prompt(version)
    agent = create_agent(
        model=llm,
        tools=[retriever_tool],  # 评测时只用知识库工具，避免 MCP 连接开销
        system_prompt=system_prompt,
    )

    results = []
    for i, case in enumerate(test_cases):
        # 跳过需要 MCP 工具的测试（评测环境不连接外部服务）
        if case.get("expected_tool"):
            print(f"  [{i+1}/{len(test_cases)}] ⏭ 跳过 (需要MCP工具): {case['question']}")
            continue

        print(f"  [{i+1}/{len(test_cases)}] 测试: {case['question']}", end=" ... ")
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": case["question"]}]},
            config={"configurable": {"thread_id": f"eval-{i}"}},
        )
        answer = response["messages"][-1].content

        # 评分
        kw_score = keyword_score(answer, case.get("expected_keywords", []))
        llm_score = None
        if use_llm_judge and kw_score >= 0:
            llm_score = await llm_judge_score(llm, case["question"], case["expected_keywords"], answer)

        final_score = llm_score if llm_score is not None else kw_score
        print(f"得分: {final_score}")

        results.append({
            "question": case["question"],
            "category": case["category"],
            "answer": answer,
            "keyword_score": kw_score,
            "llm_score": llm_score,
            "final_score": final_score,
        })

    # 统计
    scored = [r for r in results if r["final_score"] >= 0]
    avg_score = sum(r["final_score"] for r in scored) / len(scored) if scored else 0

    print(f"\n{'─'*60}")
    print(f"  版本: {version}  |  平均分: {avg_score:.2f}/5  |  测试数: {len(scored)}")
    print(f"{'─'*60}")
    for r in results:
        status = "✓" if r["final_score"] >= 4 else ("△" if r["final_score"] >= 3 else "✗")
        print(f"  {status} [{r['category']}] {r['question']}  →  {r['final_score']}")

    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULTS_DIR, f"{version}_{timestamp}.json")
    output = {
        "version": version,
        "timestamp": timestamp,
        "avg_score": avg_score,
        "total_cases": len(scored),
        "results": results,
    }
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {result_file}")

    return output


async def compare_versions(versions: list[str], use_llm_judge: bool = False):
    """对比多个 prompt 版本的评测结果"""
    all_results = {}
    for v in versions:
        all_results[v] = await run_eval_for_version(v, use_llm_judge)

    print(f"\n{'='*60}")
    print(f"  版本对比结果")
    print(f"{'='*60}")
    for v, data in all_results.items():
        print(f"  {v}: 平均分 {data['avg_score']:.2f}/5")

    # 按分数排序
    ranked = sorted(all_results.items(), key=lambda x: x[1]["avg_score"], reverse=True)
    print(f"\n  推荐使用: {ranked[0][0]} (得分最高)")


def main():
    parser = argparse.ArgumentParser(description="智能客服 Prompt 评测工具")
    parser.add_argument("--version", "-v", type=str, help="评测指定 prompt 版本 (如 v1)")
    parser.add_argument("--compare", "-c", nargs="+", help="对比多个版本 (如 --compare v1 v2)")
    parser.add_argument("--llm-judge", action="store_true", help="启用 LLM 裁判评分（更准确，但消耗 API）")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有可用版本")
    args = parser.parse_args()

    if args.list:
        print("可用的 Prompt 版本:")
        for v in list_versions():
            print(f"  {v['version']}: {v['description']} ({v['file']})")
        return

    if args.compare:
        asyncio.run(compare_versions(args.compare, args.llm_judge))
    elif args.version:
        asyncio.run(run_eval_for_version(args.version, args.llm_judge))
    else:
        # 默认评测最新版本
        files = sorted((os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/prompts").replace("\\", "/").split("/"))
        # 简单获取最新版本名
        versions = list_versions()
        if versions:
            latest = versions[-1]["version"]
            asyncio.run(run_eval_for_version(latest, args.llm_judge))
        else:
            print("没有找到任何 prompt 版本文件")


if __name__ == "__main__":
    main()
