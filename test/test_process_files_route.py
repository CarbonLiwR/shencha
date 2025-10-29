import os
import json
import time
from typing import List, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# 可通过环境变量覆盖默认地址
BASE_URL = os.getenv('PROCESS_FILES_URL', 'http://61.171.100.110:8031/api/v1/process_files')


def build_test_files() -> List[Tuple[str, str, str]]:
    """从 test/文档示例 目录构造文件信息列表"""
    base_dir = os.path.dirname(__file__)
    samples_dir = os.path.join(base_dir, "文档示例")

    # 可根据需要增减测试样例文件
    candidates = [
        "1.张长林-论文-共通讯-American J Hematol - 2024 - Jiang - Report of IRF2BP1 as a novel partner of RARA in variant acute promyelocytic leukemia.pdf",
        "尼日利亚发明专利证书.pdf",
        "一种基于改进近似消息传递的心电信号重构方法.pdf",
        "case1.pdf"
    ]

    files_info = []
    for name in candidates:
        path = os.path.join(samples_dir, name)
        if not os.path.exists(path):
            print(f"缺少测试文件: {path}")
            continue
        files_info.append(("files", name, "application/pdf"))

    return files_info


def create_files_payload(files_info):
    """根据文件信息创建用于requests的文件payload"""
    base_dir = os.path.dirname(__file__)
    samples_dir = os.path.join(base_dir, "文档示例")

    files = []
    opened_files = []

    for field_name, filename, content_type in files_info:
        file_path = os.path.join(samples_dir, filename)
        if not os.path.exists(file_path):
            continue

        fo = open(file_path, "rb")
        opened_files.append(fo)
        files.append((field_name, (filename, fo, content_type)))

    return files, opened_files


def send_request(request_id, files_info):
    """发送单个请求"""
    files, opened_files = create_files_payload(files_info)

    try:
        if not files:
            return {
                "request_id": request_id,
                "success": False,
                "status_code": None,
                "response_time": None,
                "error": "No files to upload"
            }

        start_time = time.time()
        resp = requests.post(BASE_URL, files=files, timeout=3600)
        end_time = time.time()

        response_time = end_time - start_time

        # 尝试解析 JSON 响应
        try:
            data = resp.json()
            success = resp.status_code < 400
        except ValueError:
            data = {"error": "Invalid JSON response", "text": resp.text[:500]}
            success = False

        return {
            "request_id": request_id,
            "success": success,
            "status_code": resp.status_code,
            "response_time": response_time,
            "data": data
        }

    except Exception as e:
        return {
            "request_id": request_id,
            "success": False,
            "status_code": None,
            "response_time": None,
            "error": str(e)
        }
    finally:
        # 关闭文件句柄
        for fo in opened_files:
            try:
                fo.close()
            except Exception:
                pass


def main():
    # 获取文件信息
    files_info = build_test_files()

    if not files_info:
        print("未找到任何可用的测试文件，请检查 test/文档示例 目录。")
        return

    # 配置压测参数
    concurrent_requests = 1  # 并发请求数量
    total_requests = 5  # 总请求数量

    print(f"开始压测: {concurrent_requests} 并发，总共 {total_requests} 次请求")
    print(f"目标地址: {BASE_URL}")
    print(f"每次上传 {len(files_info)} 个文件")
    print("-" * 50)

    results = []
    start_time = time.time()

    # 使用线程池进行并发请求
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        # 提交所有任务
        future_to_id = {
            executor.submit(send_request, i, files_info): i
            for i in range(total_requests)
        }

        completed = 0
        for future in as_completed(future_to_id):
            request_id = future_to_id[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                status = "成功" if result["success"] else "失败"
                rt = f"{result['response_time']:.2f}s" if result["response_time"] else "N/A"
                status_code = result.get('status_code', 'N/A')
                print(f"请求 {request_id:3d} [{status}] 状态码: {status_code} 响应时间: {rt}")

                # 如果失败，显示错误信息
                if not result["success"]:
                    error_msg = result.get('error', 'Unknown error')
                    print(f"     错误: {error_msg}")

            except Exception as exc:
                print(f"请求 {request_id} 产生异常: {exc}")
                results.append({
                    "request_id": request_id,
                    "success": False,
                    "error": str(exc)
                })

    total_time = time.time() - start_time

    # 统计结果
    successful_requests = [r for r in results if r.get("success")]
    failed_requests = [r for r in results if not r.get("success")]

    response_times = [r["response_time"] for r in successful_requests if r.get("response_time") is not None]

    print("\n" + "=" * 50)
    print("压测结果统计:")
    print(f"总请求数: {total_requests}")
    print(f"成功请求: {len(successful_requests)}")
    print(f"失败请求: {len(failed_requests)}")
    print(f"总耗时: {total_time:.2f} 秒")

    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        print(f"平均响应时间: {avg_response_time:.2f} 秒")
        print(f"最大响应时间: {max_response_time:.2f} 秒")
        print(f"最小响应时间: {min_response_time:.2f} 秒")
    else:
        print("平均响应时间: N/A")
        print("最大响应时间: N/A")
        print("最小响应时间: N/A")

    if total_time > 0:
        print(f"吞吐量: {total_requests / total_time:.2f} 请求/秒")
    else:
        print("吞吐量: N/A")

    # 保存详细结果
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "压测结果.json")

    # 准备统计信息
    summary = {
        "total_requests": total_requests,
        "successful_requests": len(successful_requests),
        "failed_requests": len(failed_requests),
        "total_time": total_time,
    }

    if response_times:
        summary.update({
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "min_response_time": min_response_time,
            "throughput": total_requests / total_time
        })
    else:
        summary.update({
            "avg_response_time": 0,
            "max_response_time": 0,
            "min_response_time": 0,
            "throughput": 0
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "detailed_results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {out_path}")

    # 显示失败原因统计
    if failed_requests:
        print("\n失败原因统计:")
        error_counts = {}
        for req in failed_requests:
            error = req.get('error', 'Unknown error')
            error_counts[error] = error_counts.get(error, 0) + 1

        for error, count in error_counts.items():
            print(f"  {error}: {count} 次")


if __name__ == "__main__":
    main()