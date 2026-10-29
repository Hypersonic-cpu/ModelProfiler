# ModelProfiler

用于常驻 LLM serving 的按需 Nsight Systems 采集。`modelprofiler/nsys/` 只管理 profiler；`scripts/` 负责请求和实验编排。Python ≥ 3.10，无第三方 Python 运行依赖。

**限制：Linux x86_64 的 nsys 不支持对任意已运行、未注入的进程直接 attach 做 CUDA tracing。** 必须在服务首次启动时通过 `nsys launch` 预先注入，之后再连接该 session，反复 start/stop，无需重新加载模型。没有预注入的现有服务无法用这个模块补做 CUDA tracing，也不会被自动重启。参见 [NVIDIA 关于 Linux attach 的说明](https://forums.developer.nvidia.com/t/nsight-system-2019-6-1-does-not-have-option-to-attach-to-pid-on-target/108137) 和 [官方交互式 CLI 文档](https://docs.nvidia.com/nsight-systems/UserGuide/#example-interactive-cli-command-sequences)。

## 安装与首次启动

```bash
pip install -e .
export NSYS_BIN=/path/to/real/nsys  # 可选；默认使用 PATH 中的 nsys
"$NSYS_BIN" --version

# 用你原有的 vLLM + Mooncake 启动命令替换下方示例。
# launch 不采集；服务加载完成后再执行 start。该命令常驻前台。
python -m modelprofiler.nsys --session llm launch -- \
  /root/.venvs/mooncake-qwen3/bin/python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model --served-model-name experiment --port 8000
```

需要后台常驻时，用 `nohup ... >server.log 2>&1 &` 或原有服务管理器包装完整 launch 命令。控制端与 launch 端应使用同一用户、同一个 nsys 安装和 PID namespace，并共享临时目录。`launch` 成功不代表模型已就绪，发请求前由编排端检查服务就绪。

## 按需采集

服务就绪后，在另一个终端：

```bash
python -m modelprofiler.nsys --session llm status
python -m modelprofiler.nsys --session llm start --output artifacts/run1
# 只有出现 READY，才开始发请求。请求完成后：
python -m modelprofiler.nsys --session llm stop

# 或在开始后约 10 秒自动停止；CLI 返回后计时器仍有效。
python -m modelprofiler.nsys --session llm start --output artifacts/run2 --duration 10
```

`stop` 等待 nsys 完成报告生成，**不退出 session、不终止 server**；之后可换一个输出名继续采集。已有报告不会覆盖。同一 session 只允许一次采集；多个实验需由调用端串行编排。

自动停止由独立 Python watchdog 调用 `nsys stop` 实现（已验证的 nsys 2024.6.1 的 `start` 无 `--duration`）。临时目录 `/tmp/modelprofiler-nsys-<uid>/`（或 `$TMPDIR` 下）只保存本次采集标记、文件锁和 watchdog 日志，trace 由 nsys 写入输出路径。提前 stop 会使旧标记失效，旧计时器不会误停后续采集。控制期间不要混用绕过模块的原生 `nsys start/stop`，否则无法保证计时器归属。

## Python API 与请求示例

```python
from modelprofiler.nsys import NsysSession

profiler = NsysSession("llm")
with profiler.capture("artifacts/request1", duration=30) as report:
    # 进入这里时已确认 collecting；自行发送请求并等待完成。
    send_request()
# 正常返回或请求抛异常时，都会停止采集。
```

`duration` 是采集上限，不是请求超时；到时仍未结束的请求只记录前半段。需要完整记录时不设置 duration。

仓库附带一个非流式 OpenAI-compatible completion 请求示例：

```bash
python -m scripts.profile_query --session llm --model experiment \
  --base-url http://127.0.0.1:8000/v1 \
  --prompt 'Say hello.' --max-tokens 8 --output artifacts/query1
```

## 查看 kernel 时间

输出为 `<output>.nsys-rep`，可在 Nsight Systems GUI 打开。默认追踪 `cuda,nvtx,osrt`，关闭 CPU sampling/context-switch tracing，并启用 `--cuda-graph-trace=node`，记录 CUDA Graph 内的单个 kernel（可能增加开销）。trace 覆盖 launch 的进程树，包括正常 spawn/exec 的 worker；独立启动的 Mooncake daemon、其他节点的 worker 需要各自预注入和采集。

```bash
# GPU kernel / memcpy 等事件的开始时间和持续时间
"${NSYS_BIN:-nsys}" stats --quiet --report cuda_gpu_trace --format csv \
  artifacts/query1.nsys-rep > artifacts/gpu_trace.csv

# 关联 CPU launch API 与 GPU kernel 执行时间
"${NSYS_BIN:-nsys}" stats --quiet --report cuda_kern_exec_trace --format csv \
  artifacts/query1.nsys-rep > artifacts/kernel_launch.csv
```

默认时间单位为 ns，结束时间 = Start + Duration。区分 CPU launch API 的开始/结束与 GPU kernel 的实际开始/结束；CUDA Graph 内节点不一定对应独立的 CPU launch API，完整节点记录以 GPU trace 为准。这里只记录采集窗口内事件，不记录服务启动历史；并发请求的 kernel 也会一并采集。

## 验证

```bash
NSYS_INTEGRATION_BIN=/path/to/nsys python -m unittest discover -s tests -v
```

测试只创建自己的 CPU sleep 进程和本地 HTTP stub，验证 READY 屏障、请求脚本、异常清理、定时停止、旧计时器隔离和重复采集，最后只关闭测试自己的 session。

已在 A100 上用 nsys 2024.6.1、vLLM 0.10.2 和 `DeepSeek-R1-Distill-Qwen-1.5B`（Qwen2 架构）实测：32 MiB KV cache、128 token 上下文、eager 模式，显存约 4.1 GiB；一次 4 token 请求记录到 1,308 个 kernel 和 1,521 个 CUDA runtime API，停止采集后服务健康检查仍返回 200。报告和 CSV 位于 `artifacts/qwen-validation.*`、`artifacts/gpu_trace.csv`、`artifacts/kernel_launch.csv`，不纳入 Git。测试后已清理临时服务、释放显存。原有 Qwen3-32B 服务未被操作；它自行退出后才启动测试模型。尚未实测 Mooncake 传输和 CUDA Graph 模式。
