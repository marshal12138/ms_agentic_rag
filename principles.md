## Runtime Permissions

- 启动服务、访问 GPU、运行 GPU 压测或访问本机 HTTP 服务时，必须确认当前命令是否在沙盒中执行。沙盒中即使 `nvidia-smi` 能看到 GPU，PyTorch 仍可能报 `No CUDA GPUs are available`，本机 socket 访问也可能出现 `Operation not permitted`。这类任务需要使用非沙盒/提权执行，并把日志、PID、压测结果写入项目内指定目录。
