"""CoAgenticRetriever v2 train launcher 的 Python 配置编译模块。

这个包只负责把 launcher 参数、main_run_config、resource、overlay 和外部环境变量
编译成确定的运行文件。真正启动服务、等待 GPU 和执行训练仍由 Bash launcher 完成。
"""
