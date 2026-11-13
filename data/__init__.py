# 让 data 成为普通包。否则环境里若装有同名的 data 包，会盖过这个目录，
# 导致 `from data.prepare import ...` 找不到模块。
