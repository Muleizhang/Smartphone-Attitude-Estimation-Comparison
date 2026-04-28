# 报告编译说明

Overleaf 中请把整个 `report/` 目录内容上传，并以 `main.tex` 为主文件编译。

建议使用 XeLaTeX，因为报告使用 `ctexart` 编写中文正文。图像文件在 `figures/`，自动生成的表格在 `tables/`，整理后的统计结果在 `processed/`。

本地重新生成结果：

```bash
conda run -n phyphox python src/analyze.py
```
