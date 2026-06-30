# 📋 回家待办备忘录

## 1. 确认 conda 环境

```bash
conda activate torch_env
python -c "import torch; print(torch.cuda.is_available())"
```

如果环境不存在或丢了，重建：

```bash
conda create -n torch_env python=3.10 -y
conda activate torch_env
pip install -r requirements.txt
```

## 2. 运行错误分析脚本

```bash
cd C:\Users\admin3823\Desktop\-
python scripts/error_analysis.py
```

运行后会：
- 控制台打印**各类别准确率**（记下来，填入报告）
- 生成 `results/misclassified_samples.png`

## 3. 填入报告数据

打开 `results/report_summary.md`，在 **3.4 节** 表格中填入准确率数值：

```
| without_mask | 0.718 | 0.786 | 0.752 | 这里填入 |
| with_mask    | 0.797 | 0.849 | 0.827 | 这里填入 |
```

## 4. 提交推送

```bash
git add -A
git commit -m "docs: fill accuracy data & add misclassified samples"
git push
```

---

就这三步，10 分钟搞定。
