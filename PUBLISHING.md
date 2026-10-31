# 发布到 GitHub

建议仓库名：`fastwam-efficient-routing`

建议 Description：`Shallow-branch and dynamic-routing experiments for FastWAM: LIBERO results, core routing components, and reproducible analysis.`

建议 topics：`robotics`, `embodied-ai`, `inference-optimization`, `pytorch`, `libero`, `dynamic-routing`。

本目录是经过单独整理的发布候选。请只发布本目录，不要对原实验工作目录执行 `git add .`。发布前按实际负责范围核对 README 中的贡献描述，并决定作者信息和许可证。

## 浏览器创建仓库，再推送

1. 在 https://github.com/new 创建空仓库 `fastwam-efficient-routing`。用于公开简历展示时可选择 Public；不要勾选生成 README、许可证或 .gitignore，以免与本地初始内容冲突。
2. 在本目录完成首次提交并推送。将 `YOUR_GITHUB_USERNAME` 换成实际用户名。

```bash
git add .
git commit -m "Add FastWAM routing study and reproducible result analysis"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/fastwam-efficient-routing.git
git push -u origin main
```

如提示作者身份未配置，请使用自己的 Git name/email。认证使用 GitHub 支持的登录流程；不要把访问令牌写进代码或远端 URL。

若本目录尚未初始化 Git，可先执行 `git init -b main`。若已存在 origin，应检查它是否为预期仓库，再设置远端，避免重复添加。

## GitHub CLI 方式

已安装并登录 GitHub CLI 且已有本地提交时，可使用：

```bash
gh repo create fastwam-efficient-routing --public --source=. --remote=origin --push
```

参考：[GitHub 官方上传本地仓库指南](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 简历链接

成功发布后可写：

`具身策略模型推理加速与动态计算优化 | GitHub: github.com/<用户名>/fastwam-efficient-routing`

README 中的固定结构和动态路由指标来自不同实验轮次，保持分别表述。源码组件、聚合复核与完整闭环复现的范围也应保持明确。
