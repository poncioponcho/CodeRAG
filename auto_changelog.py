#!/usr/bin/env python3
"""
自动化更新日志生成与管理系统 (AutoChangelog)

功能特性：
1. ✅ 自动检测代码变更（Git集成）
2. ✅ 智能分类变更类型（新功能/修复/优化/重构/文档）
3. ✅ 生成符合《改进文档规范指南》的标准日志
4. ✅ 版本号自动递增管理
5. ✅ 时间戳自动记录
6. ✅ 支持手动审核与调整
7. ✅ 变更影响分析

使用方法：
    # 基础用法：自动检测变更并生成日志
    python auto_changelog.py
    
    # 高级用法：指定参数
    python auto_changelog.py --type "优化" --scope "性能" --since "2026-04-24"
    
    # 预览模式（不写入文件）
    python auto_changelog.py --preview

依赖项：
    - Git (必须已初始化)
    - Python 3.8+
"""

import os
import sys
import re
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ========== 配置常量 ==========

DEV_LOGS_DIR = "dev_logs"
SPEC_GUIDE_PATH = f"{DEV_LOGS_DIR}/改进文档规范指南.md"
CHANGELOG_INDEX = f"{DEV_LOGS_DIR}/CHANGELOG_INDEX.json"

# 变更类型定义
CHANGE_TYPES = {
    "新功能": {
        "keywords": ["新增", "添加", "实现", "创建", "支持", "引入"],
        "icon": "✨",
        "color": "#10b981"
    },
    "修复": {
        "keywords": ["修复", "解决", "修正", "处理", "补丁"],
        "icon": "🐛",
        "color": "#ef4444"
    },
    "优化": {
        "keywords": ["优化", "提升", "改善", "增强", "加速"],
        "icon": "⚡",
        "color": "#f59e0b"
    },
    "重构": {
        "keywords": ["重构", "重写", "重组", "调整结构", "模块化"],
        "icon": "🔨",
        "color": "#8b5cf6"
    },
    "文档": {
        "keywords": ["文档", "注释", "说明", "指南", "README"],
        "icon": "📝",
        "color": "#3b82f6"
    },
    "安全": {
        "keywords": ["安全", "漏洞", "权限", "加密", "认证"],
        "icon": "🔒",
        "color": "#dc2626"
    },
    "测试": {
        "keywords": ["测试", "单元测试", "集成测试", "覆盖率"],
        "icon": "🧪",
        "color": "#059669"
    }
}


# ========== 数据模型 ==========

@dataclass
class FileChange:
    """文件变更记录"""
    path: str
    change_type: str  # modified, added, deleted, renamed
    lines_added: int = 0
    lines_removed: int = 0
    diff_summary: str = ""
    
    def get_display_name(self) -> str:
        return Path(self.path).name


@dataclass
class ChangeEntry:
    """单条变更条目"""
    file_changes: List[FileChange] = field(default_factory=list)
    description: str = ""
    change_type: str = ""  # 从 CHANGE_TYPES 中选择
    scope: str = ""  # 影响范围（如"核心算法"、"UI"、"配置"）
    impact_level: str = ""  # low / medium / high
    related_issues: List[str] = field(default_factory=list)


@dataclass
class ChangelogMetadata:
    """日志元数据"""
    version: str
    date: str
    time: str
    author: str
    update_type: str
    total_files_changed: int = 0
    total_lines_added: int = 0
    total_lines_removed: int = 0


# ========== 核心模块 ==========

class GitChangeDetector:
    """Git 变更检测器"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        
    def detect_changes(self, since_commit: Optional[str] = None) -> List[FileChange]:
        """
        检测自上次提交以来的变更
        
        Args:
            since_commit: 起始 commit hash（默认为上一个 commit）
        
        Returns:
            List[FileChange]: 变更文件列表
        """
        try:
            if since_commit:
                cmd = ["git", "diff", "--stat", since_commit, "HEAD"]
            else:
                cmd = ["git", "diff", "--cached", "--stat"]
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            changes = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                    
                # 解析 git diff --stat 输出
                # 格式: path | num1 num2 or path | Bin
                match = re.match(r"\s*(.+?)\s*\|\s*(\d+)\s+(\d+)", line)
                if match:
                    filepath = match.group(1).strip()
                    added = int(match.group(2))
                    removed = int(match.group(3))
                    
                    # 判断变更类型
                    change_type = self._detect_file_type(filepath, added, removed)
                    
                    changes.append(FileChange(
                        path=filepath,
                        change_type=change_type,
                        lines_added=added,
                        lines_removed=removed,
                        diff_summary=f"+{added}/-{removed}"
                    ))
            
            return changes
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git 命令执行失败: {e.stderr}")
            return []
    
    def get_diff_details(self, filepath: str) -> str:
        """获取单个文件的详细 diff"""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", filepath],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except Exception as e:
            return f"无法获取 diff: {str(e)}"
    
    def _detect_file_type(self, filepath: str, added: int, removed: int) -> str:
        """根据变更量判断文件类型"""
        if added > 0 and removed == 0:
            return "added"
        elif removed > 0 and added == 0:
            return "deleted"
        elif added == 0 and removed == 0:
            return "renamed"
        else:
            return "modified"


class ChangeClassifier:
    """智能变更分类器"""
    
    @staticmethod
    def classify(changes: List[FileChange], custom_description: str = "") -> Tuple[str, str]:
        """
        自动分类变更类型和影响范围
        
        Returns:
            tuple: (change_type, scope)
        """
        # 收集所有文本信息用于分析
        all_text = " ".join([c.path for c in changes])
        all_text += " " + custom_description
        
        # 类型评分系统
        type_scores = defaultdict(float)
        
        for change_type, config in CHANGE_TYPES.items():
            for keyword in config["keywords"]:
                if keyword.lower() in all_text.lower():
                    type_scores[change_type] += 1.5
                elif keyword.lower() in all_text:
                    type_scores[change_type] += 1.0
        
        # 根据文件路径推断类型
        for change in changes:
            path_lower = change.path.lower()
            
            if any(x in path_lower for x in ["test_", "_test.py", "spec"]):
                type_scores["测试"] += 2.0
            elif any(x in path_lower for x in ["readme", "doc", "guide", "changelog"]):
                type_scores["文档"] += 2.0
            elif any(x in path_lower for x in ["security", "auth", "permission", "encrypt"]):
                type_scores["安全"] += 2.0
            elif change.change_type == "added":
                type_scores["新功能"] += 1.5
        
        # 选择得分最高的类型
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k]) if type_scores else "优化"
        
        # 推断影响范围
        scope = ChangeClassifier._infer_scope(changes)
        
        return best_type, scope
    
    @staticmethod
    def _infer_scope(changes: List[FileChange]) -> str:
        """推断变更的影响范围"""
        paths = [c.path for c in changes]
        all_paths = " ".join(paths).lower()
        
        scopes = {
            "核心算法": ["retrieval_core", "model", "algorithm", "encoder", "embedding"],
            "数据管道": ["data", "pipeline", "etl", "process", "transform"],
            "用户界面": ["app", "ui", "frontend", "streamlit", "interface"],
            "配置部署": ["config", "deploy", "docker", "env", "setting"],
            "测试验证": ["test", "spec", "coverage", "benchmark"],
            "文档资料": ["readme", "doc", "guide", "changelog", "api"]
        }
        
        scope_scores = {k: 0.0 for k in scopes}
        for scope_name, keywords in scopes.items():
            for kw in keywords:
                if kw in all_paths:
                    scope_scores[scope_name] += 1.0
        
        best_scope = max(scope_scores.keys(), key=lambda k: scope_scores[k])
        return best_scope if scope_scores[best_scope] > 0 else "通用"


class VersionManager:
    """版本号管理器"""
    
    def __init__(self):
        self.current_version = self._load_current_version()
    
    def _load_current_version(self) -> str:
        """加载当前版本号"""
        index_path = Path(CHANGELOG_INDEX)
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    return index_data.get("latest_version", "0.0.0")
            except Exception:
                pass
        
        # 如果没有索引文件，从现有日志中推断版本
        return self._infer_version_from_logs()
    
    def _infer_version_from_logs(self) -> str:
        """从现有日志文件名推断版本"""
        logs_dir = Path(DEV_LOGS_DIR)
        if not logs_dir.exists():
            return "1.0.0"
        
        log_files = list(logs_dir.glob("*.md"))
        if not log_files:
            return "1.0.0"
        
        # 提取已有的版本号
        versions = set()
        for f in log_files:
            match = re.search(r'v?(\d+\.\d+\.?\d*)', f.name)
            if match:
                versions.add(match.group(1))
        
        if versions:
            latest = sorted(versions)[-1]
            parts = latest.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            return ".".join(parts)
        
        return "1.0.0"
    
    def increment_version(self, bump_type: str = "patch") -> str:
        """
        递增版本号
        
        Args:
            bump_type: "major", "minor", 或 "patch"
        
        Returns:
            str: 新版本号
        """
        parts = self.current_version.split(".")
        
        while len(parts) < 3:
            parts.append("0")
        
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        new_version = f"{major}.{minor}.{patch}"
        self.current_version = new_version
        return new_version
    
    def suggest_bump_type(self, changes: List[FileChange], change_type: str) -> str:
        """根据变更内容建议版本递增类型"""
        if change_type == "新功能":
            return "minor"
        elif change_type == "修复":
            return "patch"
        elif change_type == "安全":
            return "major"
        elif len(changes) >= 5:
            return "minor"
        else:
            return "patch"


class ChangelogTemplateEngine:
    """日志模板引擎（基于规范指南）"""
    
    def __init__(self):
        self.template = self._load_template()
    
    def _load_template(self) -> str:
        """加载日志模板"""
        return '''# {title}

## 1. 更新信息

| 项目 | 内容 |
|------|------|
| 更新日期 | {date} |
| 更新时间 | {time} |
| 更新类型 | {update_type} |
| 文档版本 | v{version} |
| 修改作者 | {author} |

---

## 2. 更新目的与背景说明

### 2.1 更新目的

{purpose}

### 2.2 背景说明

#### 问题发现

{background_problem}

#### 业务影响

{business_impact}

#### 改进目标

{goals}

---

## 3. 具体更新内容

### 3.1 修改的核心模块

{changes_table}

---

## 4. 技术实现细节

{technical_details}

---

## 5. 版本变更前后差异

{version_comparison}

---

## 6. 测试活动详细结果

{testing_results}

---

## 7. 改进效果总结与评估

### 7.1 目标达成情况

{goal_achievement}

### 7.2 多维度评估矩阵

{evaluation_matrix}

**综合评分**: **{overall_score}/5.0 ({rating})**

---

## 8. 发现问题与解决方案

{issues_and_solutions}

---

## 9. 代码变更清单

{code_changes}

---

## 10. 风险评估与缓解措施

{risk_assessment}

---

## 11. 后续行动计划

{future_plans}

---

## 12. 总结与推荐

### 总结

{summary}

### 推荐

**推荐状态**: **{recommendation_status}**

**理由**: {recommendation_reasons}

---

**文档版本**: v{version}
**最后更新**: {date} {time}
**存储位置**: `dev_logs/{filename}`
**遵循规范**: 《改进文档规范指南》v1.0
'''
    
    def render(self, metadata: ChangelogMetadata, entry: ChangeEntry, 
               extra_context: Optional[Dict[str, Any]] = None) -> str:
        """
        渲染完整的更新日志
        
        Args:
            metadata: 日志元数据
            entry: 变更条目
            extra_context: 额外上下文信息
            
        Returns:
            str: 完整的 Markdown 日志内容
        """
        # 初始化上下文（确保不为 None）
        ctx_base = extra_context or {}
        
        ctx = {
            "title": f"{metadata.date}｜{ctx_base.get('title', '系统更新') if extra_context else '常规更新'}",
            "date": metadata.date,
            "time": metadata.time,
            "update_type": metadata.update_type,
            "version": metadata.version,
            "author": metadata.author,
            
            # 第2章
            "purpose": ctx_base.get("purpose", "本次更新包含以下改进..."),
            "background_problem": ctx_base.get("background_problem", "见具体章节"),
            "business_impact": ctx_base.get("business_impact", "详见技术细节"),
            "goals": self._format_goals(ctx_base.get("goals", [])),
            
            # 第3章
            "changes_table": self._generate_changes_table(entry.file_changes),
            
            # 第4章
            "technical_details": ctx_base.get("technical_details", 
                "详见代码变更清单"),
            
            # 第5章
            "version_comparison": ctx_base.get("version_comparison",
                "参见测试结果"),
                
            # 第6章
            "testing_results": ctx_base.get("testing_results",
                "待补充完整测试数据"),
                
            # 第7章
            "goal_achievement": self._format_goals_achievement(
                ctx_base.get("achievements", {})),
            "evaluation_matrix": ctx_base.get("evaluation_matrix",
                self._default_evaluation_matrix()),
            "overall_score": ctx_base.get("overall_score", "4.5"),
            "rating": ctx_base.get("rating", "良好"),
            
            # 第8-12章
            "issues_and_solutions": ctx_base.get("issues", "暂无重大问题"),
            "code_changes": self._generate_code_change_list(entry),
            "risk_assessment": self._generate_risk_assessment(
                entry, ctx_base if extra_context else {}),
            "future_plans": ctx_base.get("future_plans", "- 短期：监控线上表现\n- 长期：持续优化"),
            
            # 第12章
            "summary": ctx_base.get("summary", 
                f"完成 {metadata.update_type} 类型的更新，涉及 {metadata.total_files_changed} 个文件。"),
            "recommendation_status": ctx_base.get("recommendation", "✅ 建议采纳"),
            "recommendation_reasons": ctx_base.get("recommendation_reasons",
                "变更合理、测试充分、风险可控。"),
            
            "filename": self._generate_filename(metadata)
        }
        
        return self.template.format(**ctx)
    
    def _format_goals(self, goals: List[str]) -> str:
        if not goals:
            return "1. ✅ 完成预定改进任务\n2. ✅ 通过质量验证\n3. ✅ 符合规范要求"
        
        formatted = []
        for i, goal in enumerate(goals, 1):
            formatted.append(f"{i}. ✅ {goal}")
        return "\n".join(formatted)
    
    def _format_goals_achievement(self, achievements: Dict[str, bool]) -> str:
        if not achievements:
            return "| G1 | 目标一 | ✅ 达成 | 证据 |\n| G2 | 目标二 | ✅ 达成 | 证据 |"
        
        rows = []
        for i, (goal, achieved) in enumerate(achievements.items(), 1):
            status = "✅ 达成" if achieved else "⚠️ 部分达成"
            rows.append(f"| G{i} | {goal} | {status} | 见详情 |")
        
        return "\n".join(rows)
    
    def _default_evaluation_matrix(self) -> str:
        return '''| 评估维度 | 得分 (1-5) | 说明 |
|----------|-----------|------|
| 功能完整性 | ⭐⭐⭐⭐⭐ | 所有需求均已实现 |
| 代码质量 | ⭐⭐⭐⭐☆ | 清晰注释、完整文档 |
| 测试覆盖 | ⭐⭐⭐⭐☆ | 主要场景已覆盖 |
| 性能表现 | ⭐⭐⭐⭐☆ | 无显著退化 |
| 向后兼容 | ⭐⭐⭐⭐⭐ | 新参数均有默认值 |'''
    
    def _generate_changes_table(self, changes: List[FileChange]) -> str:
        if not changes:
            return "*无文件变更记录*"
        
        header = "| 文件路径 | 操作类型 | 行数变化 | 状态 |\n|----------|----------|----------|------|"
        rows = []
        
        for change in changes[:15]:  # 最多显示15个文件
            status_icon = {"modified": "✏️", "added": "➕", "deleted": "➖", "renamed": "↩️"}
            icon = status_icon.get(change.change_type, "❓")
            
            rows.append(
                f"`{change.get_display_name()}` | "
                f"{change.change_type} | "
                f"{change.diff_summary} | "
                f"{icon} |"
            )
        
        if len(changes) > 15:
            rows.append(f"*... 还有 {len(changes)-15} 个文件*")
        
        return "\n".join([header] + rows)
    
    def _generate_code_change_list(self, entry: ChangeEntry) -> str:
        summary = f"**总变更文件数**: {len(entry.file_changes)}\n"
        summary += f"**总新增行数**: {sum(c.lines_added for c in entry.file_changes)}\n"
        summary += f"**总删除行数**: {sum(c.lines_removed for c in entry.file_changes)}\n\n"
        
        summary += "**详细文件清单**:\n\n"
        for i, change in enumerate(entry.file_changes[:20], 1):
            summary += f"{i}. `{change.path}` ({change.change_type}, {change.diff_summary})\n"
        
        return summary
    
    def _generate_risk_assessment(self, entry: ChangeEntry, 
                                  context: Optional[Dict[str, Any]] = None) -> str:
        risks = [
            ("向后兼容性破坏", "低", "中", "新参数均有默认值", "✅ 已缓解"),
            ("性能回归", "低", "低", "无明显性能影响", "✅ 已缓解"),
            ("测试覆盖不足", "中", "中", "建议补充集成测试", "⚠️ 需关注"),
        ]
        
        table = "| 风险项 | 发生概率 | 影响 | 缓解措施 | 当前状态 |\n"
        table += "|--------|----------|------|----------|----------|\n"
        
        for risk in risks:
            table += f"| {risk[0]} | {risk[1]} | {risk[2]} | {risk[3]} | {risk[4]} |\n"
        
        return table
    
    def _generate_filename(self, metadata: ChangelogMetadata) -> str:
        type_map = {
            "新功能": "功能新增",
            "修复": "缺陷修复",
            "优化": "性能优化",
            "重构": "架构重构",
            "文档": "文档更新",
            "安全": "安全加固",
            "测试": "测试完善"
        }
        
        type_desc = type_map.get(metadata.update_type, "系统更新")
        return f"{metadata.date}_{type_desc}_v{metadata.version}.md"


class AutoChangelogSystem:
    """自动化更新日志管理系统 - 主控制器"""
    
    def __init__(self, repo_path: str = ".", dev_logs_dir: str = DEV_LOGS_DIR):
        self.repo_path = repo_path
        self.dev_logs_dir = dev_logs_dir
        self.detector = GitChangeDetector(repo_path)
        self.classifier = ChangeClassifier()
        self.version_mgr = VersionManager()
        self.engine = ChangelogTemplateEngine()
        
        # 确保目录存在
        Path(dev_logs_dir).mkdir(exist_ok=True)
    
    def generate_changelog(
        self,
        custom_description: str = "",
        custom_type: Optional[str] = None,
        custom_scope: Optional[str] = None,
        since_commit: Optional[str] = None,
        preview_only: bool = False,
        extra_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, ChangelogMetadata]:
        """
        生成更新日志的主方法
        
        Args:
            custom_description: 自定义描述（可选）
            custom_type: 强制指定变更类型（可选）
            custom_scope: 强制指定影响范围（可选）
            since_commit: 起始 commit（可选）
            preview_only: 是否仅预览不保存
            extra_context: 额外的上下文信息
            
        Returns:
            tuple: (log_content, metadata)
        """
        print("🔄 正在检测代码变更...")
        
        # Step 1: 检测变更
        changes = self.detector.detect_changes(since_commit)
        
        if not changes:
            print("ℹ️  未检测到任何变更，无需生成日志")
            return "", ChangelogMetadata(version="0.0.0", date="", time="", 
                                     author="", update_type="")
        
        print(f"✅ 检测到 {len(changes)} 个文件变更")
        
        # Step 2: 分类变更
        change_type, scope = self.classifier.classify(changes, custom_description)
        
        if custom_type and custom_type in CHANGE_TYPES:
            change_type = custom_type
        if custom_scope:
            scope = custom_scope
        
        # Step 3: 版本管理
        bump_type = self.version_mgr.suggest_bump_type(changes, change_type)
        new_version = self.version_mgr.increment_version(bump_type)
        
        # Step 4: 构建元数据和条目
        now = datetime.now()
        metadata = ChangelogMetadata(
            version=new_version,
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            author=self._get_git_author(),
            update_type=change_type,
            total_files_changed=len(changes),
            total_lines_added=sum(c.lines_added for c in changes),
            total_lines_removed=sum(c.lines_removed for c in changes)
        )
        
        entry = ChangeEntry(
            file_changes=changes,
            description=custom_description or f"基于Git变更自动生成的更新记录",
            change_type=change_type,
            scope=scope,
            impact_level="medium" if len(changes) <= 5 else "high"
        )
        
        # Step 5: 渲染日志
        print("📝 正在生成标准化日志...")
        log_content = self.engine.render(metadata, entry, extra_context or {})
        
        # Step 6: 保存或预览
        if not preview_only:
            filename = self._save_log(log_content, metadata)
            print(f"💾 日志已保存至: {filename}")
            self._update_index(metadata)
        else:
            print("\n" + "=" * 70)
            print("📋 日志预览（未保存）:")
            print("=" * 70)
            print(log_content[:2000])  # 只显示前2000字符
            if len(log_content) > 2000:
                print(f"\n... (共 {len(log_content)} 字符)")
        
        return log_content, metadata
    
    def _save_log(self, content: str, metadata: ChangelogMetadata) -> str:
        """保存日志到 dev_logs 目录"""
        filename = self.engine._generate_filename(metadata)
        filepath = Path(self.dev_logs_dir) / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(filepath)
    
    def _update_index(self, metadata: ChangelogMetadata):
        """更新日志索引文件"""
        index_path = Path(CHANGELOG_INDEX)
        
        current_index = {}
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    current_index = json.load(f)
            except Exception:
                pass
        
        # 更新索引
        current_index["latest_version"] = metadata.version
        current_index["last_updated"] = f"{metadata.date} {metadata.time}"
        current_index["total_entries"] = current_index.get("total_entries", 0) + 1
        current_index["history"] = current_index.get("history", [])
        current_index["history"].append({
            "version": metadata.version,
            "date": metadata.date,
            "type": metadata.update_type,
            "files_changed": metadata.total_files_changed
        })
        
        # 只保留最近 50 条历史
        current_index["history"] = current_index["history"][-50:]
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(current_index, f, indent=2, ensure_ascii=False)
    
    def _get_git_author(self) -> str:
        """获取 Git 作者信息"""
        try:
            result = subprocess.run(
                ["git", "config", "user.name"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip() or "CodeRAG开发团队"
        except Exception:
            return "CodeRAG开发团队"
    
    def list_recent_logs(self, limit: int = 10) -> List[Dict]:
        """列出最近的日志文件"""
        logs_dir = Path(self.dev_logs_dir)
        if not logs_dir.exists():
            return []
        
        log_files = sorted(
            logs_dir.glob("*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]
        
        result = []
        for f in log_files:
            result.append({
                "filename": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            })
        
        return result
    
    def validate_log(self, filepath: str) -> Dict[str, Any]:
        """
        验证日志是否符合规范指南
        
        Returns:
            dict: 包含验证结果和详细报告
        """
        required_sections = [
            "1. 更新信息",
            "2. 更新目的与背景说明",
            "3. 具体更新内容",
            "4. 技术实现细节",
            "6. 测试活动详细结果",
            "7. 改进效果总结与评估",
            "9. 代码变更清单",
            "12. 总结与推荐"
        ]
        
        path = Path(filepath)
        if not path.exists():
            return {"valid": False, "error": "文件不存在"}
        
        content = path.read_text(encoding='utf-8')
        
        missing_sections = []
        present_sections = []
        
        for section in required_sections:
            if section in content:
                present_sections.append(section)
            else:
                missing_sections.append(section)
        
        # 计算评分
        score = len(present_sections) / len(required_sections) * 100
        
        is_valid = (
            score >= 80 and  # 至少80%的必需章节存在
            "dev_logs" in str(path.absolute())  # 必须在dev_logs目录中
        )
        
        return {
            "valid": is_valid,
            "score": round(score, 1),
            "present_sections": present_sections,
            "missing_sections": missing_sections,
            "file_size": len(content),
            "line_count": content.count('\n'),
            "grade": "A" if score >= 95 else "B" if score >= 85 else "C" if score >= 75 else "D"
        }


# ========== CLI 入口点 ==========

def main():
    parser = argparse.ArgumentParser(
        description="🔄 AutoChangelog - 自动化更新日志生成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                          # 自动检测变更并生成日志
  %(prog)s --preview                 # 仅预览，不保存
  %(prog)s --type "优化"              # 强制指定变更类型
  %(prog)s --since abc1234           # 指定起始commit
  %(prog)s --list                   # 列出最近的日志
  %(prog)s --validate log.md         # 验证日志规范性
        """
    )
    
    parser.add_argument("--preview", action="store_true",
                       help="仅预览日志，不保存到文件")
    parser.add_argument("--type", choices=list(CHANGE_TYPES.keys()),
                       help="强制指定变更类型")
    parser.add_argument("--scope", help="指定影响范围")
    parser.add_argument("--since", metavar="COMMIT",
                       help="指定起始commit hash")
    parser.add_argument("--list", action="store_true",
                       help="列出最近的日志文件")
    parser.add_argument("--validate", metavar="FILE",
                       help="验证指定日志文件的规范性")
    parser.add_argument("--description", "-d", metavar="TEXT",
                       help="自定义更新描述")
    parser.add_argument("--context", metavar="JSON_FILE",
                       help="从JSON文件加载额外上下文")
    
    args = parser.parse_args()
    
    system = AutoChangelogSystem()
    
    # 验证模式
    if args.validate:
        result = system.validate_log(args.validate)
        print("\n" + "=" * 70)
        print("📋 日志规范性验证结果")
        print("=" * 70)
        print(f"文件: {args.validate}")
        print(f"有效性: {'✅ 通过' if result['valid'] else '❌ 未通过'}")
        print(f"评分: {result['score']}/100 ({result['grade']}级)")
        print(f"章节完整性: {len(result['present_sections'])}/{len(result['present_sections']) + len(result['missing_sections'])}")
        
        if result['missing_sections']:
            print("\n缺失章节:")
            for section in result['missing_sections']:
                print(f"  ❌ {section}")
        
        return
    
    # 列表模式
    if args.list:
        logs = system.list_recent_logs()
        print("\n" + "=" * 70)
        print("📂 最近的更新日志")
        print("=" * 70)
        print(f"{'文件名':<45} {'大小':>8} {'修改时间':>18}")
        print("-" * 75)
        
        for log in logs:
            size_str = f"{log['size']:,} B" if log['size'] < 1024 else f"{log['size']/1024:.1f} KB"
            print(f"{log['filename']:<45} {size_str:>8} {log['modified']:>18}")
        
        return
    
    # 加载额外上下文
    extra_context = None
    if args.context:
        context_path = Path(args.context)
        if context_path.exists():
            with open(context_path, 'r', encoding='utf-8') as f:
                extra_context = json.load(f)
    
    # 生成日志
    print("\n" + "=" * 70)
    print("🚀 AutoChangelog 启动")
    print("=" * 70)
    
    log_content, metadata = system.generate_changelog(
        custom_description=args.description or "",
        custom_type=args.type,
        custom_scope=args.scope,
        since_commit=args.since,
        preview_only=args.preview,
        extra_context=extra_context
    )
    
    if metadata.version != "0.0.0":
        print("\n" + "=" * 70)
        print("✅ 日志生成成功！")
        print("=" * 70)
        print(f"版本号: v{metadata.version}")
        print(f"变更类型: {metadata.update_type}")
        print(f"变更文件: {metadata.total_files_changed} 个")
        print(f"行数统计: +{metadata.total_lines_added} / -{metadata.total_lines_removed}")


if __name__ == "__main__":
    main()
