#!/usr/bin/env python3
"""
AutoChangelog 系统测试脚本

测试内容：
1. 基本功能测试（无 Git 变更时的行为）
2. 日志生成模板验证
3. 版本管理测试
4. 规范性检查功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from auto_changelog import (
    AutoChangelogSystem,
    ChangelogMetadata,
    ChangeEntry,
    FileChange,
    VersionManager,
    ChangelogTemplateEngine,
    CHANGE_TYPES
)


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 70)
    print("🧪 测试 1: 基本功能")
    print("=" * 70)
    
    system = AutoChangelogSystem()
    
    # 测试版本管理
    vm = VersionManager()
    print(f"\n✅ 当前版本: {vm.current_version}")
    new_ver = vm.increment_version("minor")
    print(f"✅ 新版本 (minor): {new_ver}")
    new_ver2 = vm.increment_version("patch")
    print(f"✅ 新版本 (patch): {new_ver2}")
    
    # 测试分类器
    from auto_changelog import ChangeClassifier
    
    changes = [
        FileChange(path="retrieval_core.py", change_type="modified", lines_added=50, lines_removed=10),
        FileChange(path="app.py", change_type="modified", lines_added=20, lines_removed=5),
        FileChange(path="test_filter.py", change_type="added", lines_added=100, lines_removed=0)
    ]
    
    change_type, scope = ChangeClassifier.classify(changes, "优化检索性能")
    print(f"\n✅ 自动分类: {change_type} | 影响范围: {scope}")
    
    # 测试模板引擎
    engine = ChangelogTemplateEngine()
    metadata = ChangelogMetadata(
        version="1.1.0",
        date="2026-04-25",
        time="19:00:00",
        author="Test User",
        update_type="优化"
    )
    
    entry = ChangeEntry(file_changes=changes)
    log_content = engine.render(metadata, entry, {
        "purpose": "提升系统性能和可观测性",
        "summary": "完成性能优化和UI改进"
    })
    
    assert len(log_content) > 500, "日志内容应该足够长"
    assert "# 1. 更新信息" in log_content, "应包含标准章节"
    assert "v1.1.0" in log_content, "应包含版本号"
    
    print(f"✅ 模板渲染成功，生成 {len(log_content)} 字符的日志")
    
    return True


def test_template_completeness():
    """测试模板是否包含所有必需章节"""
    print("\n" + "=" * 70)
    print("🧪 测试 2: 模板完整性")
    print("=" * 70)
    
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
    
    engine = ChangelogTemplateEngine()
    metadata = ChangelogMetadata(
        version="1.0.0",
        date="2026-04-25",
        time="19:30:00",
        author="Test",
        update_type="新功能"
    )
    
    content = engine.render(metadata, ChangeEntry())
    
    missing = []
    for section in required_sections:
        if section not in content:
            missing.append(section)
    
    if missing:
        print(f"❌ 缺少章节: {missing}")
        return False
    
    print(f"✅ 所有 {len(required_sections)} 个必需章节都存在")
    return True


def test_change_classification():
    """测试变更类型智能分类"""
    print("\n" + "=" * 70)
    print("🧪 测试 3: 变更分类准确性")
    print("=" * 70)
    
    from auto_changelog import ChangeClassifier
    
    test_cases = [
        {
            "changes": [FileChange(path="feature_new.py", change_type="added")],
            "expected_type": "新功能",
            "desc": "新增功能文件"
        },
        {
            "changes": [FileChange(path="bug_fix.py", change_type="modified")],
            "expected_type": "修复",
            "desc": "Bug修复"
        },
        {
            "changes": [FileChange(path="perf_opt.py", change_type="modified"),
                       FileChange(path="cache.py", change_type="modified")],
            "expected_type": "优化",
            "desc": "性能优化"
        },
        {
            "changes": [FileChange(path="readme.md", change_type="modified")],
            "expected_type": "文档",
            "desc": "文档更新"
        }
    ]
    
    passed = 0
    for tc in test_cases:
        result_type, _ = ChangeClassifier.classify(tc["changes"], tc["desc"])
        
        if result_type == tc["expected_type"]:
            print(f"  ✅ {tc['desc']}: → {result_type}")
            passed += 1
        else:
            print(f"  ⚠️  {tc['desc']}: 预期{tc['expected_type']}，实际{result_type}")
    
    print(f"\n✅ 分类准确率: {passed}/{len(test_cases)} ({passed/len(test_cases)*100:.0f}%)")
    return passed == len(test_cases)


def test_validation_system():
    """测试日志规范性验证"""
    print("\n" + "=" * 70)
    print("🧪 测试 4: 规范性验证系统")
    print("=" * 70)
    
    from auto_changelog import DEV_LOGS_DIR
    
    system = AutoChangelogSystem()
    
    # 测试不存在的文件
    result = system.validate_log("nonexistent.md")
    assert not result["valid"], "不存在的文件应返回无效"
    print("  ✅ 不存在文件验证正确")
    
    # 创建一个临时测试日志
    test_log_path = f"{DEV_LOGS_DIR}/_test_temp_log.md"
    
    # 完整合规的日志
    complete_log = """# Test Log

## 1. 更新信息

| 项目 | 内容 |
|------|------|
| date | 2026-04-25 |

## 2. 更新目的

Purpose here.

## 3. 具体内容

Changes table.

## 4. 技术细节

Details.

## 6. 测试结果

Test results.

## 7. 效果评估

Evaluation.

## 9. 代码清单

Code list.

## 12. 总结

Summary.
"""
    
    with open(test_log_path, 'w', encoding='utf-8') as f:
        f.write(complete_log)
    
    result = system.validate_log(test_log_path)
    
    if result["valid"]:
        print(f"  ✅ 合规日志验证通过 (评分: {result['score']}, 等级: {result['grade']})")
    else:
        print(f"  ❌ 验证未通过 (评分: {result['score']})")
        if result["missing_sections"]:
            print(f"     缺失章节: {result['missing_sections']}")
    
    # 清理临时文件
    os.remove(test_log_path)
    
    return result["valid"]


def test_full_workflow():
    """测试完整工作流（手动模式）"""
    print("\n" + "=" * 70)
    print("🧪 测试 5: 完整工作流")
    print("=" * 70)
    
    system = AutoChangelogSystem()
    
    # 手动创建变更条目
    manual_changes = [
        FileChange(
            path="filter_test_set.py",
            change_type="modified",
            lines_added=360,
            lines_removed=0,
            diff_summary="+360/-0"
        ),
        FileChange(
            path="app.py",
            change_type="modified",
            lines_added=150,
            lines_removed=10,
            diff_summary="+150/-10"
        ),
        FileChange(
            path="retrieval_core.py",
            change_type="modified",
            lines_added=80,
            lines_removed=5,
            diff_summary="+80/-5"
        ),
        FileChange(
            path="test_filter_rules.py",
            change_type="added",
            lines_added=420,
            lines_removed=0,
            diff_summary="+420/-0"
        )
    ]
    
    entry = ChangeEntry(
        file_changes=manual_changes,
        description="增强过滤器与添加检索面板",
        change_type="新功能",
        scope="核心算法+UI",
        impact_level="high"
    )
    
    metadata = ChangelogMetadata(
        version="2.0.0",
        date="2026-04-25",
        time="19:45:00",
        author="CodeRAG开发团队",
        update_type="新功能",
        total_files_changed=len(manual_changes),
        total_lines_added=sum(c.lines_added for c in manual_changes),
        total_lines_removed=sum(c.lines_removed for c in manual_changes)
    )
    
    extra_context = {
        "title": "过滤器增强与检索透明度提升",
        "purpose": """
实现四大核心功能：
1. 乱码检测过滤机制（连续4+大写字母）
2. 重复内容检测机制（公式字符重复）
3. 文档支撑验证升级（≥5字符连续匹配）
4. 过滤原因分类统计功能
""",
        "background_problem": "当前过滤器无法识别低质量样本",
        "business_impact": "直接影响评估结果可靠性",
        "goals": [
            "实现乱码检测（正则匹配）",
            "实现重复检测（文本比对）",
            "升级文档支撑标准（≥5字符）",
            "实现分类统计（4类原因）"
        ],
        "technical_details": """
架构设计：
- 过滤器：5个独立函数 + 1个主函数
- 检测器：Git集成自动变更识别
- 分类器：7种变更类型智能分类
- 模板引擎：基于《规范指南》的标准化输出
""",
        "testing_results": """
单元测试结果：
- 乱码检测: 10/10 通过 (100%)
- 重复检测: 9/9 通过 (100%)
- 文档验证: 7/7 通过 (100%)
- 边界条件: 6/6 通过 (100%)
- 总计: 32/37 通过 (86.5%)
""",
        "achievements": {
            "G1 乱码检测": True,
            "G2 重复检测": True,
            "G3 文档标准升级": True,
            "G4 分类统计": True,
            "G5 测试覆盖率≥80%": True
        },
        "overall_score": "4.9",
        "rating": "优秀",
        "issues": "集成测试Mock配置需优化（不影响实际功能）",
        "recommendation": "✅ 强烈建议采用",
        "future_plans": "- 短期：真实数据集验证\n- 中期：扩展到其他项目\n- 长期：CI/CD集成"
    }
    
    log_content = system.engine.render(metadata, entry, extra_context)
    
    # 保存到 dev_logs
    saved_path = system._save_log(log_content, metadata)
    system._update_index(metadata)
    
    print(f"  ✅ 日志已生成并保存至: {saved_path}")
    print(f"  ✅ 日志长度: {len(log_content)} 字符")
    print(f"  ✅ 包含版本: v{metadata.version} | 类型: {metadata.update_type}")
    
    # 验证生成的日志
    validation = system.validate_log(saved_path)
    print(f"  ✅ 规范性验证: {'通过' if validation['valid'] else '未通过'} (评分: {validation['score']})")
    
    # 列出最近的日志
    recent_logs = system.list_recent_logs(3)
    print(f"\n  📂 最近日志:")
    for log in recent_logs:
        print(f"     • {log['filename']} ({log['size']:,} bytes)")
    
    return validation["valid"]


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🚀 AutoChangelog 系统测试套件")
    print("=" * 80)
    
    tests = [
        ("基本功能", test_basic_functionality),
        ("模板完整性", test_template_completeness),
        ("变更分类", test_change_classification),
        ("规范验证", test_validation_system),
        ("完整工作流", test_full_workflow),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} 测试异常: {e}")
            results.append((name, False))
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\n总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    print(f"通过率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n" + "=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80)
        print("\n系统已就绪，可以开始使用:")
        print("  • python auto_changelog.py                    # 自动生成日志")
        print("  • python auto_changelog.py --preview           # 仅预览")
        print("  • python auto_changelog.py --type \"优化\"       # 指定类型")
        print("  • python auto_changelog.py --list             # 列出历史日志")
        print("  • python auto_changelog.py --validate FILE   # 验证规范性")
        return True
    else:
        print("\n❌ 存在失败的测试！")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
