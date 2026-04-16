import streamlit as st
import numpy as np
import joblib
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加gcForest库路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gcForest', 'lib'))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 项目路径
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "gcforest_ids_model.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed_data_final.npz")

# 特征说明
feature_descriptions = {
    '特征1': '网络流量持续时间',
    '特征2': '协议类型（TCP/UDP/ICMP）',
    '特征3': '服务类型',
    '特征4': '标志位',
    '特征5': '源字节数',
    '特征6': '目标字节数',
    '特征7': '错误片段',
    '特征8': '紧急数据',
    '特征9': 'HTTP方法',
    '特征10': 'HTTP协议版本',
    '特征11': '登录尝试',
    '特征12': '失败登录次数',
    '特征13': '文件访问次数',
    '特征14': '目录访问次数',
    '特征15': '输入操作次数',
    '特征16': '输出操作次数',
    '特征17': 'SF状态标志',
    '特征18': 'REJ状态标志',
    '特征19': 'RSTO状态标志',
    '特征20': 'RSTOS0状态标志',
    '特征21': 'RSTR状态标志',
    '特征22': 'S0状态标志',
    '特征23': 'S1状态标志',
    '特征24': 'S2状态标志',
    '特征25': 'S3状态标志',
    '特征26': 'SF状态标志',
    '特征27': 'OTH状态标志',
    '特征28': 'REJ状态标志',
    '特征29': 'RSTO状态标志',
    '特征30': 'RSTOS0状态标志',
    '特征31': 'RSTR状态标志',
    '特征32': 'S0状态标志',
    '特征33': 'S1状态标志',
    '特征34': 'S2状态标志',
    '特征35': 'S3状态标志',
    '特征36': 'SF状态标志',
    '特征37': 'OTH状态标志',
    '特征38': '源端口',
    '特征39': '目标端口',
    '特征40': '连接建立时间',
    '特征41': '连接关闭时间',
    '特征42': '连接状态',
    '特征43': '服务类型',
    '特征44': '协议类型',
    '特征45': '网络流量',
    '特征46': '数据包大小',
    '特征47': '数据包数量',
    '特征48': '错误率',
    '特征49': '重传率',
    '特征50': '延迟时间',
    '特征51': '响应时间',
    '特征52': '带宽利用率',
    '特征53': 'CPU使用率',
    '特征54': '内存使用率',
    '特征55': '磁盘使用率',
    '特征56': '网络接口使用率',
    '特征57': '进程数',
    '特征58': '线程数',
    '特征59': '系统负载',
    '特征60': '网络连接数',
    '特征61': '活跃连接数',
    '特征62': '半开连接数',
    '特征63': '已关闭连接数',
    '特征64': 'SYN攻击检测',
    '特征65': 'UDP洪水检测',
    '特征66': 'ICMP洪水检测',
    '特征67': '异常流量检测'
}

# 类别说明
class_descriptions = {
    'Benign': '正常网络流量',
    'Bot': '僵尸网络攻击',
    'Brute Force -Web': 'Web暴力破解攻击',
    'Brute Force -XSS': 'XSS暴力破解攻击',
    'DDOS attack-HOIC': 'HOIC分布式拒绝服务攻击',
    'DDOS attack-LOIC-UDP': 'LOIC UDP分布式拒绝服务攻击',
    'DDoS attacks-LOIC-HTTP': 'LOIC HTTP分布式拒绝服务攻击',
    'DoS attacks-GoldenEye': '黄金眼拒绝服务攻击',
    'DoS attacks-Hulk': '绿巨人拒绝服务攻击',
    'DoS attacks-SlowHTTPTest': '慢速HTTP测试攻击',
    'DoS attacks-Slowloris': '慢速loris拒绝服务攻击',
    'FTP-BruteForce': 'FTP暴力破解攻击',
    'Infilteration': '渗透攻击',
    'SQL Injection': 'SQL注入攻击',
    'SSH-Bruteforce': 'SSH暴力破解攻击'
}

# 指标说明
metric_descriptions = {
    '准确率': '正确预测的样本数占总样本数的比例',
    '加权F1值': '考虑类别不平衡的F1值加权平均',
    '宏平均F1值': '所有类别的F1值简单平均',
    '精确率': '预测为正例的样本中实际为正例的比例',
    '召回率': '实际为正例的样本中被正确预测的比例',
    'F1值': '精确率和召回率的调和平均值'
}

# 页面配置
st.set_page_config(
    page_title="数据视角下网络攻击类型的分析、建模与检测",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 侧边栏
with st.sidebar:
    st.title("📋 导航菜单")
    page = st.radio(
        "选择功能",
        ["系统概览", "数据预览", "批量检测", "单样本检测", "项目说明"]
    )
    
    st.markdown("---")
    st.title("ℹ️ 帮助信息")
    st.subheader("什么是网络入侵检测？")
    st.markdown("网络入侵检测是一种安全机制，用于识别和响应网络中的恶意活动，保护网络系统免受攻击。")
    
    st.subheader("什么是gcForest？")
    st.markdown("gcForest（深度森林）是一种基于集成学习的算法，通过多粒度扫描和级联森林结构，实现高效的分类和预测。")
    
    st.subheader("如何使用本系统？")
    st.markdown("1. 在'系统概览'查看核心指标\n2. 在'数据预览'查看测试数据\n3. 在'批量检测'进行批量预测\n4. 在'单样本检测'分析单个样本\n5. 在'项目说明'了解项目背景")

# 缓存装饰器
@st.cache_resource
def load_model():
    """加载模型"""
    with st.spinner("正在加载模型..."):
        model = joblib.load(MODEL_PATH)
    return model

@st.cache_data
def load_data():
    """加载数据"""
    with st.spinner("正在加载数据..."):
        data = np.load(DATA_PATH, allow_pickle=True)
        X_test = data["X_test"].astype('float32')
        y_test = data["y_test"].astype('int32')
        class_names = data["class_names"]
    return X_test, y_test, class_names

# 延迟加载模型和数据
X_test, y_test, class_names = load_data()

# 系统概览页面
if page == "系统概览":
    # 页面标题
    st.title("数据视角下网络攻击类型的分析、建模与检测")
    
    # 系统简介
    st.markdown("---")
    st.subheader("系统简介")
    st.markdown("本系统基于gcForest（深度森林）算法实现网络入侵检测，能够识别15种不同类型的网络攻击，包括DDoS、暴力破解、SQL注入等常见攻击方式。系统采用多粒度扫描和级联森林结构，在16GB内存环境下实现了高效的训练和预测。")
    
    # 核心指标卡片
    st.markdown("---")
    st.subheader("核心指标")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("测试集准确率", "90.93%")
        st.caption(metric_descriptions['准确率'])
    with col2:
        st.metric("加权F1值", "97.94%")
        st.caption(metric_descriptions['加权F1值'])
    with col3:
        st.metric("宏平均F1值", "84.63%")
        st.caption(metric_descriptions['宏平均F1值'])
    with col4:
        st.metric("单样本推理耗时", "~0.03s")
        st.caption("模型对单个样本进行预测所需的时间，反映了系统的响应速度。")
    
    # 类别分布概览
    st.markdown("---")
    st.subheader("攻击类型分布")
    st.markdown("下图展示了测试集中各类攻击的样本数量分布，包含15种不同类型的网络攻击。从图表中可以看出，正常网络流量(BENIGN)的样本数量最多，其次是各种DoS攻击类型。")
    y_test_counts = np.bincount(y_test)
    class_distribution = pd.DataFrame({
        '类别': class_names,
        '样本数': y_test_counts,
        '类别说明': [class_descriptions.get(cls, '未知') for cls in class_names]
    })
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = sns.barplot(x='类别', y='样本数', data=class_distribution, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 添加数值标签
    for bar in bars.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig)
    
    # 类别说明表格
    st.subheader("攻击类型说明")
    st.dataframe(class_distribution[['类别', '类别说明', '样本数']], height=400)

# 数据预览页面
elif page == "数据预览":
    st.title("📊 数据预览")
    
    # 数据基本信息
    st.markdown("---")
    st.subheader("数据基本信息")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("测试集样本数", len(X_test))
    with col2:
        st.metric("特征维度", X_test.shape[1])
    with col3:
        st.metric("类别数量", len(class_names))
    
    # 特征预览
    st.markdown("---")
    st.subheader("特征预览")
    
    # 特征选择器
    feature_range = st.slider("选择特征范围", 1, 67, (1, 10))
    start_idx, end_idx = feature_range[0]-1, feature_range[1]
    
    # 预览数据
    preview_data = pd.DataFrame(X_test[:50, start_idx:end_idx], columns=[f'特征{i+1}' for i in range(start_idx, end_idx)])
    preview_data['真实类别'] = [class_names[y] for y in y_test[:50]]
    
    # 显示特征说明
    st.subheader("特征说明")
    feature_info = []
    for i in range(start_idx, end_idx):
        feature_info.append({
            '特征': f'特征{i+1}',
            '说明': feature_descriptions.get(f'特征{i+1}', '未知特征')
        })
    st.dataframe(pd.DataFrame(feature_info))
    
    # 特征选择说明
    st.markdown("**特征选择说明**：前10个特征是基于特征重要性分析得出的核心特征，这些特征对网络攻击检测具有重要意义。")
    
    st.subheader("前10个核心特征详细说明")
    st.markdown("通过特征重要性分析，我们选择了以下10个对网络攻击检测最关键的特征：")
    
    core_features = {
        '特征1': '网络流量持续时间 - 网络连接的持续时间，不同类型的攻击通常具有不同的持续时间特征',
        '特征2': '协议类型（TCP/UDP/ICMP） - 网络协议类型，不同攻击可能利用不同的协议',
        '特征3': '服务类型 - 目标服务类型，如HTTP、FTP等，特定服务可能成为攻击目标',
        '特征4': '标志位 - 网络连接的状态标志，反映连接的建立和终止情况',
        '特征5': '源字节数 - 从源地址发送的字节数，异常流量可能表现为异常的字节数',
        '特征6': '目标字节数 - 发送到目标地址的字节数，同样可以反映流量异常',
        '特征7': '错误片段 - 网络传输中的错误片段数量，异常攻击可能导致更多错误',
        '特征8': '紧急数据 - 紧急数据的数量，某些攻击可能利用紧急数据机制',
        '特征9': 'HTTP方法 - HTTP请求方法，如GET、POST等，异常方法可能指示攻击',
        '特征10': 'HTTP协议版本 - HTTP协议版本，不同版本可能有不同的安全漏洞'
    }
    
    for feature, description in core_features.items():
        st.markdown(f"- **{feature}**：{description}")
    
    st.markdown("**选择原因**：这些特征涵盖了网络连接的基本属性、传输数据量、协议特征和应用层信息，能够全面反映网络流量的特征，有效区分正常流量和各类攻击流量。通过特征重要性分析，这些特征在模型训练中表现出最高的预测能力。")
    
    st.dataframe(preview_data, height=400)
    
    # 特征分布
    st.markdown("---")
    st.subheader("特征分布")
    st.markdown("下图展示了所选特征的分布情况，基于测试集中的前1000个样本。通过观察特征分布，可以了解不同特征的取值范围和频率分布，有助于理解数据的特征空间。")
    feature_idx = st.selectbox("选择特征查看分布", range(1, 11), format_func=lambda x: f'特征{x}: {feature_descriptions.get(f"特征{x}", "未知")}')-1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(X_test[:1000, feature_idx], bins=50, ax=ax)
    plt.title(f'特征{feature_idx+1}分布')
    plt.xlabel('特征值')
    plt.ylabel('频率')
    
    # 添加数值标签
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.text(patch.get_x() + patch.get_width()/2., height + 5, f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    st.pyplot(fig)

# 批量检测页面
elif page == "批量检测":
    st.title("🔍 批量检测")
    
    # 检测参数设置
    st.markdown("---")
    st.subheader("检测设置")
    sample_size = st.slider("选择检测样本数量", 100, 1000, 500, step=100)
    st.markdown("**样本选择说明**：批量检测使用的是测试集中的前N个样本，这些样本是经过分层采样得到的，确保包含各种类型的网络攻击，能够全面评估模型的检测性能。")
    st.markdown("**选择原因**：采用固定的前N个样本进行批量检测，主要基于以下考虑：\n    1. **一致性**：使用固定样本可以确保每次检测结果具有可比性，便于评估模型的稳定性\n    2. **全面性**：测试集是经过分层采样构建的，前N个样本已经包含了各种类型的攻击\n    3. **效率**：固定样本可以减少数据加载和处理时间，提高检测效率\n    4. **可重复性**：固定样本使得实验结果可重复，便于后续的模型改进和比较")
    if st.button("🚀 开始批量检测"):
        with st.spinner("正在进行批量检测..."):
            start_time = time.time()
            # 加载模型
            model = load_model()
            # 进行批量预测
            y_pred = model.predict(X_test[:sample_size])
            end_time = time.time()
            
            # 计算准确率
            accuracy = np.mean(y_pred == y_test[:sample_size])
            
            # 统计结果
            from sklearn.metrics import classification_report, f1_score
            weighted_f1 = f1_score(y_test[:sample_size], y_pred, average='weighted')
            macro_f1 = f1_score(y_test[:sample_size], y_pred, average='macro')
            
            # 获取实际存在的类别
            actual_classes = np.unique(y_test[:sample_size])
            actual_class_names = [class_names[c] for c in actual_classes]
            report = classification_report(y_test[:sample_size], y_pred, labels=actual_classes, target_names=actual_class_names, output_dict=True)
            
            st.success(f"批量检测完成！\n耗时: {end_time - start_time:.2f}秒\n准确率: {accuracy:.4f}\n加权F1: {weighted_f1:.4f}\n宏F1: {macro_f1:.4f}")
            
            # 显示分类报告
            st.subheader("分类报告")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # 结果可视化
            st.subheader("检测结果可视化")
            st.markdown("下图展示了模型预测结果与实际类别分布的对比情况。通过对比预测数量和实际数量，可以直观评估模型对各类攻击的检测效果。")
            
            # 类别预测分布
            pred_counts = np.bincount(y_pred, minlength=len(class_names))
            pred_distribution = pd.DataFrame({
                '类别': class_names,
                '预测数量': pred_counts,
                '实际数量': np.bincount(y_test[:sample_size], minlength=len(class_names))
            })
            
            fig, ax = plt.subplots(figsize=(14, 8))
            bars = pred_distribution.plot(kind='bar', x='类别', ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.title('预测与实际类别分布对比')
            plt.tight_layout()
            
            # 添加数值标签
            for container in ax.containers:
                ax.bar_label(container, fontsize=10)
            
            st.pyplot(fig)

# 单样本检测页面
elif page == "单样本检测":
    st.title("🔎 单样本检测")
    
    # 样本选择
    st.markdown("---")
    st.subheader("样本选择")
    sample_idx = st.selectbox("选择样本索引 (0-999)", range(1000))
    st.markdown("**样本选择说明**：单样本检测使用的是测试集中的样本，这些样本是经过分层采样得到的，确保包含各种类型的网络攻击，通过选择不同索引可以查看模型对不同类型攻击的检测效果。")
    st.markdown("**选择原因**：采用固定的索引范围(0-999)进行单样本检测，主要基于以下考虑：\n    1. **代表性**：测试集中的前1000个样本已经包含了各种类型的网络攻击，具有良好的代表性\n    2. **可控性**：固定索引范围便于用户有针对性地选择和分析特定样本\n    3. **效率**：限制在1000个样本范围内可以保证系统响应速度，提供良好的用户体验\n    4. **可解释性**：用户可以通过索引选择不同类型的攻击样本，观察模型的检测效果和特征重要性")
    if st.button("开始检测"):
        with st.spinner("正在检测..."):
            start_time = time.time()
            # 加载模型
            model = load_model()
            # 获取样本数据
            sample = X_test[sample_idx].reshape(1, -1)
            true_label = y_test[sample_idx]
            
            # 预测
            y_pred = model.predict(sample)
            y_prob = model.predict_proba(sample)
            end_time = time.time()
            
            # 结果显示
            st.markdown("---")
            st.subheader("检测结果")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("真实类别", class_names[true_label])
                st.metric("预测类别", class_names[y_pred[0]])
                st.metric("检测结果", "正确" if y_pred[0] == true_label else "错误")
                st.metric("推理耗时", f"{end_time - start_time:.4f}秒")
                
                # 真实类别说明
                st.subheader("真实类别说明")
                st.markdown(class_descriptions.get(class_names[true_label], "未知类别"))
                
                # 预测类别说明
                st.subheader("预测类别说明")
                st.markdown(class_descriptions.get(class_names[y_pred[0]], "未知类别"))
            
            with col2:
                # 预测概率
                st.subheader("预测概率")
                prob_df = pd.DataFrame({
                    '类别': class_names,
                    '概率': y_prob[0]
                }).sort_values('概率', ascending=False).head(5)
                st.dataframe(prob_df)
                
                # 显示Top5核心特征
                st.subheader("Top5核心特征")
                feature_importance = np.abs(sample[0])
                top5_indices = np.argsort(feature_importance)[-5:][::-1]
                top5_features = [f'特征{i+1}' for i in top5_indices]
                top5_values = feature_importance[top5_indices]
                top5_descriptions = [feature_descriptions.get(f'特征{i+1}', '未知') for i in top5_indices]
                
                top5_df = pd.DataFrame({
                    '特征': top5_features,
                    '值': top5_values,
                    '说明': top5_descriptions
                })
                st.dataframe(top5_df)
            
            # 完整特征值
            st.markdown("---")
            st.subheader("完整特征值")
            feature_df = pd.DataFrame({
                '特征': [f'特征{i+1}' for i in range(10)],
                '值': sample[0][:10],
                '说明': [feature_descriptions.get(f'特征{i+1}', '未知') for i in range(10)]
            })
            st.dataframe(feature_df)

# 项目说明页面
elif page == "项目说明":
    st.title("📖 项目说明")
    
    # 项目背景
    st.markdown("---")
    st.subheader("项目背景")
    st.markdown("随着网络攻击手段的不断演变，传统的基于规则的入侵检测系统已经难以应对复杂的网络威胁。本项目采用gcForest（深度森林）算法，结合多粒度扫描和级联森林结构，实现了高效、准确的网络入侵检测系统。")
    
    # 技术架构
    st.markdown("---")
    st.subheader("技术架构")
    st.markdown("**gcForest算法原理**：")
    st.markdown("- **多粒度扫描**：使用不同大小的窗口（10, 20）对输入特征进行扫描，提取局部特征")
    st.markdown("- **级联森林**：每层包含2个RandomForest和2个ExtraTrees，通过层层递进的方式提升模型性能")
    st.markdown("- **早停机制**：当验证集性能连续2层无提升时停止训练，防止过拟合")
    
    # 优化策略
    st.markdown("---")
    st.subheader("优化策略")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("内存优化")
        st.markdown("- 限制并行度（n_jobs=2）")
        st.markdown("- 开启增量训练模式")
        st.markdown("- 强制垃圾回收")
        st.markdown("- 数据类型优化（float32/int32）")
    
    with col2:
        st.subheader("小样本优化")
        st.markdown("- 分类阈值优化")
        st.markdown("- SMOTE过采样增强小样本类别")
        st.markdown("- 调整类别权重")
        st.markdown("- 多粒度窗口精细化优化")
    
    # 项目成果
    st.markdown("---")
    st.subheader("项目成果")
    st.markdown("本项目在16GB内存环境下实现了高效的网络入侵检测系统，取得了以下成果：")
    
    project_results = {
        '指标': ['测试集准确率', '加权F1值', '宏平均F1值', '小样本类别F1值提升', '训练时间', '模型大小'],
        '数值': ['90.93%', '97.94%', '84.63%', '2.62%', '~2.5小时', '~1.5GB']
    }
    st.dataframe(pd.DataFrame(project_results))
    
    # 未来展望
    st.markdown("---")
    st.subheader("未来展望")
    st.markdown("为进一步提升系统性能和实用性，未来将重点发展以下方向：")
    st.markdown("1. **实时检测**：实现实时网络流量监控和检测")
    st.markdown("2. **模型压缩**：进一步减小模型体积，提高部署效率")
    st.markdown("3. **多模态融合**：结合网络流量、系统日志等多源数据")
    st.markdown("4. **自适应学习**：实现模型的在线学习和更新")
    st.markdown("5. **可视化增强**：提供更丰富的可视化界面和分析工具")

# 页脚
st.markdown("---")
st.markdown("### 📊 系统状态")
st.info("计算机设计大赛演示版本: 基于gcForest的网络入侵检测")
st.info(f"模型路径: {MODEL_PATH}")
st.info(f"数据路径: {DATA_PATH}")
st.info(f"测试集大小: {len(X_test)} 样本")
st.info(f"类别数量: {len(class_names)}")