#!/bin/bash
#########################################################################
# MER-UniBench 9个数据集 Emotion_Peak Frame特征批量预提取脚本
#########################################################################

echo "======================================================================="
echo "🚀 MER-UniBench Frame Emotion_Peak Feature Extraction"
echo "======================================================================="
echo ""

# 配置参数
OUTPUT_ROOT="../preextracted_features"  # 相对于MER-UniBench目录
MER_FACTORY_OUTPUT="/home/project/MER-Factory/output"
VISUAL_ENCODER="CLIP_VIT_LARGE"
N_FRMS=8
DEVICE="cuda:0"

# 数据集列表
DATASETS=(
    "cmumosei"
    "cmumosi"
    "iemocap"
    "meld"
    "mer2023"
    "mer2024"
    "ovmerdplus"
    "sims"
    "simsv2"
)

echo "📊 将处理以下数据集:"
for dataset in "${DATASETS[@]}"; do
    echo "  - $dataset"
done
echo ""

# 检查MER-Factory输出目录
if [ ! -d "$MER_FACTORY_OUTPUT" ]; then
    echo "❌ MER-Factory output directory not found: $MER_FACTORY_OUTPUT"
    echo "   emotion_peak采样需要MER-Factory生成的au_info"
    echo ""
    echo "请先运行MER-Factory处理这些数据集:"
    echo "  cd /home/project/MER-Factory"
    echo "  python main.py --dataset <dataset_name> --modality video"
    echo ""
    exit 1
fi

# 检查哪些数据集已有MER-Factory输出
echo "🔍 检查MER-Factory输出..."
for dataset in "${DATASETS[@]}"; do
    # 根据数据集名称映射MER-Factory目录名（实际文件夹名）
    case $dataset in
        "cmumosei") mer_dir="CMUMOSEI" ;;
        "cmumosi")  mer_dir="CMUMOSI" ;;
        "iemocap")  mer_dir="IEMOCAPFour" ;;
        "meld")     mer_dir="MELD" ;;
        "mer2023")  mer_dir="MER2023" ;;
        "mer2024")  mer_dir="MER2024" ;;
        "ovmerdplus") mer_dir="OVMERDPlus" ;;
        "sims")     mer_dir="SIMS" ;;
        "simsv2")   mer_dir="SIMSv2" ;;
        *)          mer_dir="$dataset" ;;
    esac
    
    if [ -d "$MER_FACTORY_OUTPUT/$mer_dir" ]; then
        sample_count=$(find "$MER_FACTORY_OUTPUT/$mer_dir" -name "*_au_analysis.json" 2>/dev/null | wc -l)
        echo "  ✅ $dataset: $sample_count 个样本已处理"
    else
        echo "  ⚠️  $dataset: MER-Factory输出不存在"
    fi
done
echo ""

# 询问是否继续
read -p "是否继续提取特征? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 运行特征提取
echo ""
echo "======================================================================="
echo "⏳ 开始提取特征..."
echo "======================================================================="
echo ""

START_TIME=$(date +%s)

python3 extract_frame_emotion_peak_batch.py \
    --datasets "${DATASETS[@]}" \
    --output-root "$OUTPUT_ROOT" \
    --mer-factory-output "$MER_FACTORY_OUTPUT" \
    --visual-encoder "$VISUAL_ENCODER" \
    --n-frms "$N_FRMS" \
    --device "$DEVICE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "======================================================================="
echo "✅ 批量提取完成!"
echo "======================================================================="
printf "⏱️  总耗时: %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS
echo ""

# 检查生成的特征文件
echo "📊 特征文件统计:"
for dataset in "${DATASETS[@]}"; do
    feat_dir="$OUTPUT_ROOT/$dataset/frame_${VISUAL_ENCODER}_emotion_peak_${N_FRMS}frms"
    if [ -d "$feat_dir" ]; then
        count=$(ls "$feat_dir"/*.npy 2>/dev/null | wc -l)
        size=$(du -sh "$feat_dir" 2>/dev/null | cut -f1)
        echo "  - $dataset: $count 个文件, $size"
    fi
done
echo ""

echo "💡 使用方法:"
echo "   在eval_configs/*.yaml中配置:"
echo "   datasets:"
echo "     <dataset_name>:"
echo "       frame_sampling: 'emotion_peak'"
echo "       use_preextracted_features: True"
echo "       preextracted_root: './preextracted_features/<dataset_name>'"
echo ""
