set -euo pipefail
cd /workspace/torchtitan
export PYTHONPATH=/workspace/torchtitan GLM53_FLASH_EXPERTS_IMPL=grouped
for leg in base seed42 seed43; do
  adapter=""; cond=base
  if [ $leg != base ]; then adapter="--adapter /workspace/torchtitan/experiments/glm53_flash_sft_ladder/results/t08f-20261014T213158Z/attempt-01/adapters/$leg"; cond=adapter; fi
  python -u -m torchtitan.experiments.glm53_flash.reference_eval --checkpoint /project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16 $adapter     --problems /project/formal-cache/datasets/countdown-sft/v1/dev/problems.jsonl --output-dir /workspace/torchtitan/experiments/glm53_flash_sft_ladder/results/t08f-20261014T213158Z/attempt-01/eval_reference/dev/$leg     --model-id $leg --condition $cond --limit 100 --num-samples 1 --temperature 0 --max-new-tokens 192     --batch-size 25
done
