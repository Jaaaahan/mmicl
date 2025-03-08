huggingface-cli download --repo-type dataset --resume-download --local-dir-use-symlinks False openflamingo/eval_benchmark --token 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 --local-dir ./datasets/

nohup python main.py --batch_size 64 --dataset coco --model idefics_9b_base --labeling truth --num_shots 5 --sampling random --ordering leave --limit_support_set 5000 --tag experiment1 --support_set random --seed 42 --num_runs 3
python main.py --batch_size 32 --dataset coco --model idefics_9b_base --labeling truth --num_shots 5 --sampling random --ordering leave --limit_support_set 5000 --tag experiment1 --support_set random --seed 42 --num_runs 3

pip install numpy==1.23.5


huggingface-cli download --resume-download --local-dir-use-symlinks False llava-hf/llava-1.5-7b-hf --local-dir ./models--llava-hf--llava-1.5-7b-hf
