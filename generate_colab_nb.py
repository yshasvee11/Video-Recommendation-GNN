import json
import os

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Video-Recommendation-GNN (Vynora 2.0)\n",
                "### Denoising & Vlogger-Aware Meta-Path Graph CL with Paper 5 Diversity Recommendation\n",
                "\n",
                "Google Colab T4 GPU execution notebook.\n",
                "\n",
                "**Hardware Acceleration Check:**\n",
                "Go to **Runtime** > **Change runtime type** > Select **T4 GPU** > Click **Save**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Verify T4 GPU is active\n",
                "!nvidia-smi\n",
                "import torch\n",
                "print('CUDA Available:', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('Device:', torch.cuda.get_device_name(0))\n",
                "    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                "    print(f'VRAM: {vram:.2f} GB')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Setup Codebase and Dataset\n",
                "Clones the repository and automatically prepares the takatak and wechat benchmark datasets."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "# 1. Clone repository if running in fresh Colab session\n",
                "if not os.path.exists('/content/Video-Recommendation-GNN'):\n",
                "    !git clone https://github.com/yshasvee11/Video-Recommendation-GNN.git /content/Video-Recommendation-GNN\n",
                "%cd /content/Video-Recommendation-GNN\n",
                "\n",
                "# 2. Download VA-GNN tripartite datasets (takatak & wechat)\n",
                "if not os.path.exists('/content/VAgnn-dataset'):\n",
                "    !git clone --depth 1 https://github.com/laiweijiang/VAGNN.git /content/vagnn_raw\n",
                "    !mv /content/vagnn_raw/data /content/VAgnn-dataset\n",
                "    !rm -rf /content/vagnn_raw\n",
                "    print('Dataset ready in /content/VAgnn-dataset!')\n",
                "\n",
                "# 3. Install requirements\n",
                "!pip install -q -r requirements.txt\n",
                "print('Environment initialized successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Train Model\n",
                "Runs dual-view GNN propagation, cross-view InfoNCE contrastive learning, and CONDE iterative graph denoising."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train on Takatak dataset (switch to --dataset wechat if desired)\n",
                "!python train.py --dataset takatak"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Paper 5 Diversity Recommendation\n",
                "Generate top-K recommendations for user 42 with custom diversity trade-off $\\lambda \\in [0, 1]$.\n",
                "- `lambda=0.0`: pure relevance baseline (can have 5+ videos from same vlogger)\n",
                "- `lambda=0.4`: Paper 5 MMR diversification (maximizes creator variety & content diversity)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Baseline recommendations (lambda=0.0)\n",
                "!python predict.py --user 42 --topk 10 --dataset takatak --diversity 0.0\n",
                "\n",
                "# Paper 5 Diverse recommendations (lambda=0.4, MMR)\n",
                "!python predict.py --user 42 --topk 10 --dataset takatak --diversity 0.4"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Accuracy vs. Diversity Benchmark\n",
                "Evaluate the Pareto trade-off across $\\lambda \\in [0.0, 1.0]$ on the validation set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python diversity_benchmark.py --dataset takatak --mode val"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Run 6-Variant Ablation Study (Optional)\n",
                "Evaluates variants A through F to isolate the contribution of each paper's module."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python ablation.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Download Checkpoints and Benchmark Results"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import files\n",
                "!zip -r vynora2_trained_results.zip checkpoints/ results/\n",
                "files.download('vynora2_trained_results.zip')"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "vynora2_colab.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated Colab notebook at: {out_path}")
