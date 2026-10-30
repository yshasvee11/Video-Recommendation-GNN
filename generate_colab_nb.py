import json
import os

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Vynora 2.0: Denoising & Vlogger-Aware Meta-Path Graph Contrastive Learning\n",
                "\n",
                "### Google Colab Training Notebook (T4 GPU)\n",
                "This notebook trains the full **Vynora 2.0** recommendation model on a Google Colab T4 GPU with **zero compromise on quality**.\n",
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
                "## 2. Extract Project and Dataset\n",
                "Upload `vynora2_colab_bundle.zip` (only 11 MB) to the Colab files panel on the left, then run the cell below."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Unzip the bundle into /content\n",
                "import os\n",
                "if os.path.exists('/content/vynora2_colab_bundle.zip'):\n",
                "    !unzip -q /content/vynora2_colab_bundle.zip -d /content/\n",
                "    print('Unzipped vynora2_colab_bundle.zip successfully!')\n",
                "    !ls -la /content/vynora2\n",
                "    !ls -la /content/VAgnn-dataset\n",
                "else:\n",
                "    print('Please upload vynora2_colab_bundle.zip to /content/ via the Colab files panel!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install requirements\n",
                "!pip install -q scipy numpy torch tqdm"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Train Vynora 2.0 (Full Quality, No Compromises)\n",
                "Runs full BPR dual-stream training, cross-view InfoNCE contrastive learning, and CONDE iterative graph denoising.\n",
                "You can train on **Takatak** (default) or **WeChat** using the `--dataset` flag.\n",
                "Checkpoints will be saved automatically to `checkpoints/best_<dataset>.pt` on validation improvements."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%cd /content/vynora2\n",
                "# Train on WeChat dataset (patience=10)\n",
                "!python train.py --dataset wechat\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Home Screen Video Recommendations\n",
                "Predict top-K personalized video recommendations for any user ID using the trained model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Recommend top-10 videos on WeChat for user 42\n",
                "!python predict.py --user 42 --topk 10 --dataset wechat\n",
                "\n",
                "# Try another user (e.g. user 100)\n",
                "!python predict.py --user 100 --topk 10 --dataset wechat"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Paper 5 Diversity Recommendation & Benchmark\n",
                "Evaluate and demonstrate Paper 5 diversity recommendation (Chun et al., ESWA 2024).\n",
                "- Eliminates creator redundancy using Publisher and Content Diversity metrics (Eq 1 & 2).\n",
                "- Re-ranks candidate videos using Maximal Marginal Relevance (MMR) or Score Blending."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Diverse recommendation for User 42 (lambda=0.4 MMR)\n",
                "!python predict.py --user 42 --topk 10 --dataset wechat --diversity 0.4\n",
                "\n",
                "# Run full accuracy vs diversity trade-off benchmark across lambda values\n",
                "!python diversity_benchmark.py --dataset wechat --mode val"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Run Full 6-Variant Ablation Study (Optional)\n",
                "Evaluates variants A through F to prove the contribution of each component."
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
                "## 7. Download Checkpoints and Results\n",
                "Download the best model checkpoint and evaluation results back to your machine."
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
