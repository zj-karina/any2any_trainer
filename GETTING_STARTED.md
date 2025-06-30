# Getting Started with Any2Any Trainer

Quick guide to get you up and running with Any2Any Trainer in minutes!

## 🚀 Quick Installation & Test

1. **Install dependencies:**
   ```bash
   cd any2any_trainer
   pip install -e .
   ```

2. **Test the installation:**
   ```bash
   python test_installation.py
   ```

3. **Run your first training:**
   ```bash
   python scripts/train_multimodal.py --config configs/test/minimal_test.yaml
   ```

## 📋 What You Need

- **Python 3.8+**
- **PyTorch** (with CUDA support recommended)
- **Internet connection** (for downloading models and datasets)
- **~2GB free disk space** (for model cache)

## 🎯 First Training Run

The minimal test configuration will:
- Download GPT-2 model (small, ~500MB)
- Use WikiText-2 dataset (public, small)
- Train for 1 epoch with LoRA (memory efficient)
- Save results to `outputs/test_run/`

Expected time: 5-10 minutes on CPU, 1-2 minutes on GPU.

## 🔧 Common Issues

**Import errors?**
```bash
pip install torch transformers datasets peft accelerate
```

**CUDA out of memory?**
- Reduce `batch_size` in config (try `batch_size: 1`)
- Enable `fp16: true` in config
- Use `use_deepspeed: true` for very large models

**Model download fails?**
- Check internet connection
- Some models may be gated (need HuggingFace login)
- Try a different model like `"microsoft/DialoGPT-small"`

## 📚 Next Steps

1. **Explore configurations:** Check `configs/` directory for more examples
2. **Try different models:** Edit `model_name` in your config
3. **Use your own data:** See `HF_MODELS_USAGE.md` for dataset formats
4. **Scale up:** Enable DeepSpeed/Accelerate for multi-GPU training

## 💡 Quick Tips

- Start with small models (`gpt2`, `microsoft/DialoGPT-small`)
- Always use LoRA (`use_peft: true`) for memory efficiency
- Enable FP16 (`fp16: true`) to halve memory usage
- Check logs in `outputs/*/logs/` if something goes wrong

Happy training! 🎉 