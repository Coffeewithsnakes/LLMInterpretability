# Model Selection Guide

## Current: GPT-2 Small (124M parameters)

**Pros:**
- Fast to download and run
- Well-studied in interpretability research
- Guaranteed to fit on any hardware

**Cons:**
- Very small and fragile
- Circuit interventions often break coherence
- Limited language understanding

**Issues you're seeing:**
- Amplified circuits → repetition ("comfortable comfortable...")  
- Dampened circuits → gibberish (random tokens)
- Poor baseline quality

---

## Recommended: Models for 3070 (8GB VRAM)

### 1. **GPT-2 Medium (355M)** - RECOMMENDED FIRST UPGRADE

```python
detector = EmotionCircuitDetector(
    model_name="gpt2-medium",
    device="cuda"
)
```

**Size:** ~1.4 GB  
**VRAM:** ~3-4 GB  
**Quality:** 2.8x more parameters than small  
**Pros:**
- Much more robust to interventions
- Better baseline text quality
- Still fast (< 5 min for circuit discovery)
- Well-supported in TransformerLens

**Cons:**
- Still GPT-2 architecture (dated)
- Not state-of-the-art

**Verdict:** ✅ Best immediate upgrade for your use case

---

### 2. **GPT-2 Large (774M)**

```python
detector = EmotionCircuitDetector(
    model_name="gpt2-large",
    device="cuda"
)
```

**Size:** ~3 GB  
**VRAM:** ~5-6 GB  
**Quality:** 6.2x more parameters than small  
**Pros:**
- Even more robust
- Better coherence
- Still fits comfortably on 3070

**Cons:**
- Slower than medium
- Still GPT-2 limitations

**Verdict:** ✅ Good if medium still has issues

---

### 3. **Pythia-1B** - BEST FOR RESEARCH

```python
detector = EmotionCircuitDetector(
    model_name="EleutherAI/pythia-1b",
    device="cuda"
)
```

**Size:** ~2 GB  
**VRAM:** ~4-5 GB  
**Quality:** Modern architecture, 1B parameters  
**Pros:**
- **Designed specifically for interpretability research**
- Trained with detailed logging
- Multiple checkpoints at different training stages
- Better architecture than GPT-2
- Strong TransformerLens support

**Cons:**
- Less well-known than GPT-2
- Slightly slower loading

**Verdict:** ✅✅ Best choice for serious interpretability work

---

### 4. **Pythia-1.4B**

```python
detector = EmotionCircuitDetector(
    model_name="EleutherAI/pythia-1.4b",
    device="cuda"
)
```

**Size:** ~2.8 GB  
**VRAM:** ~5-6 GB  
**Quality:** 1.4B parameters  
**Pros:**
- Even better than 1B
- Still fits on 3070
- Very robust to interventions

**Cons:**
- Slower than 1B

**Verdict:** ✅ If you want max quality while staying safe

---

### 5. **GPT-2 XL (1.5B)** - BORDERLINE

```python
detector = EmotionCircuitDetector(
    model_name="gpt2-xl",
    device="cuda"
)
```

**Size:** ~6 GB  
**VRAM:** ~7-8 GB (TIGHT!)  
**Quality:** 1.5B parameters  
**Pros:**
- Largest GPT-2
- Best GPT-2 quality

**Cons:**
- **Might not fit on 3070 with 8GB**
- Need to close other programs
- Risk of OOM errors

**Verdict:** ⚠️ Risky on 3070, but possible if you're careful

---

## Advanced: Larger Models (Requires Quantization)

### Llama-2-7B (with 8-bit quantization)

```python
# Requires bitsandbytes library
# pip install bitsandbytes

from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)
```

**Size:** ~7 GB (quantized from 14 GB)  
**VRAM:** ~7-8 GB  
**Quality:** State-of-the-art  
**Pros:**
- Much better language understanding
- More robust circuits
- Modern architecture

**Cons:**
- Requires quantization
- Slower inference
- Not fully tested with TransformerLens
- Needs HuggingFace account + Llama access

**Verdict:** ⚠️ Advanced users only, requires extra setup

---

## Comparison Table

| Model | Params | VRAM | Speed | Robustness | Ease | Best For |
|-------|--------|------|-------|------------|------|----------|
| gpt2-small | 124M | 1-2GB | ⚡⚡⚡ | ⭐ | ✅✅✅ | Testing/Learning |
| **gpt2-medium** | 355M | 3-4GB | ⚡⚡ | ⭐⭐⭐ | ✅✅✅ | **First upgrade** |
| gpt2-large | 774M | 5-6GB | ⚡ | ⭐⭐⭐⭐ | ✅✅ | Better quality |
| **pythia-1b** | 1B | 4-5GB | ⚡ | ⭐⭐⭐⭐ | ✅✅ | **Research** |
| pythia-1.4b | 1.4B | 5-6GB | ⚡ | ⭐⭐⭐⭐⭐ | ✅✅ | Max quality |
| gpt2-xl | 1.5B | 7-8GB | ⚡ | ⭐⭐⭐⭐⭐ | ⚠️ | If you're brave |
| llama-2-7b-8bit | 7B | 7-8GB | 🐌 | ⭐⭐⭐⭐⭐ | ⚠️⚠️ | Advanced |

---

## My Recommendation for Your 3070

### Path 1: Safe and Easy
```
1. Start with gpt2-medium (guaranteed to work)
2. If still having issues → pythia-1b
3. If want even better → pythia-1.4b
```

### Path 2: Maximum Quality
```
1. Go straight to pythia-1.4b
2. If VRAM issues → pythia-1b
3. If still issues → gpt2-large
```

---

## How to Switch Models

### In Python Code:
```python
# Old
detector = EmotionCircuitDetector(model_name="gpt2-small", device="cuda")

# New - just change the name!
detector = EmotionCircuitDetector(model_name="gpt2-medium", device="cuda")
# or
detector = EmotionCircuitDetector(model_name="EleutherAI/pythia-1b", device="cuda")
```

### In Interactive Menu:

We can add a model selection option to the menu! Let me know if you want that.

---

## Monitoring VRAM Usage

```bash
# On Linux/Windows with nvidia-smi
watch -n 1 nvidia-smi

# In Python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

---

## Expected Improvements

### Current (GPT-2 Small):
```
Normal: "I feel bad for those who voted for Hillary..."
Amplified: "comfortable comfortable comfortable comfortable..." ❌ BROKEN
Dampened: ",,,--, 11,n,by inno..." ❌ BROKEN
```

### With GPT-2 Medium (expected):
```
Normal: "I feel excited about the possibilities ahead..."
Amplified: "I feel absolutely wonderful and overjoyed about..." ✅ COHERENT
Dampened: "I feel somewhat neutral about the situation..." ✅ COHERENT
```

### With Pythia-1B (expected):
```
Normal: "I feel optimistic about the future and ready to..."
Amplified: "I feel incredibly happy and enthusiastic about..." ✅ VERY COHERENT
Dampened: "I feel calm and measured about these events..." ✅ VERY COHERENT
```

---

## TransformerLens Model Support

Check supported models:
```python
from transformer_lens import HookedTransformer
print(HookedTransformer.get_model_list())
```

All GPT-2 and Pythia models listed above are fully supported!

---

## Next Steps

1. **Try gpt2-medium first** - safest upgrade
2. **If that works well, try pythia-1b** - best for research
3. **Enable smart bounds** - already done! (prevents total collapse)
4. **Adjust intervention intensity** - try 1.2-1.5 instead of 2.0 for amplify

Let me know which model you want to try and I can help optimize settings!
